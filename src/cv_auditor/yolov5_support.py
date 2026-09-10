"""
yolov5_support.py
------------------
Loads a YOLOv5 object-detection checkpoint (Ultralytics YOLOv5, normal
``.pt`` training/release format) and adapts it to the torchvision
detection API that the rest of this project -- ``InferenceEngine``,
Phase 3's behavioral comparison, IoU matching, suspicion analysis --
already speaks:

    model(list_of_CHW_float_tensors) -> [{"boxes", "labels", "scores"}, ...]

Nothing downstream needs to know it is talking to YOLOv5.

Why an adapter (and not a rewrite)
----------------------------------
YOLOv5 differs from the torchvision detectors in three ways that all get
normalized here, at the boundary:

1. **Checkpoint format.** A YOLOv5 ``.pt`` is a pickled dict containing
   the *full model object* (``{"model": DetectionModel, "ema": ...}``),
   whose classes live in the YOLOv5 repository's ``models``/``utils``
   packages. Loading it therefore requires the local YOLOv5 source tree
   on ``sys.path`` (see ``find_yolov5_repo``); without it, torch raises
   the classic ``No module named 'models'``.
2. **Pre/post-processing.** YOLOv5 expects letterboxed fixed-stride
   input and emits raw ``[cx, cy, w, h, obj, 80 class scores]`` rows
   that need confidence filtering + NMS + un-letterboxing. The adapter
   does all of that internally (NMS via ``torchvision.ops``, so no
   extra YOLOv5 utility imports are needed at inference time).
3. **Class space.** YOLOv5 COCO models predict 80 classes indexed 0-79
   with NO background slot ("person" is id 0), unlike the 91-entry
   background-first list used by the torchvision COCO models. The
   adapter exposes the model's own names as ``.class_names`` and
   callers (ModelLoader / ModelAuditor) use those instead of assuming
   the Faster R-CNN label space.

Checkpoint trust & safe loading
--------------------------------
Full-model YOLOv5 checkpoints are arbitrary pickles, so they can only be
loaded with ``torch.load(..., weights_only=False)``. That flag is passed
EXPLICITLY and ONLY for the user-supplied checkpoint file here -- no
global PyTorch safety mechanism is altered, no ``add_safe_globals``
blanket allow-listing, no monkey-patching of ``torch.load``. The audit
tool's threat model already assumes the *auditor's own* reference and
candidate files are trusted local artifacts (their integrity is exactly
what the SHA-256 fingerprinting layer checks); never point it at a
checkpoint from an untrusted source.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .utils import get_logger


@contextlib.contextmanager
def _posix_path_load_compat():
    """Allow unpickling checkpoints that embed ``PosixPath`` objects on Windows.

    Some YOLOv5 training checkpoints (as opposed to the plain Ultralytics
    release weights) carry extra bookkeeping in the pickle -- e.g. an
    ``opt.save_dir`` -- that was recorded as a ``pathlib.PosixPath`` when
    the checkpoint was produced on Linux/macOS. Python's unpickler
    reconstructs that exact class by name, and on Windows
    ``pathlib.PosixPath`` refuses to instantiate at all (it exists only
    as a placeholder), raising
    ``UnsupportedOperation: cannot instantiate 'PosixPath' on your system``.

    This is a narrow, load-time-only compatibility shim: it temporarily
    points ``pathlib.PosixPath`` at ``pathlib.WindowsPath`` so the
    unpickler can build a (Windows-flavored) stand-in object instead of
    crashing, then restores the original class immediately afterwards --
    on every platform, and even if loading raises. It does not touch
    ``torch.load``'s ``weights_only`` safety behavior at all; it only
    patches how one unrelated stdlib class name resolves during this
    single call.
    """
    if os.name != "nt":
        # Only Windows refuses to instantiate PosixPath; elsewhere this
        # is a no-op so behavior is unchanged on Linux/macOS.
        yield
        return

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path
logger = get_logger(__name__)

# Environment variable that can point at a YOLOv5 source checkout, for
# deployments that keep it somewhere other than third_party/yolov5.
YOLOV5_DIR_ENV = "CV_AUDITOR_YOLOV5_DIR"

# Standard 80-class COCO name list used by official YOLOv5 models
# (index 0 = "person"; NO background entry). Only used as a fallback if
# a checkpoint somehow carries no usable ``names`` attribute -- normally
# the names stored inside the checkpoint itself are authoritative.
YOLOV5_COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Inference defaults matching YOLOv5's own detect.py conventions.
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45
DEFAULT_IMG_SIZE = 640
MAX_DETECTIONS = 300      # cap after NMS (YOLOv5 max_det default)
MAX_NMS_CANDIDATES = 30000  # cap before NMS (YOLOv5 max_nms default)


def find_yolov5_repo() -> Path:
    """Locate the local YOLOv5 source tree.

    Search order:
      1. ``$CV_AUDITOR_YOLOV5_DIR`` (explicit override),
      2. ``<project_root>/third_party/yolov5`` (the vendored copy).

    A directory qualifies if it contains ``models/yolo.py``.
    """
    candidates = []
    env_dir = os.environ.get(YOLOV5_DIR_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    # src/cv_auditor/yolov5_support.py -> parents[2] == project root
    candidates.append(Path(__file__).resolve().parents[2] / "third_party" / "yolov5")

    for candidate in candidates:
        if (candidate / "models" / "yolo.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Local YOLOv5 implementation not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + f". Either place the YOLOv5 source at third_party/yolov5 or set "
        f"${YOLOV5_DIR_ENV} to your YOLOv5 checkout."
    )


def _ensure_yolov5_importable() -> Path:
    """Make the YOLOv5 repo's ``models``/``utils`` packages importable.

    The path is *appended* (not prepended) to ``sys.path`` so it can
    never shadow installed packages or this project's own modules; it is
    added at most once.
    """
    repo = find_yolov5_repo()
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.append(repo_str)
        logger.info("Added local YOLOv5 implementation to sys.path: %s", repo_str)
    return repo


def _xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


class YOLOv5Adapter(nn.Module):
    """Wraps a loaded YOLOv5 ``DetectionModel`` behind the torchvision
    detection interface used everywhere else in this project.

    Input : list of CHW float32 tensors in [0, 1] (any sizes) -- exactly
            what Phase 1's ``ImageLoader`` produces.
    Output: list of ``{"boxes": [N,4] xyxy in ORIGINAL image pixels,
            "labels": [N] int64 class ids (0-based, no background),
            "scores": [N] float32}`` -- exactly what ``InferenceEngine``
            consumes.
    """

    def __init__(self, yolo_model: nn.Module, class_names: List[str],
                 conf_thres: float = DEFAULT_CONF_THRES,
                 iou_thres: float = DEFAULT_IOU_THRES,
                 img_size: int = DEFAULT_IMG_SIZE):
        super().__init__()
        self.model = yolo_model
        self.class_names = list(class_names)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.img_size = int(img_size)
        stride = getattr(yolo_model, "stride", None)
        self.stride = int(stride.max()) if isinstance(stride, torch.Tensor) else 32

    # ---------------------------------------------------------------- #
    # pre-processing
    # ---------------------------------------------------------------- #
    def _letterbox(self, image: torch.Tensor):
        """Aspect-preserving resize + gray padding to a stride multiple.

        Returns (padded CHW tensor, scale ratio, (pad_left, pad_top)).
        """
        _, h, w = image.shape
        ratio = min(self.img_size / h, self.img_size / w)
        new_h, new_w = round(h * ratio), round(w * ratio)

        resized = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        pad_h = (self.stride - new_h % self.stride) % self.stride
        pad_w = (self.stride - new_w % self.stride) % self.stride
        pad_top, pad_left = pad_h // 2, pad_w // 2
        padded = torch.nn.functional.pad(
            resized,
            (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
            value=114 / 255,  # YOLOv5's canonical gray padding
        )
        return padded, ratio, (pad_left, pad_top)

    # ---------------------------------------------------------------- #
    # post-processing
    # ---------------------------------------------------------------- #
    def _postprocess(self, raw: torch.Tensor, ratio: float, pad, orig_hw):
        """conf filter -> NMS -> un-letterbox, mirroring YOLOv5's own
        ``non_max_suppression`` semantics (single-label mode) but using
        ``torchvision.ops.batched_nms``."""
        from torchvision.ops import batched_nms

        # raw: [num_candidates, 5 + num_classes] = cx,cy,w,h,obj,cls...
        obj_conf = raw[:, 4]
        cls_conf, cls_id = raw[:, 5:].max(dim=1)
        scores = obj_conf * cls_conf

        keep = scores > self.conf_thres
        boxes = _xywh_to_xyxy(raw[keep, :4])
        scores = scores[keep]
        cls_id = cls_id[keep]

        if scores.numel() > MAX_NMS_CANDIDATES:
            top = scores.topk(MAX_NMS_CANDIDATES).indices
            boxes, scores, cls_id = boxes[top], scores[top], cls_id[top]

        if scores.numel():
            keep = batched_nms(boxes, scores, cls_id, self.iou_thres)[:MAX_DETECTIONS]
            boxes, scores, cls_id = boxes[keep], scores[keep], cls_id[keep]

        # map letterboxed coords back to the original image
        pad_left, pad_top = pad
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes /= ratio
        orig_h, orig_w = orig_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

        return {
            "boxes": boxes.float(),
            "labels": cls_id.to(torch.int64),
            "scores": scores.float(),
        }

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, images: List[torch.Tensor]):
        device = next(self.model.parameters()).device
        outputs = []
        for image in images:
            _, orig_h, orig_w = image.shape
            padded, ratio, pad = self._letterbox(image.to(device=device,
                                                          dtype=torch.float32))
            raw = self.model(padded.unsqueeze(0))[0][0]  # -> [N, 5+nc]
            outputs.append(self._postprocess(raw, ratio, pad, (orig_h, orig_w)))
        return outputs


def _extract_class_names(yolo_model: nn.Module) -> List[str]:
    """Read the checkpoint's own class configuration (list or {id: name}
    dict); fall back to the standard 80-class COCO list only if absent."""
    names = getattr(yolo_model, "names", None)
    if isinstance(names, dict) and names:
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, (list, tuple)) and names:
        return [str(n) for n in names]
    logger.warning("YOLOv5 checkpoint carries no class names; "
                   "falling back to the standard COCO-80 list.")
    return list(YOLOV5_COCO_CLASS_NAMES)


def load_yolov5_model(
    checkpoint_path: str,
    device: torch.device,
    conf_thres: Optional[float] = None,
    iou_thres: float = DEFAULT_IOU_THRES,
    img_size: int = DEFAULT_IMG_SIZE,
) -> YOLOv5Adapter:
    """Load a YOLOv5 checkpoint (normal ``.pt`` format) into an adapter.

    Handles both training checkpoints (dict with ``model``/``ema``/
    ``optimizer``/... entries) and bare pickled models. ``conf_thres``
    of None uses YOLOv5's conventional 0.25 default; Phase 3 forwards
    ``detector_score_thresh`` here, mirroring how it overrides
    ``box_score_thresh`` for the torchvision architectures.
    """
    _ensure_yolov5_importable()

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLOv5 checkpoint not found: {checkpoint_path}")

    try:
        # weights_only=False is required for full-model YOLOv5 pickles and
        # is applied ONLY to this trusted local file -- see module docstring.
        # _posix_path_load_compat guards against checkpoints saved on
        # Linux/macOS that embed a PosixPath in their metadata, which
        # would otherwise crash the unpickler on Windows (see its
        # docstring above) -- it does not affect weights_only at all.
        with _posix_path_load_compat():
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:  # pragma: no cover - env guard
        raise RuntimeError(
            f"Unpickling the YOLOv5 checkpoint needs the YOLOv5 source tree "
            f"(missing module: {exc.name}). Ensure third_party/yolov5 is "
            f"present or ${YOLOV5_DIR_ENV} points at a YOLOv5 checkout."
        ) from exc

    if isinstance(ckpt, dict):
        yolo_model = ckpt.get("ema") or ckpt.get("model")
        if yolo_model is None:
            raise ValueError(
                f"'{checkpoint_path}' is a dict checkpoint without a "
                f"'model' or 'ema' entry -- not a YOLOv5 checkpoint?"
            )
    elif isinstance(ckpt, nn.Module):
        yolo_model = ckpt
    else:
        raise ValueError(
            f"'{checkpoint_path}' did not contain a YOLOv5 model "
            f"(got {type(ckpt).__name__})."
        )

    # fp16 release weights -> fp32 for CPU compatibility; eval mode.
    yolo_model = yolo_model.float().eval()

    # Standard YOLOv5 load-time compat (mirrors attempt_load): fuse
    # Conv+BN where possible and pin Detect heads to non-in-place mode so
    # post-processing sees untouched tensors.
    try:
        yolo_model = yolo_model.fuse()
    except Exception:  # pragma: no cover - fuse is an optimization only
        logger.warning("YOLOv5 fuse() failed; continuing without fusing.")
    for module in yolo_model.modules():
        if type(module).__name__ == "Detect":
            module.inplace = False

    class_names = _extract_class_names(yolo_model)
    adapter = YOLOv5Adapter(
        yolo_model,
        class_names=class_names,
        conf_thres=DEFAULT_CONF_THRES if conf_thres is None else conf_thres,
        iou_thres=iou_thres,
        img_size=img_size,
    )
    adapter.to(device)
    adapter.eval()
    logger.info(
        "YOLOv5 model loaded from %s (%d classes, conf_thres=%.3f, device=%s)",
        checkpoint_path, len(class_names), adapter.conf_thres, device,
    )
    return adapter
