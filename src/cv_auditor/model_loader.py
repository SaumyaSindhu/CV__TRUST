"""
model_loader.py
----------------
Responsible for ONE thing: turning a model specification (architecture name
+ optional checkpoint file) into a ready-to-run PyTorch ``nn.Module``, plus
a bundle of metadata describing that model.

Design notes for future phases
-------------------------------
This loader is deliberately narrow in scope for Phase 1: it supports
TorchVision object-detection architectures loaded either
    (a) with official pretrained (COCO) weights,
    (b) with a user-supplied checkpoint (state_dict), or
    (c) as a self-contained TorchScript file.

Additional formats (e.g. ONNX, custom architectures) can be added later by
introducing new branches in ``ModelLoader.load`` / new metadata fields --
the public interface (``ModelLoader.load`` -> ``(model, ModelMetadata)``)
is designed to stay stable as that happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from .config import PipelineConfig, SUPPORTED_ARCHITECTURES
from .utils import get_logger, sha256_of_file, utc_timestamp

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Descriptive information about a loaded model.

    This is intentionally rich even though Phase 1 does not *use* most of
    it for anything beyond bookkeeping -- later audit phases will rely on
    exactly this kind of metadata (e.g. to check whether a model file has
    changed, or to compare declared vs. actual parameter counts).
    """

    model_name: str
    architecture: str
    source: str  # "pretrained" | "checkpoint" | "torchscript"
    weights_path: Optional[str]
    weights_sha256: Optional[str]
    num_parameters: int
    num_classes: int
    device: str
    framework: str = "pytorch"
    framework_version: str = torch.__version__
    loaded_at: str = ""

    def __post_init__(self):
        if not self.loaded_at:
            self.loaded_at = utc_timestamp()

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _build_architecture(architecture: str, num_classes: int, pretrained: bool,
                         score_thresh: Optional[float] = None):
    """Instantiate a torchvision detection architecture by name.

    ``score_thresh``, if given, overrides the architecture's internal
    detection-score filtering (``box_score_thresh`` for Faster R-CNN
    variants, ``score_thresh`` for RetinaNet) -- see
    ``PipelineConfig.detector_score_thresh`` for why this matters when
    working with untrained/random-weight networks. Left as None, every
    architecture keeps its normal torchvision default.
    """
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        retinanet_resnet50_fpn,
    )

    builders = {
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
        "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
        "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
        "retinanet_resnet50_fpn": retinanet_resnet50_fpn,
    }
    builder = builders[architecture]

    extra_kwargs = {}
    if score_thresh is not None:
        kwarg_name = "score_thresh" if architecture == "retinanet_resnet50_fpn" else "box_score_thresh"
        extra_kwargs[kwarg_name] = score_thresh

    if pretrained:
        # Uses torchvision's default (COCO-trained) weights. Requires
        # internet access to download weights on first use.
        return builder(weights="DEFAULT", **extra_kwargs)

    # weights=None -> random initialization for the detection head.
    # weights_backbone=None is also required, otherwise torchvision will
    # still try to download ImageNet-pretrained backbone weights.
    # num_classes lets the head match the checkpoint being loaded later.
    return builder(weights=None, weights_backbone=None, num_classes=num_classes, **extra_kwargs)


class ModelLoader:
    """Loads a CV object-detection model according to a PipelineConfig."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _resolve_device(self) -> torch.device:
        requested = self.config.device
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def load(self):
        """Load the model described by ``self.config``.

        Returns
        -------
        (model, metadata): tuple[torch.nn.Module, ModelMetadata]
        """
        cfg = self.config
        device = self._resolve_device()

        if cfg.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{cfg.architecture}'. "
                f"Supported: {SUPPORTED_ARCHITECTURES}"
            )

        weights_hash = None
        if cfg.architecture == "yolov5":
            # YOLOv5 checkpoints are full-model pickles with their own
            # pre/post-processing and class space; everything is handled
            # by the adapter, which returns a torchvision-API-compatible
            # module so the rest of the pipeline is unchanged.
            if not cfg.model_path:
                raise ValueError(
                    "architecture 'yolov5' requires model_path: pretrained "
                    "download and random-init modes are not supported for "
                    "YOLOv5 (supply a .pt checkpoint)."
                )
            model_path = Path(cfg.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
            weights_hash = sha256_of_file(str(model_path))

            from .yolov5_support import load_yolov5_model
            model = load_yolov5_model(
                str(model_path), device=device,
                conf_thres=cfg.detector_score_thresh,
            )
            num_params = sum(p.numel() for p in model.parameters())
            metadata = ModelMetadata(
                model_name=cfg.model_name,
                architecture=cfg.architecture,
                source="yolov5_checkpoint",
                weights_path=cfg.model_path,
                weights_sha256=weights_hash,
                num_parameters=num_params,
                # YOLOv5's own class configuration (80 COCO classes for
                # official models, no background slot) -- NOT cfg.num_classes,
                # which defaults to the torchvision 91-slot convention.
                num_classes=len(model.class_names),
                device=str(device),
            )
            logger.info(
                "Model ready: %s (%s params, source=yolov5_checkpoint, device=%s)",
                cfg.model_name, f"{num_params:,}", device,
            )
            return model, metadata

        if cfg.model_path:
            model_path = Path(cfg.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
            weights_hash = sha256_of_file(str(model_path))

            if model_path.suffix.lower() in (".pt",) and self._looks_like_torchscript(model_path):
                logger.info("Loading TorchScript model from %s", model_path)
                model = torch.jit.load(str(model_path), map_location=device)
                source = "torchscript"
            else:
                logger.info(
                    "Building architecture '%s' and loading checkpoint from %s",
                    cfg.architecture, model_path,
                )
                model = _build_architecture(
                    cfg.architecture, cfg.num_classes, pretrained=False,
                    score_thresh=cfg.detector_score_thresh,
                )
                state_dict = torch.load(str(model_path), map_location=device)
                # Support checkpoints saved either as a raw state_dict or
                # wrapped in a dict such as {"state_dict": ..., "epoch": ...}
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict)
                source = "checkpoint"
        else:
            if cfg.pretrained:
                logger.info(
                    "No checkpoint provided; downloading pretrained weights for '%s'.",
                    cfg.architecture,
                )
                model = _build_architecture(
                    cfg.architecture, cfg.num_classes, pretrained=True,
                    score_thresh=cfg.detector_score_thresh,
                )
                source = "pretrained"
            else:
                logger.warning(
                    "No checkpoint provided and pretrained=False; instantiating "
                    "'%s' with random weights. Predictions will NOT be meaningful "
                    "-- this mode exists only to exercise the pipeline offline.",
                    cfg.architecture,
                )
                model = _build_architecture(
                    cfg.architecture, cfg.num_classes, pretrained=False,
                    score_thresh=cfg.detector_score_thresh,
                )
                source = "random_init"

        model.to(device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())

        metadata = ModelMetadata(
            model_name=cfg.model_name,
            architecture=cfg.architecture,
            source=source,
            weights_path=cfg.model_path,
            weights_sha256=weights_hash,
            num_parameters=num_params,
            num_classes=cfg.num_classes,
            device=str(device),
        )

        logger.info(
            "Model ready: %s (%s params, source=%s, device=%s)",
            cfg.model_name, f"{num_params:,}", source, device,
        )
        return model, metadata

    @staticmethod
    def _looks_like_torchscript(path: Path) -> bool:
        """Heuristically distinguish a TorchScript archive from a plain state_dict.

        TorchScript files saved with torch.jit.save are zip archives that
        contain a top-level 'constants.pkl' / 'data.pkl' entry, whereas a
        plain torch.save(state_dict) is also a zip but only contains
        tensor data with no script metadata. We try the authoritative way:
        attempt torch.jit.load in a throwaway CPU pass would be expensive,
        so instead we peek at the zip file listing.
        """
        import zipfile
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
            return any("constants.pkl" in n or "code/" in n for n in names)
        except zipfile.BadZipFile:
            return False
