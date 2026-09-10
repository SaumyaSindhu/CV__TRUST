"""
config.py
---------
Central configuration for the Phase 1 ingestion/inference pipeline.

Keeping all tunables in one place makes it easy for later phases to
introspect *how* an inference run was configured (which is itself
useful audit metadata).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Architectures currently supported out of the box. Kept as a plain
# constant (rather than an Enum) so new entries can be added later
# without touching calling code. The first four are torchvision
# detection models; "yolov5" is handled by src/cv_auditor/yolov5_support.py
# (Ultralytics YOLOv5 checkpoints, loaded via the vendored implementation
# in third_party/yolov5) and uses the model's OWN 80-class label space,
# not COCO_INSTANCE_CATEGORY_NAMES below.
SUPPORTED_ARCHITECTURES = (
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn",
    "yolov5",
)

# COCO class names (index 0 is reserved for "background" in the
# torchvision detection models used here).
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class PipelineConfig:
    """
    Configuration for a single ingestion + inference run.

    Attributes
    ----------
    model_path:
        Path to a model weights/checkpoint file (.pth/.pt state_dict) or a
        TorchScript model (.pt). If None and ``pretrained`` is True, a
        pretrained architecture is downloaded via torchvision instead.
    architecture:
        Name of the torchvision detection architecture to instantiate.
        Must be one of SUPPORTED_ARCHITECTURES.
    pretrained:
        If True and no checkpoint is given, attempt to load torchvision's
        official pretrained (COCO) weights for the architecture. Requires
        internet access. If False, the architecture is instantiated with
        randomly-initialized weights (useful for offline pipeline testing).
    model_name:
        Human-readable identifier stored alongside every prediction so
        later audit phases can tell which model produced which result.
    num_classes:
        Number of output classes the model head predicts, including
        background. Only used when constructing an architecture from
        scratch (ignored for TorchScript models, which are self-contained).
    device:
        "cpu" or "cuda". Defaults to "cpu" for portability; the loader
        will fall back to CPU automatically if CUDA is requested but
        unavailable.
    confidence_threshold:
        Predictions with a score below this value are dropped from the
        structured output.
    detector_score_thresh:
        Optional override for the detection model's *internal* score
        threshold (passed to the torchvision constructor as
        ``box_score_thresh`` for Faster R-CNN variants or ``score_thresh``
        for RetinaNet). Defaults to None, which uses each architecture's
        normal torchvision default and preserves Phase 1's original
        behavior exactly.

        Why this exists: torchvision detection models filter out
        low-confidence boxes *internally*, before `confidence_threshold`
        above ever sees them. With a genuinely untrained/random-weight
        network (as used throughout this offline sandbox -- see README
        "Known Limitations"), that internal filter can reject literally
        everything, producing zero detections regardless of
        `confidence_threshold`. Phase 3 (model_audit) uses this override
        to obtain non-empty, genuinely-computed model output for
        comparison even from untrained weights; it has no effect on any
        Phase 1 functionality when left at its default of None.
    input_dir:
        Path to a single image file OR a directory of images.
    output_path:
        Where the structured JSON results should be written.
    class_names:
        Optional list mapping class index -> human-readable label. Defaults
        to the standard COCO category list, which matches the label space
        of every architecture in SUPPORTED_ARCHITECTURES when using the
        official pretrained weights.
    """

    model_path: Optional[str] = None
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    pretrained: bool = False
    model_name: str = "reference_model"
    num_classes: int = 91  # COCO has 80 classes + background + a few reserved ids
    device: str = "cpu"
    confidence_threshold: float = 0.5
    detector_score_thresh: Optional[float] = None
    input_dir: str = "data/sample_images"
    output_path: str = "results/inference_results.json"
    class_names: list = field(default_factory=lambda: list(COCO_INSTANCE_CATEGORY_NAMES))

    def __post_init__(self):
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{self.architecture}'. "
                f"Supported: {SUPPORTED_ARCHITECTURES}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
