"""
result_formatter.py
--------------------
Defines the structured output schema for Phase 1 and handles
saving/loading that schema to/from JSON.

Schema (see README for the full example):

{
  "run_metadata": {
      "model": { ... ModelMetadata ... },
      "config": { ... PipelineConfig ... },
      "generated_at": "..."
  },
  "results": [
      {
        "image": "image_001.jpg",
        "image_path": "data/sample_images/image_001.jpg",
        "image_sha256": "...",
        "width": 640,
        "height": 480,
        "model": "reference_model",
        "inference_timestamp": "...",
        "predictions": [
            {
              "class": "car",
              "class_id": 3,
              "confidence": 0.94,
              "bbox": [120, 80, 450, 300]
            }
        ]
      },
      ...
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from .utils import ensure_parent_dir, get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """A single detected object within one image."""

    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max] in pixel coordinates

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }


@dataclass
class ImageResult:
    """All predictions for a single image, plus provenance metadata."""

    image: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    model: str
    inference_timestamp: str
    predictions: List[DetectionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "model": self.model,
            "inference_timestamp": self.inference_timestamp,
            "predictions": [p.to_dict() for p in self.predictions],
        }


class ResultFormatter:
    """Assembles ImageResults + run metadata and saves/loads them as JSON."""

    def __init__(self, model_metadata: Optional[dict] = None, config: Optional[dict] = None):
        self.model_metadata = model_metadata or {}
        self.config = config or {}
        self.image_results: List[ImageResult] = []

    def add_result(self, image_result: ImageResult) -> None:
        self.image_results.append(image_result)

    def to_dict(self) -> dict:
        from .utils import utc_timestamp

        return {
            "run_metadata": {
                "model": self.model_metadata,
                "config": self.config,
                "generated_at": utc_timestamp(),
            },
            "results": [r.to_dict() for r in self.image_results],
        }

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Saved structured results for %d image(s) to %s",
            len(self.image_results), path,
        )
        return path

    @staticmethod
    def load(path: str) -> dict:
        """Load a previously saved results JSON file back into memory."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(p, "r") as f:
            return json.load(f)
