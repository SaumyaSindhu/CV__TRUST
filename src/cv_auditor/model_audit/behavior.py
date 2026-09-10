"""
behavior.py
------------
Runs a model on a fixed set of test images and records its predictions
as a structured "behavioural profile" -- a snapshot of *what the model
actually does*, as opposed to what its file hash says.

This deliberately reuses Phase 1's ``InferenceEngine`` and
``ImageLoader`` rather than reimplementing inference: a behavioural
profile is exactly a Phase 1 inference run, just captured and saved
specifically so it can be diffed against another run later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from ..image_loader import ImageLoader, LoadedImage
from ..inference import InferenceEngine
from ..model_loader import ModelMetadata
from ..utils import ensure_parent_dir, get_logger, utc_timestamp

logger = get_logger(__name__)


@dataclass
class BehaviorProfile:
    """A model's recorded predictions on a fixed set of test images.

    ``image_results`` holds one entry per test image, in the same shape
    as Phase 1's ``ImageResult.to_dict()`` (image filename, sha256,
    dimensions, and a list of {class, class_id, confidence, bbox}
    predictions) -- this profile IS a Phase 1 inference result set, just
    named for its Phase 3 purpose.
    """

    model_name: str
    architecture: str
    weights_sha256: Optional[str]
    test_images_dir: str
    confidence_threshold: float
    generated_at: str
    image_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "weights_sha256": self.weights_sha256,
            "test_images_dir": self.test_images_dir,
            "confidence_threshold": self.confidence_threshold,
            "generated_at": self.generated_at,
            "image_results": self.image_results,
        }

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            "Saved behavioural baseline for '%s' (%d image(s)) to %s",
            self.model_name, len(self.image_results), path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "BehaviorProfile":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Behavioural baseline not found: {path}")
        with open(p, "r") as f:
            data = json.load(f)
        return cls(**data)


def generate_behavior_profile(
    model,
    metadata: ModelMetadata,
    images: List[LoadedImage],
    class_names: List[str],
    device: torch.device,
    confidence_threshold: float,
    test_images_dir: str = "",
) -> BehaviorProfile:
    """Run ``model`` on ``images`` and package the results as a BehaviorProfile."""
    engine = InferenceEngine(
        model=model,
        model_name=metadata.model_name,
        class_names=class_names,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    results = engine.run_batch(images)
    return BehaviorProfile(
        model_name=metadata.model_name,
        architecture=metadata.architecture,
        weights_sha256=metadata.weights_sha256,
        test_images_dir=test_images_dir,
        confidence_threshold=confidence_threshold,
        generated_at=utc_timestamp(),
        image_results=[r.to_dict() for r in results],
    )


def load_test_images(test_images_dir: str) -> List[LoadedImage]:
    """Convenience wrapper around Phase 1's ``ImageLoader`` for a fixed test set."""
    return ImageLoader(test_images_dir).load_all()
