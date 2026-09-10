"""
inference.py
-------------
Responsible for ONE thing: running a loaded model on loaded images and
turning the raw tensor outputs into structured ``ImageResult`` objects.
"""

from __future__ import annotations

from typing import List

import torch

from .image_loader import LoadedImage
from .result_formatter import DetectionResult, ImageResult
from .utils import get_logger, utc_timestamp

logger = get_logger(__name__)


class InferenceEngine:
    """Runs a torchvision-style detection model on one or more images.

    The engine expects models that follow the torchvision detection API:
    calling ``model(list_of_image_tensors)`` returns a list of dicts, each
    containing "boxes", "labels", and "scores" tensors. This covers every
    architecture in ``config.SUPPORTED_ARCHITECTURES``.
    """

    def __init__(self, model, model_name: str, class_names: List[str],
                 device: torch.device, confidence_threshold: float = 0.5):
        self.model = model
        self.model_name = model_name
        self.class_names = class_names
        self.device = device
        self.confidence_threshold = confidence_threshold

    def _label_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    @torch.no_grad()
    def run_one(self, image: LoadedImage) -> ImageResult:
        """Run inference on a single loaded image and return structured results."""
        tensor = image.tensor.to(self.device)
        timestamp = utc_timestamp()

        raw_output = self.model([tensor])[0]

        boxes = raw_output.get("boxes", torch.empty((0, 4)))
        labels = raw_output.get("labels", torch.empty((0,), dtype=torch.int64))
        scores = raw_output.get("scores", torch.empty((0,)))

        predictions: List[DetectionResult] = []
        for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            if score < self.confidence_threshold:
                continue
            predictions.append(
                DetectionResult(
                    class_name=self._label_for(int(label)),
                    class_id=int(label),
                    confidence=float(score),
                    bbox=[float(v) for v in box],
                )
            )

        logger.info(
            "Image '%s': %d prediction(s) above threshold %.2f",
            image.filename, len(predictions), self.confidence_threshold,
        )

        return ImageResult(
            image=image.filename,
            image_path=image.path,
            image_sha256=image.sha256,
            width=image.width,
            height=image.height,
            model=self.model_name,
            inference_timestamp=timestamp,
            predictions=predictions,
        )

    def run_batch(self, images: List[LoadedImage]) -> List[ImageResult]:
        """Run inference sequentially over a list of loaded images.

        Sequential (rather than batched-tensor) execution is used
        deliberately for Phase 1: it is simpler, easier to debug, and
        avoids padding/collation concerns for images of differing sizes.
        Batched execution can be introduced later as a performance
        optimization without changing this method's public contract.
        """
        results = []
        for image in images:
            results.append(self.run_one(image))
        return results
