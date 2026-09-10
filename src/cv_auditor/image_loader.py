"""
image_loader.py
----------------
Responsible for ONE thing: turning image paths on disk into tensors the
model can consume, plus lightweight per-image metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .utils import find_images, get_logger, sha256_of_file

logger = get_logger(__name__)


@dataclass
class LoadedImage:
    """A single image ready for inference, with basic provenance metadata."""

    path: str
    filename: str
    tensor: torch.Tensor  # shape [3, H, W], float32 in [0, 1]
    width: int
    height: int
    sha256: str


class ImageLoader:
    """Loads one image or a directory of images into model-ready tensors."""

    def __init__(self, input_path: str):
        self.input_path = input_path

    def discover(self) -> List[str]:
        """Return the list of image file paths this loader will process."""
        return find_images(self.input_path)

    def load_one(self, path: str) -> LoadedImage:
        """Load and preprocess a single image file.

        Raises
        ------
        FileNotFoundError, ValueError
            If the file is missing or cannot be decoded as an image.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                tensor = TF.to_tensor(img)  # [0,1] float32, shape [3, H, W]
                width, height = img.size
        except Exception as exc:  # PIL raises several distinct error types
            raise ValueError(f"Could not read image '{path}': {exc}") from exc

        return LoadedImage(
            path=str(p),
            filename=p.name,
            tensor=tensor,
            width=width,
            height=height,
            sha256=sha256_of_file(str(p)),
        )

    def load_all(self) -> List[LoadedImage]:
        """Discover and load every image under ``input_path``."""
        paths = self.discover()
        logger.info("Discovered %d image(s) under '%s'", len(paths), self.input_path)
        loaded = []
        for path in paths:
            try:
                loaded.append(self.load_one(path))
            except (FileNotFoundError, ValueError) as exc:
                # Skip unreadable files but keep the pipeline running for
                # the rest of the batch; caller can inspect logs.
                logger.error("Skipping unreadable image '%s': %s", path, exc)
        if not loaded:
            raise ValueError(f"No images could be successfully loaded from '{self.input_path}'")
        return loaded
