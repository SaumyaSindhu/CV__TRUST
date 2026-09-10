"""Shared helpers for Phase 2 (dataset auditor) tests.

Builds small, fully synthetic datasets on disk (under pytest's tmp_path)
so unit tests are isolated from the shipped ``data/sample_dataset`` and
from each other. The shipped sample dataset is exercised separately by
``test_dataset_auditor_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw


def make_image(path: Path, color: Tuple[int, int, int] = (200, 60, 60), size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=color)
    img.save(path)


def write_label(path: Path, class_id: int, cx=0.5, cy=0.5, w=0.4, h=0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {cx} {cy} {w} {h}\n")


def build_basic_dataset(root: Path, n: int = 4, with_labels: bool = True) -> Path:
    """A minimal, valid YOLO-folder dataset with ``n`` distinct images."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    colors = [(200, 60, 60), (60, 120, 200), (60, 180, 90), (200, 200, 60)]
    for i in range(1, n + 1):
        name = f"img_{i:03d}"
        make_image(images_dir / f"{name}.jpg", color=colors[(i - 1) % len(colors)])
        if with_labels:
            write_label(labels_dir / f"{name}.txt", class_id=(i - 1) % 3)
    return root
