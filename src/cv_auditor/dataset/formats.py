"""
formats.py
----------
Pluggable dataset *format* discovery.

A "format" is responsible for exactly one thing: given a dataset
directory, figure out which image files exist and which label file (if
any) belongs to each image. It does NOT parse label contents (see
``labels.py``) and does NOT validate anything (see ``validation.py``) --
keeping this narrow is what makes it easy to add new dataset layouts
later (e.g. a COCO-JSON format, a Pascal-VOC-XML format) without
touching the loader, validator, or any detector.

Only one format is implemented in Phase 2:

    dataset/
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── labels/
        ├── image1.txt
        └── ...

New formats can be added by subclassing ``DatasetFormat`` and registering
an instance/class in ``FORMAT_REGISTRY``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from ..utils import IMAGE_EXTENSIONS


@dataclass
class RawSample:
    """An (image path, label path) pair discovered by a format, pre-parsing."""

    image_path: str
    image_id: str  # stable identifier, typically the filename
    label_path: Optional[str]  # None if this format/dataset has no labels


class DatasetFormat(ABC):
    """Base class for a dataset directory layout."""

    name: str = "base"

    @abstractmethod
    def discover(self, dataset_dir: str) -> List[RawSample]:
        """Return every (image, label) pair found under ``dataset_dir``."""
        raise NotImplementedError

    @abstractmethod
    def has_labels(self, dataset_dir: str) -> bool:
        """Whether this dataset appears to include any labels at all.

        Used to decide whether "missing label file" should be treated as
        a finding (labeled dataset with gaps) or as expected/normal
        (a fully unlabeled dataset).
        """
        raise NotImplementedError


class YoloFolderFormat(DatasetFormat):
    """The ``images/`` + ``labels/`` YOLO-style folder convention."""

    name = "yolo_folder"

    def _images_dir(self, dataset_dir: str) -> Path:
        return Path(dataset_dir) / "images"

    def _labels_dir(self, dataset_dir: str) -> Path:
        return Path(dataset_dir) / "labels"

    def has_labels(self, dataset_dir: str) -> bool:
        labels_dir = self._labels_dir(dataset_dir)
        if not labels_dir.is_dir():
            return False
        return any(labels_dir.glob("*.txt"))

    def discover(self, dataset_dir: str) -> List[RawSample]:
        images_dir = self._images_dir(dataset_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected an 'images/' subdirectory under '{dataset_dir}' "
                f"for the '{self.name}' dataset format."
            )
        labels_dir = self._labels_dir(dataset_dir)
        labels_present = self.has_labels(dataset_dir)

        samples: List[RawSample] = []
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path: Optional[str] = None
            if labels_present:
                candidate = labels_dir / f"{image_path.stem}.txt"
                label_path = str(candidate) if candidate.exists() else None

            samples.append(
                RawSample(
                    image_path=str(image_path),
                    image_id=image_path.name,
                    label_path=label_path,
                )
            )
        return samples


# Registry so new formats can be plugged in without changing DatasetLoader.
FORMAT_REGISTRY: Dict[str, Type[DatasetFormat]] = {
    "yolo_folder": YoloFolderFormat,
}


def get_format(name: str) -> DatasetFormat:
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown dataset format '{name}'. Registered formats: {list(FORMAT_REGISTRY)}"
        )
    return FORMAT_REGISTRY[name]()
