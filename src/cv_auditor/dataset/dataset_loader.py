"""
dataset_loader.py
------------------
Turns a dataset directory into a list of structured ``Sample`` objects.

This module deliberately does *no* judgment calls about validity -- it
just records what it observed (including read/parse failures) on each
``Sample``. The validation step (``validation.py``) is what turns those
observations into ``Finding`` objects. Keeping ingestion and validation
separate means new validation rules can be added without touching
loading logic, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from ..utils import get_logger, sha256_of_file
from .formats import DatasetFormat, RawSample, get_format
from .labels import LabelBox, parse_yolo_label_file

logger = get_logger(__name__)


@dataclass
class Sample:
    """One dataset item: an image plus whatever label information exists.

    ``readable`` / ``read_error`` capture image-loading outcome so the
    validator can report unreadable images without the loader crashing.
    Similarly, ``label_errors`` carries label-parsing problems for the
    same reason.
    """

    image_id: str
    image_path: str
    label_path: Optional[str]

    readable: bool = False
    read_error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: Optional[str] = None

    boxes: List[LabelBox] = field(default_factory=list)
    label_errors: List[str] = field(default_factory=list)

    @property
    def class_ids(self) -> List[int]:
        return [b.class_id for b in self.boxes]

    @property
    def has_label_file(self) -> bool:
        return self.label_path is not None


@dataclass
class DatasetIndex:
    """The full result of loading a dataset directory."""

    dataset_dir: str
    format_name: str
    labels_expected: bool  # whether this dataset appears to be a labeled dataset overall
    samples: List[Sample]

    @property
    def num_samples(self) -> int:
        return len(self.samples)


class DatasetLoader:
    """Loads a dataset directory into a ``DatasetIndex``.

    Parameters
    ----------
    dataset_dir:
        Root directory of the dataset.
    format_name:
        Which registered ``DatasetFormat`` to use for discovery. Only
        ``"yolo_folder"`` is implemented in Phase 2; the registry in
        ``formats.py`` is where future formats get added.
    class_names:
        Optional list mapping YOLO class_id -> human-readable name. If
        omitted, class names are left as ``None`` and only numeric class
        ids are used (this is fine -- label anomaly / OOD detection work
        on class ids just as well as names).
    """

    def __init__(self, dataset_dir: str, format_name: str = "yolo_folder",
                 class_names: Optional[List[str]] = None):
        self.dataset_dir = dataset_dir
        self.format_name = format_name
        self.class_names = class_names
        self._format: DatasetFormat = get_format(format_name)

    def load(self) -> DatasetIndex:
        raw_samples: List[RawSample] = self._format.discover(self.dataset_dir)
        labels_expected = self._format.has_labels(self.dataset_dir)

        if not raw_samples:
            raise FileNotFoundError(
                f"No supported images found under '{self.dataset_dir}/images'."
            )

        samples: List[Sample] = []
        for raw in raw_samples:
            samples.append(self._load_one(raw, labels_expected))

        logger.info(
            "Loaded %d sample(s) from '%s' (format=%s, labels_expected=%s)",
            len(samples), self.dataset_dir, self.format_name, labels_expected,
        )
        return DatasetIndex(
            dataset_dir=self.dataset_dir,
            format_name=self.format_name,
            labels_expected=labels_expected,
            samples=samples,
        )

    def _load_one(self, raw: RawSample, labels_expected: bool) -> Sample:
        sample = Sample(image_id=raw.image_id, image_path=raw.image_path, label_path=raw.label_path)

        # --- image ---
        try:
            with Image.open(raw.image_path) as img:
                img.verify()  # cheap corruption check
            # Re-open: verify() leaves the file in a state that can't be
            # used for further operations (per PIL docs), so we reopen to
            # actually read dimensions.
            with Image.open(raw.image_path) as img:
                img = img.convert("RGB")
                sample.width, sample.height = img.size
            sample.sha256 = sha256_of_file(raw.image_path)
            sample.readable = True
        except Exception as exc:  # noqa: BLE001 - PIL raises many distinct types
            sample.readable = False
            sample.read_error = str(exc)
            logger.warning("Unreadable image '%s': %s", raw.image_path, exc)

        # --- labels ---
        if raw.label_path is not None:
            result = parse_yolo_label_file(raw.label_path, class_names=self.class_names)
            sample.boxes = result.boxes
            sample.label_errors = result.errors
        elif labels_expected:
            sample.label_errors = [f"Missing label file for '{raw.image_id}'"]

        return sample
