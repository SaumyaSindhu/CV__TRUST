"""
labels.py
---------
Parsing for YOLO-style label files.

YOLO label format (one .txt file per image, same stem as the image file):

    <class_id> <x_center> <y_center> <width> <height>

One line per object, all four geometry values normalized to [0, 1]
relative to the image's width/height. This module parses that format
defensively: a malformed line never raises past the caller -- it is
reported back as a list of parse errors so the validation step (see
``validation.py``) can turn it into a ``Finding`` instead of crashing the
whole audit run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class LabelBox:
    """A single parsed YOLO label line."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    class_name: Optional[str] = None

    def is_geometry_valid(self) -> bool:
        """Whether the normalized box geometry is sane.

        A box is considered valid if all values fall in [0, 1] and the
        box has positive width/height. This is a purely geometric check;
        it says nothing about whether the *class* is correct.
        """
        values_in_range = all(
            0.0 <= v <= 1.0 for v in (self.x_center, self.y_center, self.width, self.height)
        )
        positive_extent = self.width > 0.0 and self.height > 0.0
        return values_in_range and positive_extent

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class LabelParseResult:
    """Outcome of parsing one label file."""

    boxes: List[LabelBox]
    errors: List[str]


def parse_yolo_label_file(path: str, class_names: Optional[List[str]] = None) -> LabelParseResult:
    """Parse a single YOLO-format label file.

    Never raises for malformed content -- malformed lines are skipped and
    recorded in ``LabelParseResult.errors`` instead, so one bad line does
    not prevent the rest of the file (or the rest of the dataset) from
    being processed.
    """
    p = Path(path)
    boxes: List[LabelBox] = []
    errors: List[str] = []

    if not p.exists():
        return LabelParseResult(boxes=[], errors=[f"Label file not found: {path}"])

    try:
        lines = p.read_text().splitlines()
    except Exception as exc:  # unreadable / binary garbage etc.
        return LabelParseResult(boxes=[], errors=[f"Could not read label file '{path}': {exc}"])

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{path}:{line_no}: expected 5 fields (class x y w h), got {len(parts)}"
            )
            continue

        parsed = _parse_fields(parts)
        if parsed is None:
            errors.append(f"{path}:{line_no}: non-numeric field(s) in '{line}'")
            continue

        class_id, x, y, w, h = parsed
        class_name = None
        if class_names is not None:
            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                errors.append(
                    f"{path}:{line_no}: class_id {class_id} out of range "
                    f"(known classes: 0..{len(class_names) - 1})"
                )

        box = LabelBox(
            class_id=class_id, x_center=x, y_center=y, width=w, height=h, class_name=class_name,
        )
        if not box.is_geometry_valid():
            errors.append(
                f"{path}:{line_no}: invalid bounding box geometry "
                f"(x={x}, y={y}, w={w}, h={h}) -- must be in [0,1] with positive extent"
            )
            continue

        boxes.append(box)

    return LabelParseResult(boxes=boxes, errors=errors)


def _parse_fields(parts: List[str]) -> Optional[Tuple[int, float, float, float, float]]:
    try:
        class_id = int(float(parts[0]))
        x, y, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None
    return class_id, x, y, w, h
