"""
validation.py
--------------
Turns the raw observations recorded by ``DatasetLoader`` (unreadable
images, missing/malformed labels, invalid boxes) into ``Finding`` objects
of type ``"invalid_sample"``.

This is intentionally separate from loading: the loader's job is to
survive bad data and keep going, this module's job is to decide how
noteworthy each problem is (severity) and phrase it consistently.
"""

from __future__ import annotations

from typing import List

from .dataset_loader import DatasetIndex, Sample
from .findings import Finding


def validate_dataset(index: DatasetIndex) -> List[Finding]:
    """Validate every sample in ``index`` and return invalid_sample findings.

    A single sample can produce more than one finding (e.g. an unreadable
    image with a malformed label file) -- each problem is reported
    separately so downstream consumers don't have to parse a combined
    reason string.
    """
    findings: List[Finding] = []
    for sample in index.samples:
        findings.extend(_validate_sample(sample))
    return findings


def _validate_sample(sample: Sample) -> List[Finding]:
    findings: List[Finding] = []

    if not sample.readable:
        findings.append(
            Finding(
                type="invalid_sample",
                severity="high",
                sample=sample.image_id,
                reason="Image is unreadable or corrupted",
                evidence=str(sample.read_error or "unknown decode error"),
            )
        )
        # If the image itself can't be read there is little value in also
        # reporting label problems for it -- skip further checks.
        return findings

    for error in sample.label_errors:
        is_missing = error.startswith("Missing label file")
        is_malformed_line = ": expected " in error or ": non-numeric" in error
        is_bad_box = "invalid bounding box geometry" in error
        is_bad_class = "out of range" in error

        if is_missing:
            severity = "medium"
            reason = "Missing label file for a dataset that otherwise has labels"
        elif is_bad_box:
            severity = "medium"
            reason = "Invalid bounding box"
        elif is_bad_class:
            severity = "medium"
            reason = "Class id not in known class list"
        elif is_malformed_line:
            severity = "low"
            reason = "Malformed label line"
        else:
            severity = "low"
            reason = "Label parsing issue"

        findings.append(
            Finding(
                type="invalid_sample",
                severity=severity,
                sample=sample.image_id,
                reason=reason,
                evidence=error,
            )
        )

    return findings
