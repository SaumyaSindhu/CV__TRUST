"""
duplicates.py
-------------
Exact duplicate detection using cryptographic (SHA-256) hashing of raw
image bytes.

Two images with the same SHA-256 hash are byte-for-byte identical files.
This is a strong, cheap, unambiguous signal -- it produces zero false
positives (modulo the astronomically unlikely case of a hash collision)
but also zero "near miss" detections; that is what ``near_duplicates.py``
is for.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .dataset_loader import DatasetIndex
from .findings import Finding


def find_exact_duplicate_groups(index: DatasetIndex) -> Dict[str, List[str]]:
    """Group sample image_ids by identical SHA-256 hash.

    Returns a dict of {sha256: [image_id, ...]} containing only groups
    with 2 or more members (i.e. actual duplicates). Unreadable samples
    (no hash available) are skipped.
    """
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for sample in index.samples:
        if sample.readable and sample.sha256:
            by_hash[sample.sha256].append(sample.image_id)

    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def exact_duplicate_findings(index: DatasetIndex) -> List[Finding]:
    """Run exact duplicate detection and return one finding per group member.

    Every member of a duplicate group gets its own finding (rather than
    one finding for the whole group) so that filtering/searching findings
    by ``sample`` works the same way it does for every other finding
    type. ``related_samples`` on each finding lists the other members of
    its group.
    """
    findings: List[Finding] = []
    groups = find_exact_duplicate_groups(index)

    for digest, image_ids in groups.items():
        for image_id in image_ids:
            related = [i for i in image_ids if i != image_id]
            findings.append(
                Finding(
                    type="exact_duplicate",
                    severity="low",
                    sample=image_id,
                    reason=(
                        f"Byte-for-byte identical to {len(related)} other "
                        f"sample(s) in the dataset"
                    ),
                    evidence=f"sha256={digest}",
                    related_samples=related,
                    score=1.0,
                    extra={"group_hash": digest, "group_size": len(image_ids)},
                )
            )
    return findings
