"""
near_duplicates.py
-------------------
Turns dHash-based similarity groups (``perceptual_hash.py``) into
``Finding`` objects, the same way ``duplicates.py`` does for exact
duplicates.
"""

from __future__ import annotations

from typing import List

from .dataset_loader import DatasetIndex
from .findings import Finding
from .perceptual_hash import find_near_duplicate_groups


def near_duplicate_findings(
    index: DatasetIndex,
    similarity_threshold: float = 0.90,
    hash_size: int = 8,
) -> List[Finding]:
    """Run near-duplicate detection and return one finding per group member."""
    image_paths_by_id = {
        s.image_id: s.image_path for s in index.samples if s.readable
    }
    groups = find_near_duplicate_groups(
        image_paths_by_id, similarity_threshold=similarity_threshold, hash_size=hash_size,
    )

    findings: List[Finding] = []
    for group in groups:
        members = group["members"]
        min_sim = group["min_pairwise_similarity"]
        for image_id in members:
            related = [m for m in members if m != image_id]
            findings.append(
                Finding(
                    type="near_duplicate",
                    severity="low",
                    sample=image_id,
                    reason=(
                        f"Visually near-identical to {len(related)} other "
                        f"sample(s) (perceptual hash similarity >= {min_sim:.2%})"
                    ),
                    evidence=f"dHash similarity threshold={similarity_threshold:.2f}",
                    related_samples=related,
                    score=min_sim,
                    extra={"group_size": len(members)},
                )
            )
    return findings
