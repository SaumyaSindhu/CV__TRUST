"""
ood.py
------
Baseline out-of-distribution / anomaly detection over image embeddings.

Method
------
1. For every sample, compute the mean cosine distance to its k nearest
   neighbours within the dataset itself ("local outlier distance"). A
   sample whose nearest neighbours are all comparatively far away looks
   visually/statistically different from the rest of the dataset.
2. Convert that raw distance into a **z-score** relative to the dataset's
   own mean and standard deviation of local-outlier-distance. Reporting
   a z-score (rather than the raw distance) makes the anomaly score
   comparable across datasets and across embedding backends -- raw
   cosine-distance scale varies a lot depending on what produced the
   embeddings (e.g. the handcrafted offline fallback vs. a pretrained
   CNN; see ``embeddings.py``), but "how many standard deviations from
   this dataset's own typical sample" is a stable, explainable
   statistical-distance measure regardless of backend.

A sample may be an outlier for many legitimate reasons that have
nothing to do with an attack -- see below.

Why kNN + z-score
-------------------
kNN distance has no distributional assumptions (unlike e.g. fitting a
single Gaussian and using Mahalanobis distance over raw features), and is
easy to explain: "this image's k closest matches in the dataset are
still unusually far away, relative to how far apart samples in this
dataset normally are." Chosen over a heavier learned OOD model to keep
this baseline explainable and dependency-light, per the project's design
principles.

IMPORTANT: nothing in this module calls a sample "attacked" or
"poisoned" -- see ``ood_findings`` below and the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .findings import Finding


@dataclass
class OODScore:
    image_id: str
    raw_distance: float  # mean cosine distance to k nearest neighbours
    score: float  # z-score of raw_distance relative to the dataset population
    is_anomalous: bool


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = vectors / norms
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def compute_ood_scores(
    embeddings: Dict[str, np.ndarray],
    k: int = 5,
    threshold: float = 2.0,
) -> List[OODScore]:
    """Compute a kNN-distance anomaly z-score for every embedded sample.

    Parameters
    ----------
    embeddings:
        Mapping of image_id -> embedding vector (e.g. from
        ``EmbeddingExtractor.extract_batch``).
    k:
        Number of nearest neighbours to average over. Automatically
        capped at (n_samples - 1) for small datasets.
    threshold:
        Z-score (standard deviations above the dataset's own mean local
        distance) at/above which a sample is flagged as anomalous. The
        default of 2.0 is the common "more than two standard deviations
        from the mean" statistical convention -- a reasonable, tunable
        starting point; see README for guidance on adjusting it.
    """
    image_ids = list(embeddings.keys())
    n = len(image_ids)
    if n < 4:
        # Not enough samples for a meaningful neighbourhood or population
        # statistic; report everything as non-anomalous rather than
        # computing a z-score against a near-meaningless std deviation.
        return [OODScore(image_id=i, raw_distance=0.0, score=0.0, is_anomalous=False) for i in image_ids]

    vectors = np.stack([embeddings[i] for i in image_ids])
    distances = _cosine_distance_matrix(vectors)
    effective_k = min(k, n - 1)

    raw_scores = []
    for idx in range(n):
        row = np.delete(distances[idx], idx)
        nearest = np.sort(row)[:effective_k]
        raw_scores.append(float(np.mean(nearest)))
    raw_scores = np.array(raw_scores)

    mean, std = float(raw_scores.mean()), float(raw_scores.std())
    if std < 1e-8:
        # Every sample is (near) equidistant from its neighbours -- no
        # meaningful spread to compute a z-score against.
        z_scores = np.zeros_like(raw_scores)
    else:
        z_scores = (raw_scores - mean) / std

    scores: List[OODScore] = []
    for image_id, raw, z in zip(image_ids, raw_scores, z_scores):
        scores.append(
            OODScore(image_id=image_id, raw_distance=raw, score=float(z), is_anomalous=z >= threshold)
        )
    return scores


def ood_findings(
    embeddings: Dict[str, np.ndarray],
    k: int = 5,
    threshold: float = 2.0,
    high_severity_threshold: float = 3.0,
) -> List[Finding]:
    """Run OOD scoring and return a Finding for every sample above ``threshold``."""
    scores = compute_ood_scores(embeddings, k=k, threshold=threshold)
    findings: List[Finding] = []
    for s in scores:
        if not s.is_anomalous:
            continue
        severity = "high" if s.score >= high_severity_threshold else "medium"
        findings.append(
            Finding(
                type="ood_anomaly",
                severity=severity,
                sample=s.image_id,
                reason="Potential distribution/anomaly finding: sample is visually "
                       "distant from its nearest neighbours in the dataset",
                evidence=(
                    f"mean cosine distance to {min(k, len(embeddings) - 1)} nearest "
                    f"neighbours = {s.raw_distance:.4f}, which is {s.score:.2f} standard "
                    f"deviations above this dataset's own mean (threshold={threshold:.2f})"
                ),
                score=s.score,
                extra={"k": k, "threshold": threshold, "raw_distance": s.raw_distance},
            )
        )
    return findings
