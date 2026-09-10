"""
label_anomaly.py
------------------
Flags samples whose label looks inconsistent with visually similar
samples elsewhere in the dataset.

Method
------
1. Cluster image embeddings with k-means (visually similar images end up
   in the same cluster).
2. Within each cluster of size >= ``min_cluster_size``, determine the
   "majority class" -- the most common primary class label among the
   cluster's members.
3. Any member whose primary class differs from the cluster's majority
   class is flagged as a potential label anomaly.

A sample's "primary class" is the class of its largest-area bounding box
(the most visually dominant object), so this only looks at one label per
image even for multi-object images -- a documented simplification, not a
claim of full multi-label consistency checking.

IMPORTANT LIMITATIONS (see README for the full list):
  * This is an *anomaly indicator*, not proof the label is wrong. A
    minority label in a visual cluster can be entirely correct -- visual
    similarity is not the same thing as ground-truth class identity
    (e.g. two visually similar birds can be different, correctly-labeled
    species).
  * Cluster quality depends on the embedding model; see the caveats in
    ``embeddings.py`` about pretrained vs. randomly-initialized weights.
  * Only samples with at least one label box participate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .dataset_loader import DatasetIndex, Sample
from .findings import Finding


@dataclass
class ClusterLabelSummary:
    cluster_id: int
    members: List[str]
    majority_class: Optional[int]
    minority_members: List[str]


def _primary_class(sample: Sample) -> Optional[int]:
    """Return the class_id of the sample's largest-area box, if any."""
    if not sample.boxes:
        return None
    largest = max(sample.boxes, key=lambda b: b.width * b.height)
    return largest.class_id


def cluster_embeddings(embeddings: Dict[str, np.ndarray], n_clusters: int) -> Dict[str, int]:
    """K-means cluster embeddings. Returns {image_id: cluster_id}."""
    from sklearn.cluster import KMeans

    image_ids = list(embeddings.keys())
    vectors = np.stack([embeddings[i] for i in image_ids])

    n_clusters = max(1, min(n_clusters, len(image_ids)))
    if n_clusters == 1:
        return {image_id: 0 for image_id in image_ids}

    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(vectors)
    return {image_id: int(cluster_id) for image_id, cluster_id in zip(image_ids, labels)}


def detect_label_anomalies(
    index: DatasetIndex,
    embeddings: Dict[str, np.ndarray],
    n_clusters: int = 8,
    min_cluster_size: int = 3,
    minority_ratio_threshold: float = 0.34,
) -> List[ClusterLabelSummary]:
    """Cluster embeddings and identify label-inconsistent clusters.

    A cluster member is flagged only if its class is a strict minority
    within the cluster (occurs in less than ``minority_ratio_threshold``
    of labeled, embedded members of that cluster) -- this avoids flagging
    every member of a genuinely mixed-class cluster and instead focuses
    on the samples that stick out from an otherwise-consistent group.
    """
    samples_by_id = {s.image_id: s for s in index.samples}
    labeled_embedded_ids = [
        image_id for image_id in embeddings
        if image_id in samples_by_id and _primary_class(samples_by_id[image_id]) is not None
    ]
    if len(labeled_embedded_ids) < min_cluster_size:
        return []

    labeled_embeddings = {i: embeddings[i] for i in labeled_embedded_ids}
    cluster_of = cluster_embeddings(labeled_embeddings, n_clusters=n_clusters)

    members_by_cluster: Dict[int, List[str]] = {}
    for image_id, cluster_id in cluster_of.items():
        members_by_cluster.setdefault(cluster_id, []).append(image_id)

    summaries: List[ClusterLabelSummary] = []
    for cluster_id, members in members_by_cluster.items():
        if len(members) < min_cluster_size:
            continue

        classes = [_primary_class(samples_by_id[m]) for m in members]
        counts = Counter(classes)
        majority_class, majority_count = counts.most_common(1)[0]

        minority_members = [
            m for m, c in zip(members, classes)
            if c != majority_class and (counts[c] / len(members)) < minority_ratio_threshold
        ]

        summaries.append(
            ClusterLabelSummary(
                cluster_id=cluster_id,
                members=sorted(members),
                majority_class=majority_class,
                minority_members=sorted(minority_members),
            )
        )
    return summaries


def label_anomaly_findings(
    index: DatasetIndex,
    embeddings: Dict[str, np.ndarray],
    n_clusters: int = 8,
    min_cluster_size: int = 3,
    minority_ratio_threshold: float = 0.34,
) -> List[Finding]:
    samples_by_id = {s.image_id: s for s in index.samples}
    summaries = detect_label_anomalies(
        index, embeddings,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        minority_ratio_threshold=minority_ratio_threshold,
    )

    findings: List[Finding] = []
    for summary in summaries:
        for image_id in summary.minority_members:
            sample_class = _primary_class(samples_by_id[image_id])
            findings.append(
                Finding(
                    type="label_anomaly",
                    severity="medium",
                    sample=image_id,
                    reason=(
                        "Label differs from the majority label of visually similar "
                        "images in the same cluster -- an anomaly indicator, not "
                        "proof the label is incorrect"
                    ),
                    evidence=(
                        f"cluster {summary.cluster_id}: sample class_id={sample_class}, "
                        f"cluster majority class_id={summary.majority_class}, "
                        f"cluster size={len(summary.members)}"
                    ),
                    related_samples=[m for m in summary.members if m != image_id],
                    extra={
                        "cluster_id": summary.cluster_id,
                        "sample_class_id": sample_class,
                        "majority_class_id": summary.majority_class,
                    },
                )
            )
    return findings
