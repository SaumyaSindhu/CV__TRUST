import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from cv_auditor.dataset.dataset_loader import DatasetIndex, Sample
from cv_auditor.dataset.labels import LabelBox
from cv_auditor.dataset.label_anomaly import cluster_embeddings, label_anomaly_findings


def _sample(image_id: str, class_id: int) -> Sample:
    s = Sample(image_id=image_id, image_path=f"/fake/{image_id}", label_path=f"/fake/{image_id}.txt")
    s.readable = True
    s.boxes = [LabelBox(class_id=class_id, x_center=0.5, y_center=0.5, width=0.5, height=0.5)]
    return s


def _index_and_embeddings(samples_with_classes, dim=4, seed=0):
    """Build a DatasetIndex + matching embeddings dict where samples of
    the same declared "visual group" get tightly clustered vectors and
    different visual groups are far apart -- independent of their
    (possibly mismatched) class labels."""
    rng = np.random.default_rng(seed)
    samples = []
    embeddings = {}
    group_centers = {}
    for image_id, label_class, visual_group in samples_with_classes:
        samples.append(_sample(image_id, label_class))
        if visual_group not in group_centers:
            group_centers[visual_group] = rng.normal(loc=visual_group * 10, scale=0.1, size=dim)
        vec = group_centers[visual_group] + rng.normal(scale=0.02, size=dim)
        embeddings[image_id] = vec.astype(np.float32)

    index = DatasetIndex(dataset_dir="/fake", format_name="yolo_folder", labels_expected=True, samples=samples)
    return index, embeddings


def test_consistent_labels_produce_no_anomalies():
    # Two visual groups, each internally label-consistent.
    data = (
        [(f"a{i}", 0, 0) for i in range(4)] +  # visual group 0, all labeled class 0
        [(f"b{i}", 1, 1) for i in range(4)]    # visual group 1, all labeled class 1
    )
    index, embeddings = _index_and_embeddings(data)
    findings = label_anomaly_findings(index, embeddings, n_clusters=2, min_cluster_size=3)
    assert findings == []


def test_minority_label_within_a_visual_cluster_is_flagged():
    # Visual group 0: 4 samples labeled class 0, but ONE labeled class 1
    # despite being visually identical to the rest of the group.
    data = (
        [(f"a{i}", 0, 0) for i in range(4)] +
        [("a_odd", 1, 0)] +  # mislabeled: visually group 0, labeled class 1
        [(f"b{i}", 1, 1) for i in range(4)]
    )
    index, embeddings = _index_and_embeddings(data)
    findings = label_anomaly_findings(
        index, embeddings, n_clusters=2, min_cluster_size=3, minority_ratio_threshold=0.34,
    )

    flagged = {f.sample for f in findings}
    assert "a_odd" in flagged
    # The visually-consistent majority should not be flagged.
    assert "a0" not in flagged


def test_findings_are_worded_as_indicators_not_proof():
    data = (
        [(f"a{i}", 0, 0) for i in range(4)] +
        [("a_odd", 1, 0)] +
        [(f"b{i}", 1, 1) for i in range(4)]
    )
    index, embeddings = _index_and_embeddings(data)
    findings = label_anomaly_findings(index, embeddings, n_clusters=2, min_cluster_size=3)

    assert len(findings) >= 1
    for f in findings:
        assert f.severity == "medium"
        assert "anomaly indicator" in f.reason or "not proof" in f.reason


def test_small_clusters_are_ignored():
    """A cluster smaller than min_cluster_size shouldn't produce findings,
    even if internally inconsistent -- too little evidence to say
    anything meaningful about a 'majority'."""
    data = [("a0", 0, 0), ("a1", 1, 0)]  # only 2 members, disagree
    index, embeddings = _index_and_embeddings(data)
    findings = label_anomaly_findings(index, embeddings, n_clusters=1, min_cluster_size=3)
    assert findings == []


def test_cluster_embeddings_respects_requested_cluster_count():
    data = (
        [(f"a{i}", 0, 0) for i in range(3)] +
        [(f"b{i}", 1, 1) for i in range(3)] +
        [(f"c{i}", 2, 2) for i in range(3)]
    )
    _, embeddings = _index_and_embeddings(data)
    assignment = cluster_embeddings(embeddings, n_clusters=3)
    assert len(set(assignment.values())) == 3


def test_unlabeled_samples_do_not_participate():
    """Samples with no boxes at all shouldn't be forced into the
    majority-vote calculation or flagged."""
    samples = [_sample(f"a{i}", 0) for i in range(3)]
    unlabeled = Sample(image_id="no_label", image_path="/fake/no_label", label_path=None)
    unlabeled.readable = True
    samples.append(unlabeled)

    index = DatasetIndex(dataset_dir="/fake", format_name="yolo_folder", labels_expected=False, samples=samples)
    embeddings = {s.image_id: np.random.default_rng(1).normal(size=4).astype("float32") for s in samples}

    findings = label_anomaly_findings(index, embeddings, n_clusters=1, min_cluster_size=3)
    assert all(f.sample != "no_label" for f in findings)
