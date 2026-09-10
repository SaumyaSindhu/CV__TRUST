import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv_auditor.model_audit.comparison import compare_images, compare_profiles

from .model_audit_helpers import make_prediction, make_profile


def test_identical_predictions_are_fully_matched():
    ref = [make_prediction(3, 0.9, (10, 10, 50, 50))]
    cand = [make_prediction(3, 0.9, (10, 10, 50, 50))]
    ic = compare_images("img.jpg", ref, cand, iou_threshold=0.5)

    assert ic.num_matched == 1
    assert ic.num_missing == 0
    assert ic.num_added == 0
    assert ic.num_class_flips == 0
    assert ic.confidence_diffs == [0.0]


def test_disjoint_boxes_are_missing_and_added_not_matched():
    ref = [make_prediction(3, 0.9, (10, 10, 50, 50))]
    cand = [make_prediction(3, 0.9, (200, 200, 250, 250))]  # far away, no overlap
    ic = compare_images("img.jpg", ref, cand, iou_threshold=0.5)

    assert ic.num_matched == 0
    assert ic.num_missing == 1
    assert ic.num_added == 1


def test_same_box_different_class_is_a_class_flip_not_missing_and_added():
    ref = [make_prediction(3, 0.9, (10, 10, 50, 50))]  # class 3
    cand = [make_prediction(7, 0.9, (10, 10, 50, 50))]  # same box, class 7
    ic = compare_images("img.jpg", ref, cand, iou_threshold=0.5)

    assert ic.num_matched == 1
    assert ic.num_class_flips == 1
    assert ic.num_missing == 0
    assert ic.num_added == 0


def test_confidence_diff_is_recorded_for_matched_pairs():
    ref = [make_prediction(3, 0.9, (10, 10, 50, 50))]
    cand = [make_prediction(3, 0.6, (10, 10, 50, 50))]
    ic = compare_images("img.jpg", ref, cand, iou_threshold=0.5)
    assert len(ic.confidence_diffs) == 1
    assert abs(ic.confidence_diffs[0] - 0.3) < 1e-6


def test_compare_profiles_aggregates_across_images():
    ref = make_profile("ref", "hash_a", {
        "a.jpg": [make_prediction(3, 0.9, (10, 10, 50, 50))],
        "b.jpg": [make_prediction(5, 0.8, (0, 0, 20, 20))],
    })
    cand = make_profile("cand", "hash_b", {
        "a.jpg": [make_prediction(3, 0.9, (10, 10, 50, 50))],  # identical
        "b.jpg": [],  # missing detection
    })

    result = compare_profiles(ref, cand, iou_threshold=0.5)
    assert result.total_reference == 2
    assert result.total_candidate == 1
    assert result.total_matched == 1
    assert result.total_missing == 1
    assert result.total_added == 0
    assert 0.0 < result.agreement_rate < 1.0


def test_compare_profiles_perfect_agreement_when_both_empty():
    ref = make_profile("ref", "hash_a", {"a.jpg": []})
    cand = make_profile("cand", "hash_a", {"a.jpg": []})
    result = compare_profiles(ref, cand)
    assert result.agreement_rate == 1.0
    assert result.total_matched == 0


def test_compare_profiles_only_compares_shared_images():
    ref = make_profile("ref", "hash_a", {"a.jpg": [], "only_in_ref.jpg": []})
    cand = make_profile("cand", "hash_a", {"a.jpg": [], "only_in_cand.jpg": []})
    result = compare_profiles(ref, cand)
    assert set(result.skipped_images) == {"only_in_ref.jpg", "only_in_cand.jpg"}
    assert len(result.per_image) == 1


def test_dominant_added_class_is_identified():
    ref = make_profile("ref", "hash_a", {"a.jpg": []})
    cand = make_profile("cand", "hash_b", {
        "a.jpg": [
            make_prediction(5, 0.95, (0, 0, 10, 10)),
            make_prediction(5, 0.97, (20, 20, 30, 30)),
            make_prediction(5, 0.96, (40, 40, 50, 50)),
            make_prediction(9, 0.55, (60, 60, 70, 70)),  # one different class
        ],
    })
    result = compare_profiles(ref, cand)
    assert result.dominant_added_class_id == 5
    assert result.dominant_added_class_fraction == 0.75
    assert result.dominant_added_class_mean_confidence > 0.9


def test_comparison_result_to_dict_is_json_safe():
    import json
    ref = make_profile("ref", "hash_a", {"a.jpg": [make_prediction(3, 0.9)]})
    cand = make_profile("cand", "hash_b", {"a.jpg": [make_prediction(3, 0.9)]})
    result = compare_profiles(ref, cand)
    json.dumps(result.to_dict())  # should not raise
