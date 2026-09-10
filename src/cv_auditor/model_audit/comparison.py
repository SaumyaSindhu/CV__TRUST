"""
comparison.py
--------------
Compares two ``BehaviorProfile``s (reference vs. candidate) produced on
the SAME fixed set of test images and computes explainable comparison
metrics: how often they agree, how their confidences differ, how often
a matched detection's class disagrees ("class flip"), and which
detections appear only on one side ("missing" vs "added").

Matching method
----------------
For each image, reference and candidate predictions are matched
greedily by bounding-box IoU (highest-IoU pairs first, each box used at
most once) -- a standard, explainable box-matching approach. A pair
counts as "matched" once its IoU clears ``iou_threshold``, regardless of
whether their predicted classes agree (a same-location-different-class
match is exactly the "class flip" signal that matters for spotting
suspicious behavior, so it must not be treated as an "added" + "missing"
pair -- see ``suspicion.py``).

This only compares images present in BOTH profiles (matched by
filename); a documented assumption is that both profiles were generated
against the identical fixed test-image directory.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .behavior import BehaviorProfile


def _iou(box_a: List[float], box_b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class ImageComparison:
    image: str
    num_reference: int
    num_candidate: int
    num_matched: int
    num_missing: int  # reference detections with no matching candidate detection
    num_added: int     # candidate detections with no matching reference detection
    num_class_flips: int  # matched pairs where the predicted class disagrees
    confidence_diffs: List[float] = field(default_factory=list)
    added_class_ids: List[int] = field(default_factory=list)
    added_confidences: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "num_reference": self.num_reference,
            "num_candidate": self.num_candidate,
            "num_matched": self.num_matched,
            "num_missing": self.num_missing,
            "num_added": self.num_added,
            "num_class_flips": self.num_class_flips,
            "mean_confidence_diff": (
                round(sum(self.confidence_diffs) / len(self.confidence_diffs), 4)
                if self.confidence_diffs else None
            ),
        }


@dataclass
class ComparisonResult:
    per_image: List[ImageComparison]
    total_reference: int
    total_candidate: int
    total_matched: int
    total_missing: int
    total_added: int
    total_class_flips: int
    agreement_rate: float  # Dice-style: 2*matched / (total_ref + total_cand), 1.0 if both empty
    mean_confidence_diff: Optional[float]
    class_flip_rate: Optional[float]  # class_flips / matched
    missing_rate: Optional[float]     # missing / total_reference
    added_rate: Optional[float]       # added / total_candidate
    dominant_added_class_id: Optional[int]
    dominant_added_class_fraction: Optional[float]  # fraction of ALL added detections that are this class
    dominant_added_class_mean_confidence: Optional[float]
    skipped_images: List[str] = field(default_factory=list)  # present in only one profile

    def to_dict(self) -> dict:
        return {
            "total_reference_detections": self.total_reference,
            "total_candidate_detections": self.total_candidate,
            "total_matched": self.total_matched,
            "total_missing": self.total_missing,
            "total_added": self.total_added,
            "total_class_flips": self.total_class_flips,
            "agreement_rate": round(self.agreement_rate, 4),
            "mean_confidence_diff": (
                round(self.mean_confidence_diff, 4) if self.mean_confidence_diff is not None else None
            ),
            "class_flip_rate": (
                round(self.class_flip_rate, 4) if self.class_flip_rate is not None else None
            ),
            "missing_rate": round(self.missing_rate, 4) if self.missing_rate is not None else None,
            "added_rate": round(self.added_rate, 4) if self.added_rate is not None else None,
            "dominant_added_class_id": self.dominant_added_class_id,
            "dominant_added_class_fraction": (
                round(self.dominant_added_class_fraction, 4)
                if self.dominant_added_class_fraction is not None else None
            ),
            "dominant_added_class_mean_confidence": (
                round(self.dominant_added_class_mean_confidence, 4)
                if self.dominant_added_class_mean_confidence is not None else None
            ),
            "skipped_images": self.skipped_images,
            "per_image": [ic.to_dict() for ic in self.per_image],
        }


def _match_predictions(ref_preds: List[dict], cand_preds: List[dict], iou_threshold: float):
    """Greedily match reference and candidate predictions for one image by IoU.

    Returns (matched_pairs, unmatched_ref_indices, unmatched_cand_indices)
    where matched_pairs is a list of (ref_index, cand_index, iou).
    """
    candidates = []
    for ri, rp in enumerate(ref_preds):
        for ci, cp in enumerate(cand_preds):
            iou = _iou(rp["bbox"], cp["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, ri, ci))
    candidates.sort(key=lambda t: t[0], reverse=True)

    used_ref, used_cand = set(), set()
    matched_pairs = []
    for iou, ri, ci in candidates:
        if ri in used_ref or ci in used_cand:
            continue
        used_ref.add(ri)
        used_cand.add(ci)
        matched_pairs.append((ri, ci, iou))

    unmatched_ref = [i for i in range(len(ref_preds)) if i not in used_ref]
    unmatched_cand = [i for i in range(len(cand_preds)) if i not in used_cand]
    return matched_pairs, unmatched_ref, unmatched_cand


def compare_images(image_name: str, ref_preds: List[dict], cand_preds: List[dict],
                    iou_threshold: float = 0.5) -> ImageComparison:
    """Compare one image's reference vs. candidate predictions."""
    matched_pairs, unmatched_ref, unmatched_cand = _match_predictions(ref_preds, cand_preds, iou_threshold)

    confidence_diffs: List[float] = []
    num_class_flips = 0
    for ri, ci, _iou in matched_pairs:
        ref_p, cand_p = ref_preds[ri], cand_preds[ci]
        confidence_diffs.append(abs(float(cand_p["confidence"]) - float(ref_p["confidence"])))
        if ref_p["class_id"] != cand_p["class_id"]:
            num_class_flips += 1

    added_class_ids = [cand_preds[ci]["class_id"] for ci in unmatched_cand]
    added_confidences = [float(cand_preds[ci]["confidence"]) for ci in unmatched_cand]

    return ImageComparison(
        image=image_name,
        num_reference=len(ref_preds),
        num_candidate=len(cand_preds),
        num_matched=len(matched_pairs),
        num_missing=len(unmatched_ref),
        num_added=len(unmatched_cand),
        num_class_flips=num_class_flips,
        confidence_diffs=confidence_diffs,
        added_class_ids=added_class_ids,
        added_confidences=added_confidences,
    )


def compare_profiles(
    reference: BehaviorProfile,
    candidate: BehaviorProfile,
    iou_threshold: float = 0.5,
) -> ComparisonResult:
    """Compare a reference and candidate BehaviorProfile image-by-image.

    Only images present in both profiles (matched by filename) are
    compared; any image present in only one profile is recorded in
    ``skipped_images`` rather than silently ignored -- this normally
    means the two profiles weren't generated against the same fixed test
    set, which is a setup problem worth surfacing, not guessing around.
    """
    ref_by_image = {r["image"]: r for r in reference.image_results}
    cand_by_image = {r["image"]: r for r in candidate.image_results}

    common = sorted(set(ref_by_image) & set(cand_by_image))
    skipped = sorted(set(ref_by_image) ^ set(cand_by_image))

    per_image: List[ImageComparison] = []
    for image_name in common:
        per_image.append(
            compare_images(
                image_name,
                ref_by_image[image_name]["predictions"],
                cand_by_image[image_name]["predictions"],
                iou_threshold=iou_threshold,
            )
        )

    total_reference = sum(ic.num_reference for ic in per_image)
    total_candidate = sum(ic.num_candidate for ic in per_image)
    total_matched = sum(ic.num_matched for ic in per_image)
    total_missing = sum(ic.num_missing for ic in per_image)
    total_added = sum(ic.num_added for ic in per_image)
    total_class_flips = sum(ic.num_class_flips for ic in per_image)

    all_confidence_diffs = [d for ic in per_image for d in ic.confidence_diffs]
    mean_confidence_diff = (
        sum(all_confidence_diffs) / len(all_confidence_diffs) if all_confidence_diffs else None
    )
    class_flip_rate = (total_class_flips / total_matched) if total_matched > 0 else None
    missing_rate = (total_missing / total_reference) if total_reference > 0 else None
    added_rate = (total_added / total_candidate) if total_candidate > 0 else None

    denom = total_reference + total_candidate
    agreement_rate = (2.0 * total_matched / denom) if denom > 0 else 1.0

    dominant_class_id = None
    dominant_fraction = None
    dominant_mean_conf = None
    all_added_class_ids = [cid for ic in per_image for cid in ic.added_class_ids]
    all_added_confidences = [c for ic in per_image for c in ic.added_confidences]
    if all_added_class_ids:
        counts = Counter(all_added_class_ids)
        dominant_class_id, dominant_count = counts.most_common(1)[0]
        dominant_fraction = dominant_count / len(all_added_class_ids)
        dominant_confs = [
            c for cid, c in zip(all_added_class_ids, all_added_confidences) if cid == dominant_class_id
        ]
        dominant_mean_conf = sum(dominant_confs) / len(dominant_confs)

    return ComparisonResult(
        per_image=per_image,
        total_reference=total_reference,
        total_candidate=total_candidate,
        total_matched=total_matched,
        total_missing=total_missing,
        total_added=total_added,
        total_class_flips=total_class_flips,
        agreement_rate=agreement_rate,
        mean_confidence_diff=mean_confidence_diff,
        class_flip_rate=class_flip_rate,
        missing_rate=missing_rate,
        added_rate=added_rate,
        dominant_added_class_id=dominant_class_id,
        dominant_added_class_fraction=dominant_fraction,
        dominant_added_class_mean_confidence=dominant_mean_conf,
        skipped_images=skipped,
    )
