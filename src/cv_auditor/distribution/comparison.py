"""
comparison.py (distribution)
----------------------------
Statistical comparison of an incoming batch against a stored baseline:

- per-image Mahalanobis distances + baseline-relative z-scores
- batch-level displacement (centroid distance, cosine distance,
  Mahalanobis of the batch mean, mean per-sample z)
- dispersion change (spread ratio)
- MMD^2 with an RBF kernel (median heuristic), reported as a ratio
  against a deterministic within-baseline null split
- calibrated 0-1 shift score, anomalous fraction, confidence, risk,
  and a drift-vs-anomaly shape classification

Everything is expressed relative to the baseline's OWN per-sample
distance distribution (its "self distances"), which is what calibrates
the thresholds: a z of 3 means "three baseline standard deviations
farther from the baseline centroid than the baseline's own images sit".

None of these numbers establishes intent. A strong shift can be winter
vs summer or camera A vs camera B just as easily as manipulation; the
``shift_pattern`` classification exists precisely to separate uniform
domain-style drift from concentrated per-sample anomalies -- as far as
statistics allows, and no further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .baseline import DistributionBaseline, mahalanobis_distances
from .config import DistributionConfig

RISK_LOW, RISK_MEDIUM, RISK_HIGH = "LOW", "MEDIUM", "HIGH"
PATTERN_NONE = "no_significant_shift"
PATTERN_UNIFORM = "uniform_shift"            # domain-drift-like
PATTERN_CONCENTRATED = "concentrated_anomaly"  # a few extreme outliers
PATTERN_MIXED = "mixed_shift"


# --------------------------------------------------------------------- #
# kernel two-sample statistics
# --------------------------------------------------------------------- #
def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    ).clip(min=0.0)


def rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Biased RBF-kernel MMD^2 estimate (fine for evidence purposes)."""
    kxx = np.exp(-gamma * _pairwise_sq_dists(x, x)).mean()
    kyy = np.exp(-gamma * _pairwise_sq_dists(y, y)).mean()
    kxy = np.exp(-gamma * _pairwise_sq_dists(x, y)).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def _median_heuristic_gamma(reference: np.ndarray) -> float:
    d2 = _pairwise_sq_dists(reference, reference)
    upper = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(upper)) if upper.size else 1.0
    return 1.0 / max(med, 1e-9)


# --------------------------------------------------------------------- #
@dataclass
class SampleScore:
    image: str
    mahalanobis: float
    z: float
    anomalous: bool

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "mahalanobis": round(self.mahalanobis, 4),
            "z": round(self.z, 4),
            "anomalous": self.anomalous,
        }


@dataclass
class ShiftComparison:
    """All comparison evidence for one batch-vs-baseline analysis."""

    num_current: int
    sample_scores: List[SampleScore]
    anomalous_fraction: float

    centroid_l2: float
    centroid_cosine_distance: float
    centroid_mahalanobis: float
    batch_shift_z: float                 # mean per-sample z (per-sample scale)
    dispersion_ratio: float              # current spread / baseline spread
    mmd2: Optional[float]
    mmd2_null: Optional[float]
    mmd_ratio: Optional[float]

    shift_score: float
    shift_pattern: str
    risk: str
    confidence: float
    declared_changes: Dict[str, Dict] = field(default_factory=dict)

    def evidence_dict(self) -> dict:
        return {
            "centroid_l2": round(self.centroid_l2, 4),
            "centroid_cosine_distance": round(self.centroid_cosine_distance, 4),
            "centroid_mahalanobis": round(self.centroid_mahalanobis, 4),
            "batch_shift_z": round(self.batch_shift_z, 4),
            "dispersion_ratio": round(self.dispersion_ratio, 4),
            "mmd2": round(self.mmd2, 6) if self.mmd2 is not None else None,
            "mmd2_within_baseline_null": (
                round(self.mmd2_null, 6) if self.mmd2_null is not None else None
            ),
            "mmd_ratio_vs_null": (
                round(self.mmd_ratio, 3) if self.mmd_ratio is not None else None
            ),
            "anomalous_sample_count": sum(s.anomalous for s in self.sample_scores),
            "shift_pattern": self.shift_pattern,
            "declared_metadata_changes": self.declared_changes,
            "per_sample": [s.to_dict() for s in self.sample_scores],
        }


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(1.0 - float(a @ b) / (na * nb))


def _declared_changes(baseline_meta: Optional[Dict],
                      current_meta: Optional[Dict]) -> Dict[str, Dict]:
    """Keys present in both metadata dicts whose values differ -- i.e.
    acquisition-condition changes the operator DECLARED (sensor, season,
    location, ...). Purely contextual: never proves or excuses anything,
    but a declared sensor swap makes a uniform shift *expected*."""
    if not baseline_meta or not current_meta:
        return {}
    changes = {}
    for key in sorted(set(baseline_meta) & set(current_meta)):
        if baseline_meta[key] != current_meta[key]:
            changes[key] = {"baseline": baseline_meta[key], "current": current_meta[key]}
    return changes


def _classify_pattern(sample_z: np.ndarray, anomalous: np.ndarray,
                      batch_shift_z: float, cfg: DistributionConfig) -> str:
    if batch_shift_z < cfg.batch_shift_z_medium and anomalous.mean() < cfg.anomalous_fraction_medium:
        return PATTERN_NONE
    frac = float(anomalous.mean()) if len(anomalous) else 0.0
    if frac >= 0.75:
        # essentially the whole batch moved together -> domain-style drift
        return PATTERN_UNIFORM
    if 0 < frac < 0.5 and batch_shift_z < cfg.batch_shift_z_high:
        # bulk of the batch still matches the baseline; a subset is far off
        return PATTERN_CONCENTRATED
    return PATTERN_MIXED


def _confidence(n_baseline: int, n_current: int, pretrained: bool,
                corrupt_fraction: float) -> float:
    n = min(n_baseline, n_current)
    if n <= 0:
        return 0.0
    if n < 5:
        base = 0.35
    elif n < 10:
        base = 0.55
    elif n < 20:
        base = 0.75
    else:
        base = 0.90
    if not pretrained:
        # handcrafted 16-dim features are honest but coarse
        base *= 0.85
    base *= (1.0 - min(corrupt_fraction, 0.9))
    return round(base, 3)


def _risk(batch_shift_z: float, anomalous_fraction: float,
          anomalous_count: int, cfg: DistributionConfig) -> str:
    if batch_shift_z >= cfg.batch_shift_z_high:
        return RISK_HIGH
    if (anomalous_fraction >= cfg.anomalous_fraction_high
            and anomalous_count >= cfg.min_anomalous_count):
        return RISK_HIGH
    if batch_shift_z >= cfg.batch_shift_z_medium:
        return RISK_MEDIUM
    if (anomalous_fraction >= cfg.anomalous_fraction_medium
            and anomalous_count >= cfg.min_anomalous_count):
        return RISK_MEDIUM
    return RISK_LOW


def compare_batch(
    baseline: DistributionBaseline,
    features: np.ndarray,
    image_paths: List[str],
    cfg: DistributionConfig,
    pretrained: bool,
    corrupt_fraction: float = 0.0,
    baseline_metadata: Optional[Dict] = None,
    current_metadata: Optional[Dict] = None,
) -> ShiftComparison:
    """Compare a featurized batch against the baseline."""
    if features.size == 0:
        return ShiftComparison(
            num_current=0, sample_scores=[], anomalous_fraction=0.0,
            centroid_l2=0.0, centroid_cosine_distance=0.0,
            centroid_mahalanobis=0.0, batch_shift_z=0.0, dispersion_ratio=1.0,
            mmd2=None, mmd2_null=None, mmd_ratio=None,
            shift_score=0.0, shift_pattern=PATTERN_NONE, risk=RISK_LOW,
            confidence=0.0,
            declared_changes=_declared_changes(
                baseline_metadata if baseline_metadata is not None else baseline.metadata,
                current_metadata,
            ),
        )

    # per-sample distances, calibrated by the baseline's self-distances
    d = mahalanobis_distances(features, baseline.centroid, baseline.inv_covariance)
    z = (d - baseline.self_distance_mean) / baseline.self_distance_std
    anomalous = z >= cfg.sample_anomaly_z

    sample_scores = [
        SampleScore(image=path, mahalanobis=float(di), z=float(zi),
                    anomalous=bool(ai))
        for path, di, zi, ai in zip(image_paths, d, z, anomalous)
    ]
    anomalous_fraction = float(anomalous.mean())
    anomalous_count = int(anomalous.sum())

    # batch-level displacement
    cur_centroid = features.mean(axis=0)
    centroid_l2 = float(np.linalg.norm(cur_centroid - baseline.centroid))
    centroid_cos = _cosine_distance(cur_centroid, baseline.centroid)
    centroid_mahal = float(
        mahalanobis_distances(cur_centroid[None, :], baseline.centroid,
                              baseline.inv_covariance)[0]
    )
    batch_shift_z = float(z.mean())

    # dispersion: current spread around the BASELINE centroid, vs baseline's own
    dispersion_ratio = float(
        (d.std(ddof=1) if len(d) > 1 else 0.0) / baseline.self_distance_std
    ) if baseline.self_distance_std > 0 else 1.0

    # MMD vs a deterministic within-baseline null split
    mmd2 = mmd2_null = mmd_ratio = None
    ref = baseline.stored_embeddings
    if len(ref) >= 4 and len(features) >= 2:
        gamma = _median_heuristic_gamma(ref)
        mmd2 = rbf_mmd2(ref, features, gamma)
        mmd2_null = rbf_mmd2(ref[0::2], ref[1::2], gamma)
        mmd_ratio = float(mmd2 / max(mmd2_null, 1e-9))

    # calibrated 0-1 score: monotone squash of batch displacement,
    # never below what the anomalous fraction alone implies
    shift_score = 1.0 - math.exp(-max(0.0, batch_shift_z) / 3.0)
    shift_score = round(max(shift_score, min(anomalous_fraction, 0.99)), 4)

    pattern = _classify_pattern(z, anomalous, batch_shift_z, cfg)
    risk = _risk(batch_shift_z, anomalous_fraction, anomalous_count, cfg)
    confidence = _confidence(
        baseline.num_samples, len(features), pretrained, corrupt_fraction,
    )

    return ShiftComparison(
        num_current=len(features),
        sample_scores=sample_scores,
        anomalous_fraction=round(anomalous_fraction, 4),
        centroid_l2=centroid_l2,
        centroid_cosine_distance=centroid_cos,
        centroid_mahalanobis=centroid_mahal,
        batch_shift_z=batch_shift_z,
        dispersion_ratio=dispersion_ratio,
        mmd2=mmd2, mmd2_null=mmd2_null, mmd_ratio=mmd_ratio,
        shift_score=shift_score,
        shift_pattern=pattern,
        risk=risk,
        confidence=confidence,
        declared_changes=_declared_changes(
            baseline_metadata if baseline_metadata is not None else baseline.metadata,
            current_metadata,
        ),
    )
