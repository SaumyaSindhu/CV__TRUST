#!/usr/bin/env python3
"""
audit_distribution.py
----------------------
Phase 5 CLI -- Distribution Shift & Anomaly Detection.

    # build a trusted baseline from reference images
    python audit_distribution.py build-baseline \
        --images data/reference_images \
        --baseline results/distribution/baseline.json \
        [--metadata metadata.json]

    # audit an incoming batch against the baseline
    python audit_distribution.py audit \
        --images data/incoming_batch \
        --baseline results/distribution/baseline.json \
        --output results/distribution/distribution_report.json \
        [--metadata batch_metadata.json]

Exit codes for 'audit': 0 = LOW risk, 1 = MEDIUM/HIGH risk or error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor.distribution import (  # noqa: E402
    DistributionAuditor,
    DistributionBaseline,
    DistributionConfig,
    build_baseline,
)


def _load_metadata(path):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-baseline", help="Build a trusted baseline distribution.")
    b.add_argument("--images", required=True, help="Directory of reference images.")
    b.add_argument("--baseline", default="results/distribution/baseline.json")
    b.add_argument("--metadata", default=None,
                   help="Optional JSON file with declared acquisition "
                        "conditions (sensor, season, location, ...).")
    b.add_argument("--min-samples", type=int, default=4)

    a = sub.add_parser("audit", help="Audit an incoming batch against the baseline.")
    a.add_argument("--images", required=True, help="Directory of incoming images.")
    a.add_argument("--baseline", default="results/distribution/baseline.json")
    a.add_argument("--metadata", default=None,
                   help="Optional JSON file with the batch's declared "
                        "acquisition conditions.")
    a.add_argument("--output", default="results/distribution/distribution_report.json")
    a.add_argument("--sample-anomaly-z", type=float, default=3.0)
    a.add_argument("--shift-z-medium", type=float, default=1.5)
    a.add_argument("--shift-z-high", type=float, default=3.5)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "build-baseline":
        cfg = DistributionConfig(baseline_path=args.baseline,
                                 min_baseline_samples=args.min_samples)
        baseline = build_baseline(args.images, cfg,
                                  metadata=_load_metadata(args.metadata))
        baseline.save(args.baseline)
        print(f"\nBaseline built from {baseline.num_samples} image(s)")
        print(f"  Feature backend: {baseline.feature_backend}")
        print(f"  Saved to:        {args.baseline}")
        return 0

    cfg = DistributionConfig(
        baseline_path=args.baseline,
        sample_anomaly_z=args.sample_anomaly_z,
        batch_shift_z_medium=args.shift_z_medium,
        batch_shift_z_high=args.shift_z_high,
        output_path=args.output,
    )
    baseline = DistributionBaseline.load(args.baseline)
    report = DistributionAuditor(cfg).run(
        args.images, baseline=baseline,
        current_metadata=_load_metadata(args.metadata),
    )
    report.save(args.output)
    report.print_summary()
    print(f"\nFull structured report saved to: {args.output}")
    return 0 if report.risk == "LOW" else 1


if __name__ == "__main__":
    sys.exit(main())
