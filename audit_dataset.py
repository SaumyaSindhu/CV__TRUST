#!/usr/bin/env python3
"""
audit_dataset.py
-----------------
Phase 2 CLI: run the Dataset Integrity Auditor end-to-end and save a
structured JSON report.

Usage
-----
Run against the bundled synthetic test dataset (default):

    python audit_dataset.py

Run against your own dataset:

    python audit_dataset.py --dataset path/to/dataset --output results/my_audit.json

See README.md ("Dataset Integrity Auditor") for the full option list,
what each detector means, and how to interpret the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the project root without installing the
# package first (same convention as demo.py).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor.dataset import DatasetAuditConfig, DatasetAuditor  # noqa: E402
from cv_auditor.utils import get_logger  # noqa: E402

logger = get_logger("audit_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Dataset Integrity Auditor")
    parser.add_argument(
        "--dataset", default="data/sample_dataset",
        help="Path to a dataset directory containing an images/ subfolder "
             "(and optionally a labels/ subfolder in YOLO format).",
    )
    parser.add_argument(
        "--output", default="results/dataset_audit_report.json",
        help="Where to write the structured JSON audit report.",
    )
    parser.add_argument(
        "--near-duplicate-threshold", type=float, default=0.90,
        help="Perceptual-hash similarity (0-1) at/above which two images "
             "are reported as near-duplicates.",
    )
    parser.add_argument(
        "--ood-threshold", type=float, default=2.0,
        help="Z-score (standard deviations above this dataset's own mean "
             "local distance) at/above which a sample is flagged as a "
             "potential OOD/anomaly finding.",
    )
    parser.add_argument(
        "--ood-k", type=int, default=5,
        help="Number of nearest neighbours used for the OOD/anomaly score.",
    )
    parser.add_argument(
        "--label-clusters", type=int, default=None,
        help="Number of k-means clusters used for label-anomaly detection. "
             "Defaults to the number of distinct class ids observed in the "
             "dataset if not set.",
    )
    parser.add_argument(
        "--embedding-cache-dir", default="results/embedding_cache",
        help="Directory for cached image embeddings. Pass '' to disable caching.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to run embedding extraction on.",
    )
    parser.add_argument(
        "--class-names", default=None,
        help="Optional path to a JSON file containing a list of class names "
             "(index = YOLO class_id).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    class_names = None
    if args.class_names:
        with open(args.class_names, "r") as f:
            class_names = json.load(f)

    config = DatasetAuditConfig(
        dataset_dir=args.dataset,
        output_path=args.output,
        near_duplicate_similarity_threshold=args.near_duplicate_threshold,
        ood_threshold=args.ood_threshold,
        ood_k=args.ood_k,
        label_cluster_count=args.label_clusters,
        embedding_cache_dir=args.embedding_cache_dir or None,
        device=args.device,
        class_names=class_names,
    )

    logger.info("=== Phase 2 Dataset Integrity Auditor ===")
    logger.info("Config: %s", json.dumps(config.to_dict(), indent=2, default=str))

    try:
        report = DatasetAuditor(config).run()
    except Exception as exc:
        logger.error("Dataset audit failed: %s", exc)
        return 1

    output_path = report.save(config.output_path)
    report.print_summary()
    print(f"\nFull structured report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
