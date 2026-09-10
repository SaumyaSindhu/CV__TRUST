#!/usr/bin/env python3
"""
audit_model.py
----------------
Phase 3 CLI: run the Model Integrity Auditor end-to-end and save a
structured JSON report.

Usage
-----
Compare a candidate model against a trusted reference model:

    python audit_model.py \\
        --candidate path/to/candidate.pth \\
        --reference path/to/reference.pth

Compare against a previously recorded hash + cached behavioral baseline
(no need to keep the original reference weights file around):

    python audit_model.py \\
        --candidate path/to/candidate.pth \\
        --reference-hash <sha256> \\
        --reference-behavior results/model_audit/reference_behavior.json

See README.md ("Model Integrity Auditor") for the full option list, what
each finding type means, and how to interpret the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor.config import COCO_INSTANCE_CATEGORY_NAMES  # noqa: E402
from cv_auditor.model_audit import ModelAuditConfig, ModelAuditor  # noqa: E402
from cv_auditor.utils import get_logger  # noqa: E402

logger = get_logger("audit_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Model Integrity Auditor")
    parser.add_argument("--candidate", required=True, help="Path to the candidate model checkpoint to audit.")
    parser.add_argument("--reference", default=None, help="Path to a trusted reference model checkpoint.")
    parser.add_argument("--reference-hash", default=None, help="A previously recorded trusted SHA-256 hash.")
    parser.add_argument(
        "--reference-behavior", default="results/model_audit/reference_behavior.json",
        help="Where to save/load the reference model's behavioral baseline.",
    )
    parser.add_argument(
        "--regenerate-reference-behavior", action="store_true",
        help="Re-run the reference model even if a cached baseline already exists.",
    )
    parser.add_argument("--test-images", default="data/sample_images", help="Fixed set of test images.")
    parser.add_argument(
        "--architecture", default="fasterrcnn_mobilenet_v3_large_320_fpn",
        help="One of cv_auditor.config.SUPPORTED_ARCHITECTURES (torchvision "
             "detectors or 'yolov5' for Ultralytics YOLOv5 .pt checkpoints).",
    )
    parser.add_argument("--num-classes", type=int, default=91)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--detector-score-thresh", type=float, default=None,
        help="Override the detection model's internal score threshold. Useful when "
             "auditing untrained/randomly-initialized weights, which can otherwise "
             "produce zero detections regardless of --confidence-threshold.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--candidate-name", default="candidate_model")
    parser.add_argument("--reference-name", default="reference_model")
    parser.add_argument("--output", default="results/model_audit_report.json")

    suspicion = parser.add_argument_group("suspicious-behavior thresholds")
    suspicion.add_argument("--unusual-agreement-threshold", type=float, default=0.90)
    suspicion.add_argument("--unusual-confidence-diff-threshold", type=float, default=0.25)
    suspicion.add_argument("--unusual-class-flip-rate-threshold", type=float, default=0.10)
    suspicion.add_argument("--trojan-min-added-detections", type=int, default=5)
    suspicion.add_argument("--trojan-dominant-class-fraction", type=float, default=0.60)
    suspicion.add_argument("--trojan-dominant-class-min-confidence", type=float, default=0.85)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = ModelAuditConfig(
        candidate_model_path=args.candidate,
        reference_model_path=args.reference,
        reference_sha256=args.reference_hash,
        reference_behavior_path=args.reference_behavior or None,
        regenerate_reference_behavior=args.regenerate_reference_behavior,
        architecture=args.architecture,
        num_classes=args.num_classes,
        class_names=list(COCO_INSTANCE_CATEGORY_NAMES),
        test_images_dir=args.test_images,
        confidence_threshold=args.confidence_threshold,
        detector_score_thresh=args.detector_score_thresh,
        iou_threshold=args.iou_threshold,
        device=args.device,
        candidate_model_name=args.candidate_name,
        reference_model_name=args.reference_name,
        output_path=args.output,
        unusual_agreement_threshold=args.unusual_agreement_threshold,
        unusual_confidence_diff_threshold=args.unusual_confidence_diff_threshold,
        unusual_class_flip_rate_threshold=args.unusual_class_flip_rate_threshold,
        trojan_min_added_detections=args.trojan_min_added_detections,
        trojan_dominant_class_fraction=args.trojan_dominant_class_fraction,
        trojan_dominant_class_min_confidence=args.trojan_dominant_class_min_confidence,
    )

    logger.info("=== Phase 3 Model Integrity Auditor ===")

    if not config.reference_model_path and not config.reference_sha256 and (
        not config.reference_behavior_path or not Path(config.reference_behavior_path).exists()
    ):
        logger.warning(
            "No --reference, --reference-hash, or cached --reference-behavior given. "
            "The audit will only report the candidate's own fingerprint; no behavior "
            "comparison or suspicious-behavior detection will be possible."
        )

    try:
        report = ModelAuditor(config).run()
    except Exception as exc:
        logger.error("Model audit failed: %s", exc)
        return 1

    output_path = report.save(config.output_path)
    report.print_summary()
    print(f"\nFull structured report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
