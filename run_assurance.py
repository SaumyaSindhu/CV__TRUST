#!/usr/bin/env python3
"""
run_assurance.py
-----------------
Phase 6 CLI -- run the full assurance workflow over any combination of
assets and produce ONE unified, audit-logged report.

    python run_assurance.py \
        --dataset data/sample_dataset \
        --candidate-model candidate.pth --reference-model reference.pth \
        --architecture fasterrcnn_mobilenet_v3_large_320_fpn \
        --test-images data/sample_images --detector-score-thresh 0.0 \
        --inference-record results/provenance/record.json \
        --provenance-key results/provenance/hmac.key \
        --incoming data/incoming_batch \
        --distribution-baseline results/distribution/baseline.json \
        --output results/assurance/assurance_report.json

Every section is optional -- provide only the assets you want assessed.

Recording a final analyst decision on a stored report (also audit-logged):

    python run_assurance.py decide \
        --report results/assurance/assurance_report.json \
        --decision QUARANTINE --analyst "a.sharma" --note "escalated"

Exit codes: 0 = ACCEPT, 1 = REVIEW/QUARANTINE or error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor.governance import AssuranceConfig, AssuranceEngine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Unified assurance workflow (Phase 6)")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run an assurance assessment (default).")
    _add_run_args(run)
    _add_run_args(parser)  # allow omitting the 'run' subcommand

    dec = sub.add_parser("decide", help="Record a final analyst decision.")
    dec.add_argument("--report", required=True)
    dec.add_argument("--decision", required=True,
                     choices=["ACCEPT", "REVIEW", "QUARANTINE"])
    dec.add_argument("--analyst", required=True)
    dec.add_argument("--note", default="")
    dec.add_argument("--audit-log", default="results/assurance/assurance_audit_log.jsonl")

    return parser.parse_args()


def _add_run_args(p):
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-format", default="yolo_folder")
    p.add_argument("--candidate-model", default=None)
    p.add_argument("--reference-model", default=None)
    p.add_argument("--architecture", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    p.add_argument("--test-images", default="data/sample_images")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--detector-score-thresh", type=float, default=None)
    p.add_argument("--inference-record", default=None)
    p.add_argument("--provenance-key", default=None)
    p.add_argument("--provenance-image", default=None)
    p.add_argument("--provenance-model", default=None)
    p.add_argument("--incoming", default=None)
    p.add_argument("--distribution-baseline", default=None)
    p.add_argument("--incoming-metadata", default=None,
                   help="JSON file with the incoming batch's declared conditions.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workdir", default="results/assurance")
    p.add_argument("--audit-log", default="results/assurance/assurance_audit_log.jsonl")
    p.add_argument("--output", default="results/assurance/assurance_report.json")


def main() -> int:
    args = parse_args()

    if args.command == "decide":
        entry = AssuranceEngine.record_analyst_decision(
            args.report, args.decision, args.analyst,
            audit_log_path=args.audit_log, note=args.note,
        )
        print(f"Recorded analyst decision {entry['decision']} by "
              f"{entry['analyst']} at {entry['decided_at']} (audit-logged).")
        return 0

    metadata = None
    if args.incoming_metadata:
        with open(args.incoming_metadata) as f:
            metadata = json.load(f)

    config = AssuranceConfig(
        dataset_dir=args.dataset,
        dataset_format=args.dataset_format,
        candidate_model_path=args.candidate_model,
        reference_model_path=args.reference_model,
        architecture=args.architecture,
        test_images_dir=args.test_images,
        confidence_threshold=args.confidence_threshold,
        detector_score_thresh=args.detector_score_thresh,
        inference_record_path=args.inference_record,
        provenance_key_path=args.provenance_key,
        provenance_image_path=args.provenance_image,
        provenance_model_path=args.provenance_model,
        incoming_dir=args.incoming,
        distribution_baseline_path=args.distribution_baseline,
        incoming_metadata=metadata,
        device=args.device,
        workdir=args.workdir,
        audit_log_path=args.audit_log,
    )
    report = AssuranceEngine(config).run()
    report.save(args.output)
    report.print_summary()
    print(f"\nFull unified report saved to: {args.output}")
    print(f"Audit log: {args.audit_log} (entry {report.data['audit']['entry_index']})")
    return 0 if report.recommendation == "ACCEPT" else 1


if __name__ == "__main__":
    sys.exit(main())
