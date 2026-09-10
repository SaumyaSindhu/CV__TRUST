#!/usr/bin/env python3
"""
audit_inference.py
-------------------
Phase 4 CLI: create, verify, and audit-log cryptographically protected
inference provenance records.

Usage
-----
Generate a signing key (optional, enables the authenticity layer):

    python audit_inference.py keygen --key results/provenance/signing.key

Run one inference and create a protected provenance record (also appends
to the audit log and writes the Phase 1-style prediction JSON):

    python audit_inference.py create \\
        --image data/sample_images/image_001.jpg \\
        --model data/sample_model/sample_model.pth \\
        --output results/provenance/prediction.json \\
        --record results/provenance/inference_record.json \\
        [--config config.json] [--key results/provenance/signing.key]

Verify a protected record (hash integrity + optional signature, files,
and replay protection):

    python audit_inference.py verify \\
        --record results/provenance/inference_record.json \\
        [--image path/to/image.jpg] [--model path/to/model.pth] \\
        [--key results/provenance/signing.key] \\
        [--check-replay] [--replay-registry results/provenance/replay_registry.json]

Verify the complete audit chain:

    python audit_inference.py verify-log --log results/provenance/audit_log.jsonl

Exit code is 0 when the result is VALID and 1 otherwise, so the commands
compose in shell scripts. See README.md ("Phase 4") for what each finding
code means and for the security limitations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor.config import PipelineConfig, SUPPORTED_ARCHITECTURES  # noqa: E402
from cv_auditor.utils import get_logger  # noqa: E402
from cv_auditor.provenance import (  # noqa: E402
    AuditLog,
    ProvenanceVerifier,
    ReplayRegistry,
    build_chain_report,
    generate_key,
    load_key,
    load_record,
    payload_from_inference,
    protect_payload,
    save_key,
    save_record,
    sign_record,
)

logger = get_logger("audit_inference")

DEFAULT_RECORD = "results/provenance/inference_record.json"
DEFAULT_LOG = "results/provenance/audit_log.jsonl"
DEFAULT_REGISTRY = "results/provenance/replay_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Inference Provenance & Output Integrity CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ----------------------------------------------------------- keygen
    p_key = sub.add_parser("keygen", help="Generate an HMAC signing key file.")
    p_key.add_argument("--key", default="results/provenance/signing.key",
                       help="Where to write the new key file.")
    p_key.add_argument("--force", action="store_true",
                       help="Overwrite an existing key file.")

    # ----------------------------------------------------------- create
    p_create = sub.add_parser(
        "create",
        help="Run inference on one image and create a protected provenance record.",
    )
    p_create.add_argument("--image", required=True, help="Path to the input image.")
    p_create.add_argument("--model", required=True,
                          help="Path to the model checkpoint (.pth) or TorchScript file (.pt).")
    p_create.add_argument("--output", default="results/provenance/prediction.json",
                          help="Where to write the Phase 1-style prediction JSON.")
    p_create.add_argument("--config", default=None,
                          help="Optional JSON file with PipelineConfig overrides "
                               "(e.g. architecture, confidence_threshold).")
    p_create.add_argument("--record", default=DEFAULT_RECORD,
                          help="Where to write the protected provenance record.")
    p_create.add_argument("--log", default=DEFAULT_LOG,
                          help="Append-only audit log to add this record to "
                               "(pass an empty string to skip logging).")
    p_create.add_argument("--key", default=None,
                          help="Optional signing key file (see 'keygen'); if given, "
                               "the record is HMAC-signed.")
    p_create.add_argument("--sequence", type=int, default=None,
                          help="Sequence number for the record; defaults to the "
                               "audit log's next index.")
    p_create.add_argument("--architecture", default="fasterrcnn_mobilenet_v3_large_320_fpn",
                          choices=SUPPORTED_ARCHITECTURES)
    p_create.add_argument("--model-name", default="reference_model")
    p_create.add_argument("--confidence-threshold", type=float, default=0.0)
    p_create.add_argument("--detector-score-thresh", type=float, default=None)
    p_create.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # ----------------------------------------------------------- verify
    p_verify = sub.add_parser("verify", help="Verify a protected provenance record.")
    p_verify.add_argument("--record", required=True, help="Path to the record JSON.")
    p_verify.add_argument("--image", default=None,
                          help="Optionally re-hash this image file against the record.")
    p_verify.add_argument("--model", default=None,
                          help="Optionally re-hash this model file against the record.")
    p_verify.add_argument("--key", default=None,
                          help="Verification key file; enables signature checking.")
    p_verify.add_argument("--check-replay", action="store_true",
                          help="Enable replay protection using --replay-registry.")
    p_verify.add_argument("--replay-registry", default=DEFAULT_REGISTRY,
                          help="Replay registry file (created if missing).")
    p_verify.add_argument("--output", default=None,
                          help="Optionally save the structured verification report JSON here.")

    # ------------------------------------------------------- verify-log
    p_log = sub.add_parser("verify-log", help="Verify the audit log's hash chain.")
    p_log.add_argument("--log", required=True, help="Path to the audit log (.jsonl).")
    p_log.add_argument("--output", default=None,
                       help="Optionally save the structured chain report JSON here.")

    return parser.parse_args()


# --------------------------------------------------------------------------
def cmd_keygen(args: argparse.Namespace) -> int:
    if Path(args.key).exists() and not args.force:
        logger.error("Key file %s already exists (use --force to overwrite).", args.key)
        return 1
    path = save_key(generate_key(), args.key)
    logger.info("Wrote new HMAC-SHA256 signing key to %s", path)
    print(f"Signing key written to {path}")
    print("Keep this file secret: anyone holding it can produce valid signatures.")
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    # The Phase 1 pipeline pieces are only needed by 'create', so import
    # them here rather than at module level.
    from cv_auditor import ImageLoader, InferenceEngine, ModelLoader, ResultFormatter

    overrides = {}
    if args.config:
        with open(args.config, "r") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            logger.error("--config must contain a JSON object of PipelineConfig fields.")
            return 1

    config_kwargs = dict(
        model_path=args.model,
        architecture=args.architecture,
        pretrained=False,
        model_name=args.model_name,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        detector_score_thresh=args.detector_score_thresh,
        input_dir=args.image,
        output_path=args.output,
    )
    config_kwargs.update(overrides)
    # CLI-positional facts always win over config-file leftovers.
    config_kwargs["model_path"] = args.model
    config_kwargs["input_dir"] = args.image
    config_kwargs["output_path"] = args.output

    try:
        config = PipelineConfig(**config_kwargs)
    except (TypeError, ValueError) as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    logger.info("=== Phase 4: create protected inference provenance record ===")

    # 1. Phase 1 pipeline: model + image -> structured prediction ---------
    try:
        model, model_metadata = ModelLoader(config).load()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 1
    try:
        image = ImageLoader(config.input_dir).load_one(args.image)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load image: %s", exc)
        return 1

    import torch  # local import, mirrors the lazy-import policy above

    engine = InferenceEngine(
        model=model,
        model_name=config.model_name,
        class_names=config.class_names,
        device=next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"),
        confidence_threshold=config.confidence_threshold,
    )
    image_result = engine.run_one(image)

    formatter = ResultFormatter(
        model_metadata=model_metadata.to_dict(), config=config.to_dict(),
    )
    formatter.add_result(image_result)
    formatter.save(args.output)

    # 2. Provenance record: bind image + model + configs + output ---------
    audit_log = AuditLog(args.log) if args.log else None
    sequence = args.sequence
    if sequence is None:
        sequence = audit_log.next_sequence_number() if audit_log else 0

    payload = payload_from_inference(
        image_result, model_metadata, config, sequence_number=sequence,
    )
    record = protect_payload(payload)

    # 3. Optional signature ------------------------------------------------
    if args.key:
        try:
            key = load_key(args.key)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Cannot sign record: %s", exc)
            return 1
        sign_record(record, key)
        logger.info("Record signed (HMAC-SHA256, key file %s).", args.key)
    else:
        logger.info(
            "Record NOT signed (no --key given): hash integrity only, no authenticity."
        )

    save_record(record, args.record)

    # 4. Audit log ---------------------------------------------------------
    if audit_log is not None:
        entry = audit_log.append_record(record)
        logger.info("Audit log entry %d appended to %s.", entry["entry_index"], args.log)

    print("\nProtected inference provenance record created")
    print(f"  Record:        {args.record}")
    print(f"  Record id:     {record['payload']['record_id']}")
    print(f"  Record digest: {record['protection']['record_digest']}")
    print(f"  Signed:        {'yes' if args.key else 'no'}")
    print(f"  Prediction:    {args.output}")
    if audit_log is not None:
        print(f"  Audit log:     {args.log}")
    print(f"  Predictions:   {len(image_result.predictions)} detection(s)\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    key = None
    if args.key:
        try:
            key = load_key(args.key)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Cannot load verification key: %s", exc)
            return 1

    registry = None
    if args.check_replay:
        try:
            registry = ReplayRegistry(args.replay_registry)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Cannot load replay registry: %s", exc)
            return 1

    try:
        record = load_record(args.record)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except json.JSONDecodeError as exc:
        # An unreadable record is still a structured result, not a crash.
        from cv_auditor.provenance.verification import (
            CORRUPTED_RECORD, VerificationFinding, VerificationReport,
        )
        from cv_auditor.utils import utc_timestamp

        report = VerificationReport(
            record_path=args.record,
            record_id=None,
            generated_at=utc_timestamp(),
            hash_integrity="invalid",
            signature_status="not_checked",
            replay_status="not_checked",
            findings=[
                VerificationFinding(
                    code=CORRUPTED_RECORD,
                    component="record",
                    evidence=f"file is not valid JSON: {exc}",
                    explanation="The record file cannot be parsed at all.",
                )
            ],
        )
        report.print_summary()
        if args.output:
            report.save(args.output)
        return 1

    verifier = ProvenanceVerifier(key=key, replay_registry=registry)
    report = verifier.verify(
        record,
        record_path=args.record,
        image_path=args.image,
        model_path=args.model,
    )
    report.print_summary()
    if args.output:
        report.save(args.output)
        logger.info("Saved verification report to %s", args.output)
    return 0 if report.is_valid else 1


def cmd_verify_log(args: argparse.Namespace) -> int:
    log = AuditLog(args.log)
    if not Path(args.log).exists():
        logger.error("Audit log not found: %s", args.log)
        return 1
    result = log.verify_chain()
    chain_report = build_chain_report(result, args.log)

    print("\nAudit Log Chain Verification\n")
    print(f"Log:            {args.log}")
    print(f"Entries:        {result.num_entries}")
    print(f"Chain head:     {result.head_hash}")
    print(f"Overall status: {chain_report['summary']['overall_status']}")
    if result.issues:
        print(f"\n{len(result.issues)} issue(s):")
        for issue in result.issues:
            print(f"  - line {issue.line_number}: [{issue.code}] {issue.detail}")
    print()

    if args.output:
        from cv_auditor.utils import ensure_parent_dir

        ensure_parent_dir(args.output)
        with open(args.output, "w") as f:
            json.dump(chain_report, f, indent=2)
        logger.info("Saved chain report to %s", args.output)
    return 0 if result.valid else 1


def main() -> int:
    args = parse_args()
    if args.command == "keygen":
        return cmd_keygen(args)
    if args.command == "create":
        return cmd_create(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "verify-log":
        return cmd_verify_log(args)
    return 2  # pragma: no cover - argparse enforces the choices


if __name__ == "__main__":
    raise SystemExit(main())
