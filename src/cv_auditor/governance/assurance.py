"""
assurance.py (governance)
--------------------------
The Phase 6 orchestration layer: runs any combination of the existing
Phase 2-5 auditors/verifiers over supplied assets, normalizes their
findings (finding_model), applies the governance policy (policy), and
emits ONE unified assurance report -- then records the whole assessment
in Phase 4's tamper-evident hash-chained audit log.

Nothing algorithmic lives here: every analysis is delegated to the
already-tested phase modules, and the audit trail reuses Phase 4's
``AuditLog`` unmodified (an assessment is logged as a record-shaped
{payload.record_id, protection.record_digest} entry whose digest is the
canonical SHA-256 of the report core, so editing an earlier assessment
entry breaks the chain exactly like editing a provenance entry does).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__ as project_version
from ..provenance import AuditLog
from ..provenance.canonical import sha256_of_obj
from ..utils import ensure_parent_dir, find_images, get_logger, sha256_of_file, utc_timestamp
from . import finding_model, policy

logger = get_logger(__name__)

ASSURANCE_SCHEMA_VERSION = "1.0"

UNSUPPORTED_ATTACK_CLASSES = [
    "Unknown/novel attacks in general: detection is evidence-based, not exhaustive.",
    "Adaptive attacks crafted to evade these specific detectors and features.",
    "Sophisticated novel backdoors whose triggers lie outside the fixed test set.",
    "Inferring malicious INTENT from statistical anomalies (out of scope by design).",
    "Cryptographic proof of model correctness or accuracy.",
    "Secure hardware attestation of the execution environment.",
    "Complete backdoor trigger reconstruction.",
    "Global replay prevention without external/shared infrastructure.",
]

GLOBAL_LIMITATIONS = [
    "Hashing proves content identity, not authorship.",
    "The HMAC signature scheme is symmetric: any party able to verify can also sign.",
    "Valid provenance does not prove the model is accurate or benign.",
    "Distribution shift can be entirely legitimate (terrain, season, sensor, illumination).",
    "A dataset anomaly does not automatically mean poisoning.",
    "Statistical anomalies are evidence for review, never proof of malicious activity.",
]


def _dataset_content_hash(dataset_dir: str) -> Optional[str]:
    """Deterministic content hash of a directory of images: SHA-256 over
    the sorted (filename, file-sha256) pairs."""
    root = Path(dataset_dir)
    if (root / "images").is_dir():  # yolo_folder layout hashes the images
        root = root / "images"
    try:
        files = find_images(str(root))
    except (FileNotFoundError, ValueError):
        return None
    pairs = [[Path(p).name, sha256_of_file(p)] for p in sorted(files)]
    return sha256_of_obj(pairs)


@dataclass
class AssuranceConfig:
    """What to assess. Every section is optional; the engine runs the
    phases whose inputs are provided and reports the rest as not run."""

    # phase 2 -- dataset
    dataset_dir: Optional[str] = None
    dataset_format: str = "yolo_folder"

    # phase 3 -- model
    candidate_model_path: Optional[str] = None
    reference_model_path: Optional[str] = None
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    test_images_dir: str = "data/sample_images"
    confidence_threshold: float = 0.5
    detector_score_thresh: Optional[float] = None

    # phase 4 -- provenance record to verify
    inference_record_path: Optional[str] = None
    provenance_key_path: Optional[str] = None
    provenance_image_path: Optional[str] = None
    provenance_model_path: Optional[str] = None

    # phase 5 -- incoming data vs baseline
    incoming_dir: Optional[str] = None
    distribution_baseline_path: Optional[str] = None
    incoming_metadata: Optional[Dict] = None

    device: str = "cpu"
    workdir: str = "results/assurance"
    audit_log_path: str = "results/assurance/assurance_audit_log.jsonl"

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class UnifiedAssuranceReport:
    data: Dict[str, Any]
    path: Optional[str] = None

    @property
    def assessment_id(self) -> str:
        return self.data["assessment_id"]

    @property
    def recommendation(self) -> str:
        return self.data["governance"]["recommendation"]

    @property
    def overall_risk(self) -> str:
        return self.data["governance"]["overall_risk"]

    def to_dict(self) -> dict:
        return self.data

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        self.path = path
        return path

    def print_summary(self) -> None:
        g = self.data["governance"]
        print("\nUnified Assurance Summary\n")
        print(f"Assessment:      {self.data['assessment_id']}")
        for name, section in self.data["sections"].items():
            status = section["status"] if section else "not_run"
            print(f"  {name:<14s} {status}")
        print(f"\nFindings:        {len(self.data['findings'])}")
        print(f"Overall risk:    {g['overall_risk']}  (confidence {g['overall_confidence']})")
        print(f"Recommendation:  {g['recommendation']}")
        for reason in g["decision_reasons"]:
            print(f"  - {reason}")


class AssuranceEngine:
    """Dataset -> model -> provenance -> distribution -> governance ->
    unified report -> tamper-evident audit entry."""

    def __init__(self, config: AssuranceConfig):
        self.config = config

    # ------------------------------------------------------------ phases
    def _run_dataset(self) -> Optional[dict]:
        cfg = self.config
        if not cfg.dataset_dir:
            return None
        from ..dataset import DatasetAuditConfig, DatasetAuditor

        report = DatasetAuditor(DatasetAuditConfig(
            dataset_dir=cfg.dataset_dir,
            dataset_format=cfg.dataset_format,
            device=cfg.device,
            embedding_cache_dir=None,
            output_path=str(Path(cfg.workdir) / "dataset_audit_report.json"),
        )).run()
        return report.to_dict()

    def _run_model(self) -> Optional[dict]:
        cfg = self.config
        if not cfg.candidate_model_path:
            return None
        from ..model_audit import ModelAuditConfig, ModelAuditor

        report = ModelAuditor(ModelAuditConfig(
            candidate_model_path=cfg.candidate_model_path,
            reference_model_path=cfg.reference_model_path,
            reference_behavior_path=str(Path(cfg.workdir) / "reference_behavior.json"),
            regenerate_reference_behavior=True,
            architecture=cfg.architecture,
            test_images_dir=cfg.test_images_dir,
            confidence_threshold=cfg.confidence_threshold,
            detector_score_thresh=cfg.detector_score_thresh,
            device=cfg.device,
            output_path=str(Path(cfg.workdir) / "model_audit_report.json"),
        )).run()
        return report.to_dict()

    def _run_provenance(self) -> Optional[dict]:
        cfg = self.config
        if not cfg.inference_record_path:
            return None
        from ..provenance import ProvenanceVerifier, load_key, load_record

        key = load_key(cfg.provenance_key_path) if cfg.provenance_key_path else None
        try:
            record = load_record(cfg.inference_record_path)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            return {
                "summary": {"overall_status": "CORRUPTED_RECORD",
                            "hash_integrity": "invalid",
                            "signature_status": "not_checked",
                            "replay_status": "not_checked"},
                "findings": [{
                    "code": "CORRUPTED_RECORD", "component": "record",
                    "evidence": str(exc),
                    "explanation": "The inference record could not be loaded.",
                }],
            }
        report = ProvenanceVerifier(key=key).verify(
            record,
            record_path=cfg.inference_record_path,
            image_path=cfg.provenance_image_path,
            model_path=cfg.provenance_model_path,
        )
        report_dict = report.to_dict()
        # Additive convenience for the unified report/dashboard: the
        # component hashes AS RECORDED in the payload (whose integrity
        # the verifier just checked). Phase 4 itself is unchanged.
        payload = record.get("payload", {}) if isinstance(record, dict) else {}
        report_dict["recorded_hashes"] = {
            "input_sha256": (payload.get("input") or {}).get("image_sha256"),
            "model_sha256": (payload.get("model") or {}).get("model_sha256"),
            "config_sha256": sha256_of_obj({
                "preprocessing_config": payload.get("preprocessing_config"),
                "inference_config": payload.get("inference_config"),
            }) if payload else None,
            "output_sha256": (sha256_of_obj(payload.get("output"))
                              if payload.get("output") is not None else None),
        }
        return report_dict

    def _run_distribution(self) -> Optional[dict]:
        cfg = self.config
        if not (cfg.incoming_dir and cfg.distribution_baseline_path):
            return None
        from ..distribution import (
            DistributionAuditor, DistributionBaseline, DistributionConfig,
        )

        baseline = DistributionBaseline.load(cfg.distribution_baseline_path)
        report = DistributionAuditor(DistributionConfig(
            baseline_path=cfg.distribution_baseline_path,
            device=cfg.device,
        )).run(cfg.incoming_dir, baseline=baseline,
               current_metadata=cfg.incoming_metadata)
        return report.to_dict()

    # ------------------------------------------------------- section views
    @staticmethod
    def _section(status: Optional[str], summary: Optional[dict],
                 report: Optional[dict]) -> Optional[dict]:
        if report is None:
            return None
        return {"status": status, "summary": summary, "report": report}

    # ------------------------------------------------------------- run
    def run(self) -> UnifiedAssuranceReport:
        cfg = self.config
        Path(cfg.workdir).mkdir(parents=True, exist_ok=True)
        assessment_id = uuid.uuid4().hex
        started_at = utc_timestamp()

        dataset_report = self._run_dataset()
        model_report = self._run_model()
        provenance_report = self._run_provenance()
        distribution_report = self._run_distribution()

        asset = {
            "dataset_dir": cfg.dataset_dir,
            "dataset_hash": _dataset_content_hash(cfg.dataset_dir) if cfg.dataset_dir else None,
            "candidate_model_path": cfg.candidate_model_path,
            "model_hash": (sha256_of_file(cfg.candidate_model_path)
                           if cfg.candidate_model_path
                           and Path(cfg.candidate_model_path).exists() else None),
            "reference_model_path": cfg.reference_model_path,
            "inference_record_path": cfg.inference_record_path,
            "incoming_dir": cfg.incoming_dir,
            "configuration_hash": sha256_of_obj(cfg.to_dict()),
        }

        findings: List[finding_model.GovernanceFinding] = []
        if dataset_report:
            findings += finding_model.from_dataset_report(
                dataset_report, cfg.dataset_dir or "dataset")
        if model_report:
            findings += finding_model.from_model_report(
                model_report, cfg.candidate_model_path or "model")
        if provenance_report:
            findings += finding_model.from_provenance_report(
                provenance_report, cfg.inference_record_path or "inference_record")
        if distribution_report:
            findings += finding_model.from_distribution_report(
                distribution_report, cfg.incoming_dir or "incoming_batch")
        finding_dicts = [f.to_dict() for f in findings]

        sections = {
            "dataset": self._section(
                (dataset_report or {}).get("summary", {}).get(
                    "overall_dataset_finding_level"),
                (dataset_report or {}).get("summary"), dataset_report),
            "model": self._section(
                (model_report or {}).get("summary", {}).get("category"),
                (model_report or {}).get("summary"), model_report),
            "provenance": self._section(
                (provenance_report or {}).get("summary", {}).get("overall_status"),
                (provenance_report or {}).get("summary"), provenance_report),
            "distribution": self._section(
                (distribution_report or {}).get("risk"),
                (distribution_report or {}).get("summary"), distribution_report),
        }
        section_statuses = {k: (v["summary"] if v else None)
                            for k, v in sections.items()}

        overall_risk, overall_confidence, recommendation, reasons = (
            policy.overall_decision(finding_dicts, section_statuses)
        )

        core = {
            "schema_version": ASSURANCE_SCHEMA_VERSION,
            "assessment_id": assessment_id,
            "run_metadata": {
                "tool": "cv_auditor.governance",
                "tool_version": project_version,
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "config": cfg.to_dict(),
            },
            "assets": asset,
            "sections": sections,
            "findings": finding_dicts,
            "governance": {
                "overall_risk": overall_risk,
                "overall_confidence": overall_confidence,
                "recommendation": recommendation,
                "decision_reasons": reasons,
                "analyst_decision": None,
            },
            "limitations": list(GLOBAL_LIMITATIONS),
            "unsupported_attack_classes": list(UNSUPPORTED_ATTACK_CLASSES),
        }

        audit_entry = self._append_audit_entry(core)
        core["audit"] = {
            "log_path": cfg.audit_log_path,
            "entry_index": audit_entry["entry_index"],
            "entry_hash": audit_entry["entry_hash"],
            "record_digest": audit_entry["record_digest"],
        }
        report = UnifiedAssuranceReport(core)
        logger.info(
            "Assurance assessment %s: risk=%s recommendation=%s findings=%d",
            assessment_id, overall_risk, recommendation, len(finding_dicts),
        )
        return report

    # ------------------------------------------------------- audit trail
    def _append_audit_entry(self, core: dict) -> dict:
        """Record the assessment in Phase 4's hash-chained audit log.

        The logged digest canonically covers the decision-relevant core
        (assessment id, asset hashes, findings, governance outcome), so a
        later edit of the stored report is detectable against the chain.
        """
        digest_payload = {
            "assessment_id": core["assessment_id"],
            "assets": core["assets"],
            "findings": core["findings"],
            "governance": core["governance"],
            "schema_version": core["schema_version"],
        }
        record_like = {
            "payload": {"record_id": core["assessment_id"],
                        "kind": "assurance_assessment"},
            "protection": {"record_digest": sha256_of_obj(digest_payload)},
        }
        return AuditLog(self.config.audit_log_path).append_record(record_like)

    # ---------------------------------------------------- analyst decision
    @staticmethod
    def record_analyst_decision(report_path: str, decision: str,
                                analyst: str, audit_log_path: str,
                                note: str = "") -> dict:
        """Attach a final human decision to a stored assessment and log it
        as its own chained audit entry (decisions are themselves auditable)."""
        if decision not in ("ACCEPT", "REVIEW", "QUARANTINE"):
            raise ValueError("decision must be ACCEPT, REVIEW or QUARANTINE")
        with open(report_path) as f:
            data = json.load(f)
        entry = {
            "decision": decision, "analyst": analyst, "note": note,
            "decided_at": utc_timestamp(),
        }
        data["governance"]["analyst_decision"] = entry
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)

        record_like = {
            "payload": {"record_id": f"{data['assessment_id']}-decision",
                        "kind": "analyst_decision"},
            "protection": {"record_digest": sha256_of_obj(
                {"assessment_id": data["assessment_id"], **entry})},
        }
        AuditLog(audit_log_path).append_record(record_like)
        return entry
