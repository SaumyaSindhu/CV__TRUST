"""
verification.py (provenance)
-----------------------------
Verifies protected inference records and turns every detected integrity
problem into a structured finding. As everywhere else in this project, a
finding is *evidence*, not a verdict about intent -- a TAMPERED_* finding
means "this content no longer matches what was protected", whether the
cause was malice, a disk error, or an operator mistake.

Finding codes
-------------
==================  =====================================================
VALID               (overall status only) no findings of any kind
CORRUPTED_RECORD    record is structurally unreadable / not a v1.0 record
TAMPERED_INPUT      input section changed, or the referenced image file
                    no longer matches the recorded image SHA-256
TAMPERED_MODEL      model section changed, or the referenced weights file
                    no longer matches the recorded model SHA-256
TAMPERED_CONFIG     preprocessing or inference configuration changed
TAMPERED_OUTPUT     prediction/output changed
TAMPERED_METADATA   record id / timestamps / sequence / nonce / software
                    info changed, or the protection block itself was
                    altered
INVALID_SIGNATURE   signature fails against the given key, uses an
                    unknown scheme, or is absent although the verifier
                    was given a key (i.e. authenticity was expected)
REPLAY_DETECTED     replay protection enabled and this record (by id or
                    nonce) was already accepted before
==================  =====================================================

Multiple findings are all reported -- nothing is hidden behind the first
failure.

Two distinct verification layers (never conflated in the report):

* ``hash_integrity``: does the content match the digests? Detects
  *inconsistent* tampering only -- an attacker can recompute every digest.
* ``signature``: does the (recomputed!) record digest carry a valid HMAC
  under the verifier's key? Detects consistent re-forging by anyone who
  does not hold the key. Only meaningful when a key is supplied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import ensure_parent_dir, get_logger, sha256_of_file, utc_timestamp
from .audit_log import ChainVerificationResult
from .canonical import is_sha256_hex, sha256_of_obj
from .record import (
    ALL_COMPONENTS,
    COMPONENT_SECTIONS,
    RECORD_FORMAT_VERSION,
    compute_component_digests,
    compute_record_digest,
    metadata_component,
)
from .replay import ReplayRegistry
from .signing import SIGNATURE_SCHEME, verify_digest_signature

logger = get_logger(__name__)

# Overall/finding status codes (see module docstring).
VALID = "VALID"
CORRUPTED_RECORD = "CORRUPTED_RECORD"
TAMPERED_INPUT = "TAMPERED_INPUT"
TAMPERED_MODEL = "TAMPERED_MODEL"
TAMPERED_CONFIG = "TAMPERED_CONFIG"
TAMPERED_OUTPUT = "TAMPERED_OUTPUT"
TAMPERED_METADATA = "TAMPERED_METADATA"
INVALID_SIGNATURE = "INVALID_SIGNATURE"
REPLAY_DETECTED = "REPLAY_DETECTED"

FINDING_CODES = (
    CORRUPTED_RECORD,
    TAMPERED_INPUT,
    TAMPERED_MODEL,
    TAMPERED_CONFIG,
    TAMPERED_OUTPUT,
    TAMPERED_METADATA,
    INVALID_SIGNATURE,
    REPLAY_DETECTED,
)

# Which component-digest mismatch maps to which finding code.
_COMPONENT_TO_CODE = {
    "input": TAMPERED_INPUT,
    "model": TAMPERED_MODEL,
    "preprocessing_config": TAMPERED_CONFIG,
    "inference_config": TAMPERED_CONFIG,
    "output": TAMPERED_OUTPUT,
    "metadata": TAMPERED_METADATA,
}


@dataclass
class VerificationFinding:
    """One detected integrity problem."""

    code: str
    component: str  # payload section / "record" / "file" / "signature" / "replay"
    evidence: str
    explanation: str

    def __post_init__(self):
        if self.code not in FINDING_CODES:
            raise ValueError(f"Unknown finding code '{self.code}'. Known: {FINDING_CODES}")

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "component": self.component,
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


@dataclass
class VerificationReport:
    """The full, structured result of verifying one provenance record."""

    record_path: Optional[str]
    record_id: Optional[str]
    generated_at: str
    hash_integrity: str  # "valid" | "invalid" | "not_evaluated"
    signature_status: str  # "valid" | "invalid" | "unsigned" | "not_checked"
    replay_status: str  # "new" | "replayed" | "not_checked"
    findings: List[VerificationFinding] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        if not self.findings:
            return VALID
        # Deterministic, de-duplicated ordering of all distinct codes.
        seen: List[str] = []
        for f in self.findings:
            if f.code not in seen:
                seen.append(f.code)
        return "+".join(seen)

    @property
    def is_valid(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "run_metadata": {
                "record_path": self.record_path,
                "record_id": self.record_id,
                "generated_at": self.generated_at,
            },
            "summary": {
                "overall_status": self.overall_status,
                "is_valid": self.is_valid,
                "hash_integrity": self.hash_integrity,
                "signature_status": self.signature_status,
                "replay_status": self.replay_status,
                "num_findings": len(self.findings),
                "finding_codes": sorted({f.code for f in self.findings}),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def print_summary(self) -> None:
        print("\nInference Provenance Verification\n")
        print(f"Record:          {self.record_path or '(in-memory)'}")
        print(f"Record id:       {self.record_id}")
        print(f"Hash integrity:  {self.hash_integrity}")
        print(f"Signature:       {self.signature_status}")
        print(f"Replay check:    {self.replay_status}")
        print(f"Overall status:  {self.overall_status}")
        if self.findings:
            print(f"\n{len(self.findings)} finding(s):")
            for f in self.findings:
                print(f"  - [{f.code}] ({f.component}) {f.evidence}")
        print()


class ProvenanceVerifier:
    """Verifies protected inference records (see module docstring)."""

    def __init__(
        self,
        key: Optional[bytes] = None,
        replay_registry: Optional[ReplayRegistry] = None,
    ):
        self.key = key
        self.replay_registry = replay_registry

    # ------------------------------------------------------------- helpers
    def _structural_findings(self, record: Any) -> List[VerificationFinding]:
        """Checks that must pass before digests can even be recomputed."""
        problems: List[str] = []
        if not isinstance(record, dict):
            problems.append("record is not a JSON object")
        else:
            if record.get("record_format_version") != RECORD_FORMAT_VERSION:
                problems.append(
                    f"unsupported record_format_version "
                    f"{record.get('record_format_version')!r} (expected '{RECORD_FORMAT_VERSION}')"
                )
            payload = record.get("payload")
            protection = record.get("protection")
            if not isinstance(payload, dict):
                problems.append("missing or invalid 'payload' object")
            else:
                for section in COMPONENT_SECTIONS:
                    if not isinstance(payload.get(section), dict):
                        problems.append(f"payload is missing section '{section}'")
            if not isinstance(protection, dict):
                problems.append("missing or invalid 'protection' object")
            else:
                digests = protection.get("component_digests")
                if not isinstance(digests, dict):
                    problems.append("protection is missing 'component_digests'")
                else:
                    for name in ALL_COMPONENTS:
                        if not is_sha256_hex(digests.get(name)):
                            problems.append(
                                f"component digest '{name}' is missing or not a SHA-256 hex digest"
                            )
                if not is_sha256_hex(protection.get("record_digest")):
                    problems.append("protection is missing a valid 'record_digest'")
        return [
            VerificationFinding(
                code=CORRUPTED_RECORD,
                component="record",
                evidence=problem,
                explanation=(
                    "The record cannot be (fully) verified because its structure is "
                    "damaged or incomplete. Treat its contents as untrusted."
                ),
            )
            for problem in problems
        ]

    # ---------------------------------------------------------------- main
    def verify(
        self,
        record: Any,
        record_path: Optional[str] = None,
        image_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> VerificationReport:
        """Verify one record; optionally re-hash the actual image/model files.

        Parameters
        ----------
        record:
            The parsed record (as loaded by ``record.load_record``).
        image_path / model_path:
            If given, the actual files are re-hashed and compared against the
            SHA-256 values *recorded in the payload* -- this is how tampering
            with the files themselves (rather than with the record) is caught.
        """
        findings: List[VerificationFinding] = []
        report = VerificationReport(
            record_path=record_path,
            record_id=None,
            generated_at=utc_timestamp(),
            hash_integrity="not_evaluated",
            signature_status="not_checked",
            replay_status="not_checked",
            findings=findings,
        )

        # 1. Structure -----------------------------------------------------
        structural = self._structural_findings(record)
        if structural:
            findings.extend(structural)
            report.hash_integrity = "invalid"
            if isinstance(record, dict) and isinstance(record.get("payload"), dict):
                report.record_id = record["payload"].get("record_id")
            return report

        payload: Dict[str, Any] = record["payload"]
        protection: Dict[str, Any] = record["protection"]
        report.record_id = payload.get("record_id")

        # 2. Component digests: attribute changes to specific components ---
        stored_digests: Dict[str, str] = protection["component_digests"]
        try:
            recomputed_digests = compute_component_digests(payload)
        except (TypeError, ValueError) as exc:
            findings.append(
                VerificationFinding(
                    code=CORRUPTED_RECORD,
                    component="record",
                    evidence=f"payload cannot be canonicalized: {exc}",
                    explanation=(
                        "The payload contains values outside the canonical JSON "
                        "model, so its digests cannot be recomputed."
                    ),
                )
            )
            report.hash_integrity = "invalid"
            return report

        component_mismatch = False
        for name in ALL_COMPONENTS:
            if recomputed_digests[name] != stored_digests[name]:
                component_mismatch = True
                if name == "metadata":
                    changed = "metadata fields (record id / timestamps / sequence / nonce / software)"
                    content: Any = metadata_component(payload)
                else:
                    changed = f"'{name}' section"
                    content = payload[name]
                findings.append(
                    VerificationFinding(
                        code=_COMPONENT_TO_CODE[name],
                        component=name,
                        evidence=(
                            f"recomputed digest {recomputed_digests[name][:16]}... does not "
                            f"match protected digest {str(stored_digests[name])[:16]}... "
                            f"for the {changed}; current content: "
                            f"{json.dumps(content, sort_keys=True)[:200]}"
                        ),
                        explanation=(
                            f"The {changed} was changed after the record was protected."
                        ),
                    )
                )

        # 3. Overall record digest ----------------------------------------
        recomputed_record_digest = compute_record_digest(payload, stored_digests)
        record_digest_ok = recomputed_record_digest == protection["record_digest"]
        if not record_digest_ok and not component_mismatch:
            # Every component matches its digest, yet the overall digest does
            # not: the protection block itself (digest table or record digest)
            # was altered, or an unknown payload field was added/changed.
            findings.append(
                VerificationFinding(
                    code=TAMPERED_METADATA,
                    component="record",
                    evidence=(
                        f"recomputed record digest {recomputed_record_digest[:16]}... does "
                        f"not match stored record digest "
                        f"{str(protection['record_digest'])[:16]}..., while all component "
                        f"digests match their sections"
                    ),
                    explanation=(
                        "The record's protection metadata (digest table / record digest) "
                        "or a non-component payload field was altered."
                    ),
                )
            )
        report.hash_integrity = (
            "valid" if (record_digest_ok and not component_mismatch) else "invalid"
        )

        # 4. Actual files on disk (optional) ------------------------------
        if image_path is not None:
            findings.extend(
                self._file_findings(
                    image_path,
                    payload["input"].get("image_sha256"),
                    TAMPERED_INPUT,
                    "input image",
                )
            )
        if model_path is not None:
            findings.extend(
                self._file_findings(
                    model_path,
                    payload["model"].get("model_sha256"),
                    TAMPERED_MODEL,
                    "model weights file",
                )
            )

        # 5. Signature (authenticity layer, separate from hash integrity) --
        signature = protection.get("signature")
        if self.key is None:
            report.signature_status = "unsigned" if not signature else "not_checked"
        else:
            if not isinstance(signature, dict) or not signature.get("value"):
                report.signature_status = "invalid"
                findings.append(
                    VerificationFinding(
                        code=INVALID_SIGNATURE,
                        component="signature",
                        evidence="a verification key was provided but the record carries no signature",
                        explanation=(
                            "Authenticity was expected (a key was supplied) but the record "
                            "is unsigned; an unsigned record could have been produced by "
                            "anyone, including an attacker replacing a signed record."
                        ),
                    )
                )
            elif signature.get("scheme") != SIGNATURE_SCHEME:
                report.signature_status = "invalid"
                findings.append(
                    VerificationFinding(
                        code=INVALID_SIGNATURE,
                        component="signature",
                        evidence=f"unsupported signature scheme {signature.get('scheme')!r}",
                        explanation=f"Only '{SIGNATURE_SCHEME}' signatures are supported.",
                    )
                )
            else:
                # Verify against the RECOMPUTED digest: even a fully
                # re-hashed (internally consistent) forgery fails here
                # unless the forger holds the key.
                if verify_digest_signature(
                    recomputed_record_digest, str(signature.get("value")), self.key
                ):
                    report.signature_status = "valid"
                else:
                    report.signature_status = "invalid"
                    findings.append(
                        VerificationFinding(
                            code=INVALID_SIGNATURE,
                            component="signature",
                            evidence=(
                                "HMAC-SHA256 signature does not verify against the "
                                "recomputed record digest under the provided key"
                            ),
                            explanation=(
                                "Either the record content changed after signing, or the "
                                "record was not produced by a holder of this key."
                            ),
                        )
                    )

        # 6. Replay protection (optional) ----------------------------------
        if self.replay_registry is not None:
            check = self.replay_registry.check_and_register(payload)
            if check.is_replay:
                report.replay_status = "replayed"
                findings.append(
                    VerificationFinding(
                        code=REPLAY_DETECTED,
                        component="replay",
                        evidence=check.reason
                        + (
                            f"; first accepted at {check.previous_entry.get('first_seen')}"
                            if check.previous_entry
                            else ""
                        ),
                        explanation=(
                            "This protected record was already accepted before. Its "
                            "content is unchanged, but re-presenting it as a new "
                            "inference is a replay."
                        ),
                    )
                )
            else:
                report.replay_status = "new"
                self.replay_registry.save()

        return report

    @staticmethod
    def _file_findings(
        file_path: str, recorded_sha256: Optional[str], code: str, what: str
    ) -> List[VerificationFinding]:
        p = Path(file_path)
        if not p.exists():
            return [
                VerificationFinding(
                    code=code,
                    component="file",
                    evidence=f"{what} '{file_path}' does not exist",
                    explanation=f"The {what} referenced for verification is missing.",
                )
            ]
        actual = sha256_of_file(str(p))
        if actual != recorded_sha256:
            return [
                VerificationFinding(
                    code=code,
                    component="file",
                    evidence=(
                        f"{what} '{file_path}' hashes to {actual[:16]}... but the record "
                        f"protects {str(recorded_sha256)[:16]}..."
                    ),
                    explanation=(
                        f"The {what} on disk is not the one this inference record was "
                        f"created from (substituted or modified)."
                    ),
                )
            ]
        return []


def build_chain_report(result: ChainVerificationResult, log_path: str) -> dict:
    """Wrap an audit-chain verification result in the project report shape."""
    return {
        "run_metadata": {
            "log_path": log_path,
            "generated_at": utc_timestamp(),
        },
        "summary": {
            "overall_status": VALID if result.valid else "TAMPERED_AUDIT_LOG",
            "is_valid": result.valid,
            "num_entries": result.num_entries,
            "num_issues": len(result.issues),
            "head_hash": result.head_hash,
        },
        "issues": [i.to_dict() for i in result.issues],
    }
