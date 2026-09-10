"""
cv_auditor.provenance - Phase 4: Inference Provenance & Output Integrity
=========================================================================
Creates a verifiable cryptographic binding among the input image, the
model identifier/weight digest, the preprocessing and inference
configuration, and the resulting output -- and makes post-hoc alteration,
substitution, or replay of protected inference records detectable.

Building blocks
---------------
- ``canonical``     deterministic JSON serialization + SHA-256 digests
- ``record``        provenance record schema, builders, hash protection
- ``signing``       optional HMAC-SHA256 authenticity layer (stdlib only)
- ``verification``  record verifier -> structured findings (TAMPERED_*, ...)
- ``replay``        local replay-protection registry
- ``audit_log``     append-only, hash-chained audit log + chain verifier

See ``audit_inference.py`` (project root) for the Phase 4 CLI and the
README's Phase 4 sections for concepts, examples, and limitations.
"""

from .audit_log import AuditLog, ChainVerificationResult, GENESIS_HASH
from .canonical import (
    CANONICALIZATION_ID,
    canonical_json_bytes,
    sha256_of_bytes,
    sha256_of_obj,
)
from .record import (
    RECORD_FORMAT_VERSION,
    build_payload,
    load_record,
    payload_from_inference,
    protect_payload,
    save_record,
)
from .replay import ReplayCheckResult, ReplayRegistry
from .signing import (
    SIGNATURE_SCHEME,
    generate_key,
    load_key,
    save_key,
    sign_record,
)
from .verification import (
    CORRUPTED_RECORD,
    FINDING_CODES,
    INVALID_SIGNATURE,
    REPLAY_DETECTED,
    TAMPERED_CONFIG,
    TAMPERED_INPUT,
    TAMPERED_METADATA,
    TAMPERED_MODEL,
    TAMPERED_OUTPUT,
    VALID,
    ProvenanceVerifier,
    VerificationFinding,
    VerificationReport,
    build_chain_report,
)

__all__ = [
    # canonical
    "CANONICALIZATION_ID",
    "canonical_json_bytes",
    "sha256_of_bytes",
    "sha256_of_obj",
    # record
    "RECORD_FORMAT_VERSION",
    "build_payload",
    "payload_from_inference",
    "protect_payload",
    "save_record",
    "load_record",
    # signing
    "SIGNATURE_SCHEME",
    "generate_key",
    "save_key",
    "load_key",
    "sign_record",
    # verification
    "VALID",
    "CORRUPTED_RECORD",
    "TAMPERED_INPUT",
    "TAMPERED_MODEL",
    "TAMPERED_CONFIG",
    "TAMPERED_OUTPUT",
    "TAMPERED_METADATA",
    "INVALID_SIGNATURE",
    "REPLAY_DETECTED",
    "FINDING_CODES",
    "ProvenanceVerifier",
    "VerificationFinding",
    "VerificationReport",
    "build_chain_report",
    # replay
    "ReplayRegistry",
    "ReplayCheckResult",
    # audit log
    "AuditLog",
    "ChainVerificationResult",
    "GENESIS_HASH",
]
