"""
fingerprint.py
---------------
Model fingerprinting: SHA-256 of the model file, compared against a
previously trusted/reference hash.

A hash mismatch means exactly one thing: **the file's bytes are
different from the reference**. It says nothing about *why* -- a
legitimate re-export, a metadata change, a retrain, a version bump, or a
malicious tamper would all change the hash identically. That's why a
mismatch here is reported as "model file changed", never as "malicious"
-- see ``suspicion.py`` for how this signal is combined with actual
behavioral evidence before anything stronger is said.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..utils import sha256_of_file


@dataclass
class FingerprintResult:
    candidate_path: str
    candidate_sha256: str
    reference_sha256: Optional[str]  # None if no reference hash was available to compare against
    changed: Optional[bool]  # None if there was nothing to compare against

    def to_dict(self) -> dict:
        return {
            "candidate_path": self.candidate_path,
            "candidate_sha256": self.candidate_sha256,
            "reference_sha256": self.reference_sha256,
            "changed": self.changed,
        }


def compute_fingerprint(model_path: str) -> str:
    """SHA-256 hex digest of a model file's raw bytes."""
    return sha256_of_file(model_path)


def compare_fingerprint(candidate_path: str, reference_sha256: Optional[str]) -> FingerprintResult:
    """Compute the candidate's fingerprint and compare it to a trusted reference hash.

    Parameters
    ----------
    candidate_path:
        Path to the model file being audited.
    reference_sha256:
        A previously recorded, trusted SHA-256 hash to compare against.
        Pass None if no reference hash is available (the result's
        ``changed`` field will then be None rather than a guess).
    """
    candidate_hash = compute_fingerprint(candidate_path)
    changed = None if reference_sha256 is None else (candidate_hash != reference_sha256)
    return FingerprintResult(
        candidate_path=candidate_path,
        candidate_sha256=candidate_hash,
        reference_sha256=reference_sha256,
        changed=changed,
    )
