"""
signing.py (provenance)
------------------------
Optional authenticity layer for provenance records: HMAC-SHA256 with a
locally stored secret key.

Hash vs. signature -- read this before trusting anything
--------------------------------------------------------
* The digests in ``record.py`` provide **integrity detection**: they reveal
  when a record's content no longer matches its stored digests. But anyone
  who modifies a record can also recompute every digest -- hashing proves
  nothing about *who* produced the record.
* The HMAC in this module provides **authenticity relative to a secret
  key**: a record whose HMAC verifies was produced (or endorsed) by
  someone holding the key. An attacker without the key cannot re-forge a
  consistent signed record.
* HMAC is *symmetric*: every party that can VERIFY can also SIGN. This is
  appropriate for a single-operator, fully offline audit trail (the Phase 4
  scope). If third parties must verify without being able to forge, an
  asymmetric scheme (e.g. Ed25519) is required -- deliberately out of scope
  here to avoid new dependencies (see README, Phase 4 limitations).

Only Python standard-library primitives are used (``hmac``/``hashlib``/
``secrets``), keeping Phase 4 dependency-free and fully offline.
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from typing import Any, Dict

from .canonical import canonical_json_bytes
from .record import RECORD_FORMAT_VERSION

SIGNATURE_SCHEME = "hmac-sha256"
KEY_NUM_BYTES = 32


def generate_key() -> bytes:
    """Generate a fresh 256-bit random HMAC key."""
    return secrets.token_bytes(KEY_NUM_BYTES)


def save_key(key: bytes, path: str) -> str:
    """Write a key to disk (hex, owner-read/write only)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(key.hex() + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover - e.g. exotic filesystems
        pass
    return str(p)


def load_key(path: str) -> bytes:
    """Load a hex-encoded key file written by ``save_key``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Signing key file not found: {path}")
    text = p.read_text().strip()
    try:
        key = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"Key file '{path}' does not contain valid hex") from exc
    if len(key) < 16:
        raise ValueError(f"Key in '{path}' is too short ({len(key)} bytes; need >= 16)")
    return key


def _signing_message(record_digest: str) -> bytes:
    """The exact bytes that get MACed.

    The format version and scheme are bound into the message so a signature
    can never be transplanted across record formats or schemes.
    """
    return canonical_json_bytes(
        {
            "record_format_version": RECORD_FORMAT_VERSION,
            "scheme": SIGNATURE_SCHEME,
            "record_digest": record_digest,
        }
    )


def sign_digest(record_digest: str, key: bytes) -> str:
    """HMAC-SHA256 over the signing message for ``record_digest`` (hex)."""
    return hmac.new(key, _signing_message(record_digest), "sha256").hexdigest()


def verify_digest_signature(record_digest: str, signature_hex: str, key: bytes) -> bool:
    """Constant-time verification of a signature produced by ``sign_digest``."""
    expected = sign_digest(record_digest, key)
    try:
        return hmac.compare_digest(expected, signature_hex)
    except TypeError:
        return False


def sign_record(record: Dict[str, Any], key: bytes, key_id: str = "default") -> Dict[str, Any]:
    """Attach a signature to a protected record (in place; also returned).

    Important: the signature is over the *recomputable* record digest, not
    the stored copy -- verification always recomputes the digest from the
    record's content first, so content tampering invalidates the signature
    even if the attacker also updates every stored digest.
    """
    protection = record.get("protection")
    if not isinstance(protection, dict) or "record_digest" not in protection:
        raise ValueError("Record is not protected yet; call protect_payload() first")
    protection["signature"] = {
        "scheme": SIGNATURE_SCHEME,
        "key_id": key_id,
        "value": sign_digest(protection["record_digest"], key),
    }
    return record
