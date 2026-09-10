"""
canonical.py (provenance)
--------------------------
Deterministic ("canonical") JSON serialization + SHA-256 helpers.

Why this exists
---------------
A cryptographic digest is only meaningful if the *bytes being hashed* are
reproducible. Python dicts preserve insertion order, JSON allows arbitrary
whitespace, and ``json.dumps`` defaults differ between call sites -- so the
same logical record could serialize to different bytes and therefore
different digests. This module pins down ONE serialization:

    - keys sorted lexicographically at every nesting level
    - no whitespace (separators ``(",", ":")``)
    - ``ensure_ascii=False`` (UTF-8 bytes, no ``\\uXXXX`` escaping ambiguity)
    - ``allow_nan=False`` (NaN/Infinity are rejected -- they are not valid
      JSON and their textual form is implementation-defined)
    - only JSON-native types are accepted (dict/list/str/int/float/bool/None);
      anything else raises instead of silently str()-ing

so that the same logical record always produces the same digest, on any
machine, in any process. This is the property every Phase 4 integrity check
is built on.

The canonicalization scheme is versioned (``CANONICALIZATION_ID``) and
stored inside every protected record, so a future format change cannot be
confused with tampering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Identifier stored in every protected record so a verifier knows exactly
# which serialization rules produced the digests it is checking.
CANONICALIZATION_ID = "json/sorted-keys/compact/utf-8/v1"

HASH_ALGORITHM = "sha256"


def _check_json_safe(obj: Any, path: str = "$") -> None:
    """Recursively reject anything that is not a JSON-native value.

    ``json.dumps`` would happily serialize some non-JSON types via ``default=``
    hooks or coerce dict keys with ``str()``; both are sources of silent
    non-determinism, so we forbid them outright.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        # NaN/inf are rejected later by allow_nan=False, but catching them
        # here gives a clearer error message including the JSON path.
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite float at {path} cannot be canonicalized")
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_json_safe(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Non-string dict key {key!r} at {path} cannot be canonicalized"
                )
            _check_json_safe(value, f"{path}.{key}")
        return
    raise TypeError(
        f"Value of type {type(obj).__name__} at {path} is not JSON-serializable "
        f"in canonical form (allowed: dict, list, str, int, float, bool, None)"
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical JSON bytes (see module docstring)."""
    _check_json_safe(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_of_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_obj(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON serialization of ``obj``.

    This is THE digest primitive for Phase 4: every component digest, record
    digest, and audit-log entry hash is computed through this function.
    """
    return sha256_of_bytes(canonical_json_bytes(obj))


def is_sha256_hex(value: Any) -> bool:
    """True if ``value`` looks like a SHA-256 hex digest (64 hex chars)."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
