"""
audit_log.py (provenance)
--------------------------
Append-only, hash-chained audit log of protected inference records
(JSON Lines: one entry per line).

Each entry commits to (a) the record it logs and (b) the entire log before
it, via a hash chain:

    entry_hash[i] = SHA256(canonical({
        entry_index, logged_at, record_id, record_digest,
        previous_entry_hash = entry_hash[i-1]      # 64 zeros for entry 0
    }))

Because every entry's hash covers the previous entry's hash, *modifying,
deleting, or reordering* any earlier entry breaks every subsequent link --
which is exactly what ``verify_chain`` detects. This is a plain local hash
chain, NOT a blockchain: there is no consensus, no distribution, no proof
of work.

Honest limitation (also in README): truncating the log by removing entries
*from the tail* leaves a shorter but internally consistent chain. Detecting
tail truncation requires remembering the last entry hash somewhere outside
the file (a "chain head anchor"); ``AuditLog.head_hash()`` exposes it so an
operator can note it down, but Phase 4 does not manage external anchors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import ensure_parent_dir, get_logger, utc_timestamp
from .canonical import sha256_of_obj

logger = get_logger(__name__)

GENESIS_HASH = "0" * 64

ENTRY_FIELDS = (
    "entry_index",
    "logged_at",
    "record_id",
    "record_digest",
    "previous_entry_hash",
)


def compute_entry_hash(entry: Dict[str, Any]) -> str:
    """Hash of an entry's chained fields (everything except ``entry_hash``)."""
    return sha256_of_obj({name: entry.get(name) for name in ENTRY_FIELDS})


@dataclass
class ChainIssue:
    """One problem found while verifying the audit chain."""

    line_number: int  # 1-based line in the log file
    code: str  # e.g. "ENTRY_HASH_MISMATCH", "CHAIN_BROKEN", ...
    detail: str

    def to_dict(self) -> dict:
        return {"line_number": self.line_number, "code": self.code, "detail": self.detail}


@dataclass
class ChainVerificationResult:
    """Outcome of verifying a complete audit log."""

    valid: bool
    num_entries: int
    issues: List[ChainIssue] = field(default_factory=list)
    head_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "num_entries": self.num_entries,
            "head_hash": self.head_hash,
            "issues": [i.to_dict() for i in self.issues],
        }


class AuditLog:
    """Append-only JSONL audit log with hash chaining."""

    def __init__(self, path: str):
        self.path = str(path)

    # ------------------------------------------------------------- reading
    def read_entries(self) -> List[Dict[str, Any]]:
        """Read raw entries. Unparseable lines surface as ``{"_parse_error"}``
        placeholders so verification can point at the exact line."""
        p = Path(self.path)
        if not p.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError("entry is not a JSON object")
                    entries.append(parsed)
                except (json.JSONDecodeError, ValueError) as exc:
                    entries.append({"_parse_error": str(exc)})
        return entries

    def __len__(self) -> int:
        return len(self.read_entries())

    def head_hash(self) -> str:
        """Hash of the newest entry (the chain head), or the genesis hash."""
        entries = self.read_entries()
        for entry in reversed(entries):
            if "_parse_error" not in entry:
                return str(entry.get("entry_hash", GENESIS_HASH))
        return GENESIS_HASH

    def next_sequence_number(self) -> int:
        """Convenience: sequence number for the next record (= current length)."""
        return len(self.read_entries())

    # ------------------------------------------------------------ appending
    def append_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append a protected record's identifying digests to the log."""
        payload = record.get("payload", {})
        protection = record.get("protection", {})
        entry = {
            "entry_index": len(self.read_entries()),
            "logged_at": utc_timestamp(),
            "record_id": payload.get("record_id"),
            "record_digest": protection.get("record_digest"),
            "previous_entry_hash": self.head_hash(),
        }
        entry["entry_hash"] = compute_entry_hash(entry)

        ensure_parent_dir(self.path)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        logger.info(
            "Appended audit entry %d (record %s) to %s",
            entry["entry_index"], entry["record_id"], self.path,
        )
        return entry

    # ----------------------------------------------------------- verifying
    def verify_chain(self) -> ChainVerificationResult:
        """Verify the whole chain; reports ALL issues found, not just the first."""
        entries = self.read_entries()
        issues: List[ChainIssue] = []
        expected_previous = GENESIS_HASH
        head: Optional[str] = None

        for pos, entry in enumerate(entries):
            line_no = pos + 1
            if "_parse_error" in entry:
                issues.append(ChainIssue(line_no, "CORRUPTED_ENTRY", entry["_parse_error"]))
                # The chain cannot be followed through an unreadable entry.
                expected_previous = None  # type: ignore[assignment]
                continue

            missing = [f for f in ENTRY_FIELDS + ("entry_hash",) if f not in entry]
            if missing:
                issues.append(
                    ChainIssue(line_no, "CORRUPTED_ENTRY", f"missing fields: {missing}")
                )
                expected_previous = None  # type: ignore[assignment]
                continue

            recomputed = compute_entry_hash(entry)
            if recomputed != entry["entry_hash"]:
                issues.append(
                    ChainIssue(
                        line_no,
                        "ENTRY_HASH_MISMATCH",
                        "stored entry_hash does not match the entry's content "
                        "(entry was modified after being written)",
                    )
                )

            if entry["entry_index"] != pos:
                issues.append(
                    ChainIssue(
                        line_no,
                        "INDEX_MISMATCH",
                        f"entry_index is {entry['entry_index']} but the entry sits at "
                        f"position {pos} (entries deleted, inserted, or reordered)",
                    )
                )

            if expected_previous is not None and entry["previous_entry_hash"] != expected_previous:
                issues.append(
                    ChainIssue(
                        line_no,
                        "CHAIN_BROKEN",
                        "previous_entry_hash does not match the preceding entry's hash "
                        "(an earlier entry was modified, deleted, or reordered)",
                    )
                )

            # Chain onwards from the *stored* hash: if this entry is intact,
            # later links are judged against what is actually in the file.
            expected_previous = entry["entry_hash"]
            head = entry["entry_hash"]

        return ChainVerificationResult(
            valid=not issues,
            num_entries=len(entries),
            issues=issues,
            head_hash=head,
        )
