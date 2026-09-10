"""
replay.py (provenance)
-----------------------
Simple, local, deterministic replay protection for protected inference
records.

A "replay" here means: the *same* protected record (same record id / nonce)
is submitted for acceptance again, e.g. to make one genuine inference look
like many, or to re-present an old accepted result as new. Replay is NOT a
content-integrity problem -- a replayed record hashes and verifies
perfectly -- which is exactly why it needs its own mechanism.

Mechanism
---------
A ``ReplayRegistry`` is a small JSON file remembering, for every record it
has accepted:

    record_id  ->  {nonce, sequence_number, created_at, first_seen}
    nonce      ->  record_id            (reverse index)

``check_and_register`` reports a replay when

* the record id has been seen before (identical resubmission), or
* the nonce has been seen before under a DIFFERENT record id (a forged
  "new" record recycling an old nonce).

Scope/limitations (documented, deliberate): the registry is a local file
-- it protects one verifier's acceptance decisions, not a distributed
system, and clearing the file resets its memory. Sequence numbers and
timestamps are stored as evidence for auditors; strict monotonicity is not
enforced because multiple independent producers may share one verifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import ensure_parent_dir, get_logger, utc_timestamp

logger = get_logger(__name__)

REGISTRY_FORMAT_VERSION = "1.0"


@dataclass
class ReplayCheckResult:
    """Outcome of a single replay check."""

    is_replay: bool
    reason: str = ""
    previous_entry: Optional[Dict[str, Any]] = None


class ReplayRegistry:
    """File-backed memory of already-accepted provenance records."""

    def __init__(self, path: str):
        self.path = str(path)
        self._data = self._load_or_init()

    # ------------------------------------------------------------------ io
    def _load_or_init(self) -> Dict[str, Any]:
        p = Path(self.path)
        if not p.exists():
            return {
                "registry_format_version": REGISTRY_FORMAT_VERSION,
                "records": {},
                "nonces": {},
            }
        with open(p, "r") as f:
            data = json.load(f)
        for key in ("records", "nonces"):
            if key not in data or not isinstance(data[key], dict):
                raise ValueError(f"Replay registry '{self.path}' is malformed (missing '{key}')")
        return data

    def save(self) -> str:
        ensure_parent_dir(self.path)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        return self.path

    # --------------------------------------------------------------- checks
    def check_and_register(self, payload: Dict[str, Any]) -> ReplayCheckResult:
        """Check a record payload against the registry; register if new.

        Only registers records that are NOT replays, so a replayed record
        does not overwrite the evidence about its first acceptance.
        The caller is responsible for calling ``save()`` afterwards.
        """
        record_id = str(payload.get("record_id", ""))
        nonce = str(payload.get("nonce", ""))

        records: Dict[str, Any] = self._data["records"]
        nonces: Dict[str, str] = self._data["nonces"]

        if record_id in records:
            return ReplayCheckResult(
                is_replay=True,
                reason=f"record_id '{record_id}' was already accepted",
                previous_entry=records[record_id],
            )
        if nonce and nonce in nonces:
            prev_id = nonces[nonce]
            return ReplayCheckResult(
                is_replay=True,
                reason=(
                    f"nonce '{nonce}' was already used by record_id '{prev_id}'"
                ),
                previous_entry=records.get(prev_id),
            )

        entry = {
            "nonce": nonce,
            "sequence_number": payload.get("sequence_number"),
            "created_at": payload.get("created_at"),
            "first_seen": utc_timestamp(),
        }
        records[record_id] = entry
        if nonce:
            nonces[nonce] = record_id
        return ReplayCheckResult(is_replay=False)

    def __len__(self) -> int:
        return len(self._data["records"])
