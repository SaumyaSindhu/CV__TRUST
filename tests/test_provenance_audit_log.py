"""
test_provenance_audit_log.py
-----------------------------
The tamper-evident audit chain (Phase 4 tests #10-#12): a valid chain
verifies; modifying, deleting, or reordering entries is detected.
"""

import json

from cv_auditor.provenance import AuditLog, GENESIS_HASH

from .provenance_helpers import make_record


def build_log(tmp_path, n=4):
    log = AuditLog(str(tmp_path / "audit_log.jsonl"))
    records = [
        make_record(
            tmp_path,
            record_id=f"record-{i:04d}",
            nonce=f"{i:02x}" * 16,
            sequence_number=i,
        )
        for i in range(n)
    ]
    for record in records:
        log.append_record(record)
    return log, records


def rewrite_lines(log, lines):
    with open(log.path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def read_lines(log):
    with open(log.path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


class TestValidChain:
    def test_empty_log_is_valid(self, tmp_path):
        result = AuditLog(str(tmp_path / "audit_log.jsonl")).verify_chain()
        assert result.valid
        assert result.num_entries == 0

    def test_valid_chain_verifies(self, tmp_path):
        """Phase 4 test #10."""
        log, records = build_log(tmp_path)
        result = log.verify_chain()
        assert result.valid
        assert result.num_entries == len(records)
        assert result.issues == []
        assert result.head_hash == log.head_hash()

    def test_entries_link_and_index_correctly(self, tmp_path):
        log, _ = build_log(tmp_path, n=3)
        entries = log.read_entries()
        assert entries[0]["previous_entry_hash"] == GENESIS_HASH
        for i, entry in enumerate(entries):
            assert entry["entry_index"] == i
            if i:
                assert entry["previous_entry_hash"] == entries[i - 1]["entry_hash"]

    def test_next_sequence_number_tracks_length(self, tmp_path):
        log, records = build_log(tmp_path, n=2)
        assert log.next_sequence_number() == 2


class TestModifiedEntry:
    def test_modified_entry_detected(self, tmp_path):
        """Phase 4 test #11: silently editing a logged digest breaks BOTH
        that entry's own hash and (if also recomputed) the chain."""
        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        entry = json.loads(lines[1])
        entry["record_digest"] = "0" * 64
        lines[1] = json.dumps(entry, sort_keys=True)
        rewrite_lines(log, lines)

        result = log.verify_chain()
        assert not result.valid
        assert any(i.code == "ENTRY_HASH_MISMATCH" and i.line_number == 2
                   for i in result.issues)

    def test_modified_entry_with_recomputed_hash_breaks_chain(self, tmp_path):
        """A cleverer attacker also fixes the entry's own hash -- the NEXT
        entry's previous_entry_hash then no longer matches."""
        from cv_auditor.provenance.audit_log import compute_entry_hash

        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        entry = json.loads(lines[1])
        entry["record_digest"] = "0" * 64
        entry["entry_hash"] = compute_entry_hash(entry)
        lines[1] = json.dumps(entry, sort_keys=True)
        rewrite_lines(log, lines)

        result = log.verify_chain()
        assert not result.valid
        assert any(i.code == "CHAIN_BROKEN" and i.line_number == 3
                   for i in result.issues)


class TestDeletedAndReordered:
    def test_deleted_middle_entry_detected(self, tmp_path):
        """Phase 4 test #12a."""
        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        del lines[1]
        rewrite_lines(log, lines)

        result = log.verify_chain()
        assert not result.valid
        codes = {i.code for i in result.issues}
        assert "CHAIN_BROKEN" in codes
        assert "INDEX_MISMATCH" in codes

    def test_deleted_first_entry_detected(self, tmp_path):
        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        del lines[0]
        rewrite_lines(log, lines)
        result = log.verify_chain()
        assert not result.valid

    def test_reordered_entries_detected(self, tmp_path):
        """Phase 4 test #12b."""
        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        lines[1], lines[2] = lines[2], lines[1]
        rewrite_lines(log, lines)

        result = log.verify_chain()
        assert not result.valid
        codes = {i.code for i in result.issues}
        assert "CHAIN_BROKEN" in codes
        assert "INDEX_MISMATCH" in codes

    def test_corrupted_line_detected(self, tmp_path):
        log, _ = build_log(tmp_path)
        lines = read_lines(log)
        lines[2] = "{ this is not json"
        rewrite_lines(log, lines)
        result = log.verify_chain()
        assert not result.valid
        assert any(i.code == "CORRUPTED_ENTRY" and i.line_number == 3
                   for i in result.issues)

    def test_all_issues_reported_not_just_first(self, tmp_path):
        log, _ = build_log(tmp_path, n=5)
        lines = read_lines(log)
        e1 = json.loads(lines[1]); e1["record_id"] = "evil"; lines[1] = json.dumps(e1, sort_keys=True)
        del lines[3]
        rewrite_lines(log, lines)
        result = log.verify_chain()
        assert len(result.issues) >= 2
