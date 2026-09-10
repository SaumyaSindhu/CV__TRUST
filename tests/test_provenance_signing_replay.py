"""
test_provenance_signing_replay.py
----------------------------------
Signature verification (Phase 4 test #14) and duplicate/replayed record
detection (test #9).
"""

import copy

import pytest

from cv_auditor.provenance import (
    INVALID_SIGNATURE,
    REPLAY_DETECTED,
    ProvenanceVerifier,
    ReplayRegistry,
    generate_key,
    load_key,
    protect_payload,
    save_key,
    sign_record,
)

from .provenance_helpers import finding_codes, make_record


class TestKeyHandling:
    def test_keygen_save_load_roundtrip(self, tmp_path):
        key = generate_key()
        assert len(key) == 32
        path = save_key(key, str(tmp_path / "signing.key"))
        assert load_key(path) == key

    def test_load_key_rejects_garbage(self, tmp_path):
        bad = tmp_path / "bad.key"
        bad.write_text("this is not hex")
        with pytest.raises(ValueError):
            load_key(str(bad))
        with pytest.raises(FileNotFoundError):
            load_key(str(tmp_path / "missing.key"))


class TestSignatures:
    def test_signed_record_verifies_with_correct_key(self, tmp_path):
        key = generate_key()
        record = sign_record(make_record(tmp_path), key)
        report = ProvenanceVerifier(key=key).verify(record)
        assert report.is_valid
        assert report.signature_status == "valid"

    def test_wrong_key_fails(self, tmp_path):
        record = sign_record(make_record(tmp_path), generate_key())
        report = ProvenanceVerifier(key=generate_key()).verify(record)
        assert report.signature_status == "invalid"
        assert INVALID_SIGNATURE in finding_codes(report)

    def test_content_tampering_after_signing_invalidates_signature(self, tmp_path):
        key = generate_key()
        record = sign_record(make_record(tmp_path), key)
        record["payload"]["output"]["predictions"][0]["class"] = "person"
        report = ProvenanceVerifier(key=key).verify(record)
        assert INVALID_SIGNATURE in finding_codes(report)  # plus TAMPERED_OUTPUT
        assert "TAMPERED_OUTPUT" in report.overall_status

    def test_consistent_reforge_without_key_is_caught_by_signature(self, tmp_path):
        """The attack hashing alone cannot catch: attacker rewrites content
        AND recomputes every digest -- but cannot re-sign without the key."""
        key = generate_key()
        record = sign_record(make_record(tmp_path, prediction_class="airplane"), key)

        tampered_payload = copy.deepcopy(record["payload"])
        tampered_payload["output"]["predictions"][0]["class"] = "person"
        reforged = protect_payload(tampered_payload)  # internally consistent...
        reforged["protection"]["signature"] = record["protection"]["signature"]  # ...old sig

        report = ProvenanceVerifier(key=key).verify(reforged)
        assert report.hash_integrity == "valid"  # hashes all check out!
        assert report.signature_status == "invalid"  # but authenticity fails
        assert finding_codes(report) == {INVALID_SIGNATURE}

    def test_unsigned_record_with_key_expected_is_flagged(self, tmp_path):
        report = ProvenanceVerifier(key=generate_key()).verify(make_record(tmp_path))
        assert report.signature_status == "invalid"
        assert INVALID_SIGNATURE in finding_codes(report)

    def test_unsigned_record_without_key_is_fine(self, tmp_path):
        report = ProvenanceVerifier().verify(make_record(tmp_path))
        assert report.is_valid
        assert report.signature_status == "unsigned"


class TestReplayDetection:
    def test_first_submission_accepted_then_replay_detected(self, tmp_path):
        """Phase 4 test #9: the same protected record submitted twice."""
        registry_path = tmp_path / "registry.json"
        record = make_record(tmp_path)

        first = ProvenanceVerifier(
            replay_registry=ReplayRegistry(str(registry_path))
        ).verify(record)
        assert first.is_valid
        assert first.replay_status == "new"

        # Fresh verifier + reloaded registry, as in a real second submission.
        second = ProvenanceVerifier(
            replay_registry=ReplayRegistry(str(registry_path))
        ).verify(record)
        assert not second.is_valid
        assert second.replay_status == "replayed"
        assert finding_codes(second) == {REPLAY_DETECTED}

    def test_nonce_reuse_under_new_record_id_detected(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        registry = ReplayRegistry(str(registry_path))
        first = make_record(tmp_path, record_id="record-0001", nonce="cc" * 16)
        forged = make_record(tmp_path, record_id="record-0002", nonce="cc" * 16)

        assert ProvenanceVerifier(replay_registry=registry).verify(first).is_valid
        report = ProvenanceVerifier(
            replay_registry=ReplayRegistry(str(registry_path))
        ).verify(forged)
        assert REPLAY_DETECTED in finding_codes(report)

    def test_distinct_records_are_not_replays(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        r1 = make_record(tmp_path, record_id="record-0001", nonce="aa" * 16)
        r2 = make_record(tmp_path, record_id="record-0002", nonce="dd" * 16,
                         sequence_number=1)
        assert ProvenanceVerifier(
            replay_registry=ReplayRegistry(str(registry_path))
        ).verify(r1).is_valid
        assert ProvenanceVerifier(
            replay_registry=ReplayRegistry(str(registry_path))
        ).verify(r2).is_valid

    def test_replay_check_disabled_by_default(self, tmp_path):
        record = make_record(tmp_path)
        verifier = ProvenanceVerifier()
        assert verifier.verify(record).replay_status == "not_checked"
        assert verifier.verify(record).is_valid  # twice, still fine

    def test_replayed_record_does_not_overwrite_first_evidence(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        record = make_record(tmp_path)
        reg = ReplayRegistry(str(registry_path))
        assert not reg.check_and_register(record["payload"]).is_replay
        reg.save()

        reg2 = ReplayRegistry(str(registry_path))
        result = reg2.check_and_register(record["payload"])
        assert result.is_replay
        assert result.previous_entry is not None
        assert result.previous_entry["nonce"] == record["payload"]["nonce"]
        assert len(reg2) == 1
