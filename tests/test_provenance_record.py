"""
test_provenance_record.py
--------------------------
The provenance record itself: building, protecting, saving/loading, and
the determinism of its digests.
"""

import pytest

from cv_auditor.provenance import (
    RECORD_FORMAT_VERSION,
    load_record,
    protect_payload,
    save_record,
)
from cv_auditor.provenance.record import (
    ALL_COMPONENTS,
    compute_component_digests,
    compute_record_digest,
    new_nonce,
    new_record_id,
)

from .provenance_helpers import make_payload, make_record


class TestRecordConstruction:
    def test_payload_contains_all_required_binding_fields(self, tmp_path):
        payload = make_payload(tmp_path)
        # The SIH26228 binding: input image, model digest, preprocessing +
        # inference config, output, plus anti-replay metadata.
        assert payload["input"]["image_sha256"]
        assert payload["model"]["model_sha256"]
        assert payload["model"]["model_name"]
        assert payload["preprocessing_config"]
        assert payload["inference_config"]
        assert payload["output"]["predictions"]
        assert payload["record_id"]
        assert payload["created_at"]
        assert payload["nonce"]
        assert payload["sequence_number"] == 0
        assert "software" in payload

    def test_protect_payload_shape(self, tmp_path):
        record = make_record(tmp_path)
        assert record["record_format_version"] == RECORD_FORMAT_VERSION
        protection = record["protection"]
        assert protection["hash_algorithm"] == "sha256"
        assert set(protection["component_digests"]) == set(ALL_COMPONENTS)
        assert len(protection["record_digest"]) == 64
        assert protection["signature"] is None

    def test_protection_is_deterministic(self, tmp_path):
        """Same payload -> identical digests, run after run (test #13)."""
        payload = make_payload(tmp_path)
        r1 = protect_payload(payload)
        r2 = protect_payload(payload)
        assert r1["protection"]["component_digests"] == r2["protection"]["component_digests"]
        assert r1["protection"]["record_digest"] == r2["protection"]["record_digest"]

    def test_different_payloads_produce_different_digests(self, tmp_path):
        r1 = make_record(tmp_path, prediction_class="airplane")
        r2 = make_record(tmp_path, prediction_class="person")
        assert (
            r1["protection"]["record_digest"] != r2["protection"]["record_digest"]
        )
        assert (
            r1["protection"]["component_digests"]["output"]
            != r2["protection"]["component_digests"]["output"]
        )
        # Untouched components digest identically.
        assert (
            r1["protection"]["component_digests"]["input"]
            == r2["protection"]["component_digests"]["input"]
        )

    def test_record_digest_covers_component_digest_table(self, tmp_path):
        payload = make_payload(tmp_path)
        digests = compute_component_digests(payload)
        original = compute_record_digest(payload, digests)
        fiddled = dict(digests)
        fiddled["output"] = "0" * 64
        assert compute_record_digest(payload, fiddled) != original


class TestRecordIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        record = make_record(tmp_path)
        path = tmp_path / "record.json"
        save_record(record, str(path))
        assert load_record(str(path)) == record

    def test_load_missing_record_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_record(str(tmp_path / "nope.json"))


class TestIdentifiers:
    def test_record_ids_unique(self):
        assert new_record_id() != new_record_id()

    def test_nonces_unique_and_hex(self):
        a, b = new_nonce(), new_nonce()
        assert a != b
        int(a, 16)  # must be valid hex
        assert len(a) == 32
