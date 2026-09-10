"""
test_provenance_verification.py
--------------------------------
Verifier behavior for every required tamper class (Phase 4 tests #1-#8):
valid record, modified input image, modified model, modified preprocessing
config, modified inference config, modified prediction, modified metadata,
and corrupted records. Also: multiple simultaneous problems are ALL
reported.

All records here are synthetic (see ``provenance_helpers``); tampering is
performed on the loaded/parsed record exactly as an attacker editing the
JSON would.
"""

import copy

from cv_auditor.provenance import (
    CORRUPTED_RECORD,
    TAMPERED_CONFIG,
    TAMPERED_INPUT,
    TAMPERED_METADATA,
    TAMPERED_MODEL,
    TAMPERED_OUTPUT,
    VALID,
    ProvenanceVerifier,
)

from .provenance_helpers import finding_codes, make_record, write_fixture_files


def verify(record, **kwargs):
    return ProvenanceVerifier().verify(record, **kwargs)


class TestValidRecord:
    def test_untouched_record_is_valid(self, tmp_path):
        report = verify(make_record(tmp_path))
        assert report.is_valid
        assert report.overall_status == VALID
        assert report.hash_integrity == "valid"
        assert report.findings == []

    def test_untouched_record_with_matching_files_is_valid(self, tmp_path):
        image_path, model_path = write_fixture_files(tmp_path)
        record = make_record(tmp_path, files=(image_path, model_path))
        report = verify(
            record, image_path=str(image_path), model_path=str(model_path)
        )
        assert report.is_valid
        assert report.overall_status == VALID


class TestTamperedInput:
    def test_modified_image_hash_field(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["input"]["image_sha256"] = "1" * 64
        report = verify(record)
        assert TAMPERED_INPUT in finding_codes(report)
        assert report.hash_integrity == "invalid"

    def test_modified_image_file_on_disk(self, tmp_path):
        image_path, model_path = write_fixture_files(tmp_path)
        record = make_record(tmp_path, files=(image_path, model_path))
        image_path.write_bytes(b"completely different image bytes")
        report = verify(record, image_path=str(image_path))
        assert finding_codes(report) == {TAMPERED_INPUT}
        # The record itself is untouched; only the file changed.
        assert report.hash_integrity == "valid"

    def test_missing_image_file(self, tmp_path):
        image_path, model_path = write_fixture_files(tmp_path)
        record = make_record(tmp_path, files=(image_path, model_path))
        image_path.unlink()
        report = verify(record, image_path=str(image_path))
        assert TAMPERED_INPUT in finding_codes(report)


class TestTamperedModel:
    def test_modified_model_hash_field(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["model"]["model_sha256"] = "2" * 64
        report = verify(record)
        assert TAMPERED_MODEL in finding_codes(report)

    def test_modified_model_identity(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["model"]["model_name"] = "totally_other_model"
        report = verify(record)
        assert TAMPERED_MODEL in finding_codes(report)

    def test_substituted_model_file_on_disk(self, tmp_path):
        image_path, model_path = write_fixture_files(tmp_path)
        record = make_record(tmp_path, files=(image_path, model_path))
        model_path.write_bytes(b"malicious replacement weights")
        report = verify(record, model_path=str(model_path))
        assert finding_codes(report) == {TAMPERED_MODEL}


class TestTamperedConfig:
    def test_modified_preprocessing_config(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["preprocessing_config"]["color_space"] = "BGR"
        report = verify(record)
        assert TAMPERED_CONFIG in finding_codes(report)

    def test_modified_inference_config(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["inference_config"]["confidence_threshold"] = 0.01
        report = verify(record)
        assert TAMPERED_CONFIG in finding_codes(report)


class TestTamperedOutput:
    def test_modified_prediction_class(self, tmp_path):
        """The spec's canonical example: airplane -> person."""
        record = make_record(tmp_path, prediction_class="airplane")
        record["payload"]["output"]["predictions"][0]["class"] = "person"
        report = verify(record)
        assert TAMPERED_OUTPUT in finding_codes(report)

    def test_modified_confidence(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["output"]["predictions"][0]["confidence"] = 0.01
        report = verify(record)
        assert TAMPERED_OUTPUT in finding_codes(report)

    def test_deleted_prediction(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["output"]["predictions"] = []
        report = verify(record)
        assert TAMPERED_OUTPUT in finding_codes(report)


class TestTamperedMetadata:
    def test_modified_timestamp(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["created_at"] = "1999-01-01T00:00:00+00:00"
        report = verify(record)
        assert TAMPERED_METADATA in finding_codes(report)

    def test_modified_sequence_number(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["sequence_number"] = 999
        report = verify(record)
        assert TAMPERED_METADATA in finding_codes(report)

    def test_modified_nonce(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["nonce"] = "bb" * 16
        report = verify(record)
        assert TAMPERED_METADATA in finding_codes(report)

    def test_modified_record_id(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["record_id"] = "record-9999"
        report = verify(record)
        assert TAMPERED_METADATA in finding_codes(report)

    def test_tampered_protection_block(self, tmp_path):
        """All components intact, but the stored record digest was edited."""
        record = make_record(tmp_path)
        record["protection"]["record_digest"] = "3" * 64
        report = verify(record)
        assert TAMPERED_METADATA in finding_codes(report)

    def test_consistently_rehashed_forgery_passes_hash_check(self, tmp_path):
        """Documented limitation: an attacker who recomputes ALL digests
        produces a hash-consistent record. Hashing alone cannot catch this;
        that is exactly what the signature layer is for (see
        test_provenance_signing.py)."""
        from cv_auditor.provenance import protect_payload

        record = make_record(tmp_path, prediction_class="airplane")
        tampered_payload = copy.deepcopy(record["payload"])
        tampered_payload["output"]["predictions"][0]["class"] = "person"
        reforged = protect_payload(tampered_payload)
        report = verify(reforged)
        assert report.hash_integrity == "valid"  # hashing is integrity, not authenticity


class TestCorruptedRecord:
    def test_missing_payload_section(self, tmp_path):
        record = make_record(tmp_path)
        del record["payload"]["output"]
        report = verify(record)
        assert CORRUPTED_RECORD in finding_codes(report)
        assert not report.is_valid

    def test_missing_protection_block(self, tmp_path):
        record = make_record(tmp_path)
        del record["protection"]
        report = verify(record)
        assert CORRUPTED_RECORD in finding_codes(report)

    def test_garbage_digest_values(self, tmp_path):
        record = make_record(tmp_path)
        record["protection"]["component_digests"]["output"] = "not-a-digest"
        report = verify(record)
        assert CORRUPTED_RECORD in finding_codes(report)

    def test_wrong_format_version(self, tmp_path):
        record = make_record(tmp_path)
        record["record_format_version"] = "99.0"
        report = verify(record)
        assert CORRUPTED_RECORD in finding_codes(report)

    def test_not_a_dict(self):
        report = verify(["definitely", "not", "a", "record"])
        assert CORRUPTED_RECORD in finding_codes(report)

    def test_uncanonicalizable_payload_value(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["output"]["predictions"][0]["confidence"] = float("nan")
        report = verify(record)
        assert CORRUPTED_RECORD in finding_codes(report)


class TestMultipleFindings:
    def test_all_problems_reported_not_just_first(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["output"]["predictions"][0]["class"] = "person"
        record["payload"]["model"]["model_sha256"] = "4" * 64
        record["payload"]["created_at"] = "1999-01-01T00:00:00+00:00"
        report = verify(record)
        codes = finding_codes(report)
        assert {TAMPERED_OUTPUT, TAMPERED_MODEL, TAMPERED_METADATA} <= codes
        # The combined status names every distinct problem.
        for code in (TAMPERED_OUTPUT, TAMPERED_MODEL, TAMPERED_METADATA):
            assert code in report.overall_status

    def test_report_serializes(self, tmp_path):
        record = make_record(tmp_path)
        record["payload"]["output"]["predictions"][0]["class"] = "person"
        report = verify(record)
        data = report.to_dict()
        assert data["summary"]["is_valid"] is False
        assert data["summary"]["num_findings"] == len(report.findings)
        assert data["findings"][0]["code"] == TAMPERED_OUTPUT
        path = tmp_path / "report.json"
        report.save(str(path))
        assert path.exists()
