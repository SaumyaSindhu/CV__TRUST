"""
test_provenance_e2e.py
-----------------------
The realistic Phase 4 end-to-end pipeline (required test, spec section 9):

    image -> model -> inference -> provenance record -> cryptographic
    protection -> audit log -> verification -> VALID

then tamper with the prediction (airplane -> person), the model file, the
image file, and the configs, and check that the corresponding integrity
failures (TAMPERED_OUTPUT / TAMPERED_MODEL / TAMPERED_INPUT /
TAMPERED_CONFIG) are reported.

Uses the bundled offline sample model + sample images -- no internet.
"""

import copy
import json
import shutil

import pytest
import torch

from cv_auditor import ImageLoader, InferenceEngine, ModelLoader, PipelineConfig
from cv_auditor.provenance import (
    AuditLog,
    ProvenanceVerifier,
    ReplayRegistry,
    TAMPERED_CONFIG,
    TAMPERED_INPUT,
    TAMPERED_MODEL,
    TAMPERED_OUTPUT,
    VALID,
    generate_key,
    load_record,
    payload_from_inference,
    protect_payload,
    save_record,
    sign_record,
)

from .fixtures import SAMPLE_ARCHITECTURE, SAMPLE_IMAGES_DIR, SAMPLE_MODEL_PATH
from .provenance_helpers import finding_codes

SAMPLE_IMAGE = SAMPLE_IMAGES_DIR / "image_001.jpg"


@pytest.fixture(scope="module")
def live_inference():
    """Run the real Phase 1 pipeline once for the whole module.

    detector_score_thresh=0.0 makes the untrained bundled model emit
    genuinely-computed (if meaningless) detections -- same technique
    Phase 3 uses (see PipelineConfig.detector_score_thresh).
    """
    config = PipelineConfig(
        model_path=str(SAMPLE_MODEL_PATH),
        architecture=SAMPLE_ARCHITECTURE,
        model_name="e2e_provenance_model",
        confidence_threshold=0.0,
        detector_score_thresh=0.0,
        input_dir=str(SAMPLE_IMAGE),
    )
    model, model_metadata = ModelLoader(config).load()
    image = ImageLoader(config.input_dir).load_one(str(SAMPLE_IMAGE))
    engine = InferenceEngine(
        model=model,
        model_name=config.model_name,
        class_names=config.class_names,
        device=torch.device("cpu"),
        confidence_threshold=config.confidence_threshold,
    )
    image_result = engine.run_one(image)
    return config, model_metadata, image_result


@pytest.fixture()
def protected_setup(live_inference, tmp_path):
    """Fresh signed record + audit log + verifier key per test."""
    config, model_metadata, image_result = live_inference
    key = generate_key()
    log = AuditLog(str(tmp_path / "audit_log.jsonl"))

    payload = payload_from_inference(
        image_result, model_metadata, config,
        sequence_number=log.next_sequence_number(),
    )
    record = sign_record(protect_payload(payload), key)
    record_path = tmp_path / "inference_record.json"
    save_record(record, str(record_path))
    log.append_record(record)
    return {
        "config": config,
        "record_path": record_path,
        "key": key,
        "log": log,
    }


class TestEndToEndValid:
    def test_full_pipeline_reports_valid(self, protected_setup, tmp_path):
        """image -> ... -> verification -> VALID (record, files, signature,
        replay, and audit chain all clean)."""
        record = load_record(str(protected_setup["record_path"]))

        report = ProvenanceVerifier(
            key=protected_setup["key"],
            replay_registry=ReplayRegistry(str(tmp_path / "registry.json")),
        ).verify(
            record,
            record_path=str(protected_setup["record_path"]),
            image_path=str(SAMPLE_IMAGE),
            model_path=str(SAMPLE_MODEL_PATH),
        )
        assert report.overall_status == VALID
        assert report.hash_integrity == "valid"
        assert report.signature_status == "valid"
        assert report.replay_status == "new"

        chain = protected_setup["log"].verify_chain()
        assert chain.valid and chain.num_entries == 1

    def test_record_binds_real_pipeline_facts(self, protected_setup):
        from cv_auditor.utils import sha256_of_file

        record = load_record(str(protected_setup["record_path"]))
        payload = record["payload"]
        assert payload["input"]["image_sha256"] == sha256_of_file(str(SAMPLE_IMAGE))
        assert payload["model"]["model_sha256"] == sha256_of_file(str(SAMPLE_MODEL_PATH))
        assert payload["model"]["architecture"] == SAMPLE_ARCHITECTURE
        assert payload["inference_config"]["confidence_threshold"] == 0.0
        assert isinstance(payload["output"]["predictions"], list)


class TestEndToEndTampering:
    def test_modified_prediction_reports_tampered_output(self, protected_setup):
        """Spec section 9: original prediction class -> "person"."""
        record = load_record(str(protected_setup["record_path"]))
        predictions = record["payload"]["output"]["predictions"]
        if predictions:
            # e.g. "airplane" (or whatever the model produced) -> "person"
            predictions[0]["class"] = "person"
            predictions[0]["class_id"] = 1
        else:  # pragma: no cover - untrained model edge case
            predictions.append(
                {"class": "person", "class_id": 1, "confidence": 0.99,
                 "bbox": [0.0, 0.0, 10.0, 10.0]}
            )

        report = ProvenanceVerifier(key=protected_setup["key"]).verify(record)
        assert TAMPERED_OUTPUT in finding_codes(report)
        assert "TAMPERED_OUTPUT" in report.overall_status

    def test_modified_model_file_reports_tampered_model(self, protected_setup, tmp_path):
        tampered_model = tmp_path / "tampered_model.pth"
        shutil.copy(SAMPLE_MODEL_PATH, tampered_model)
        with open(tampered_model, "ab") as f:
            f.write(b"\x00backdoor-bytes")

        record = load_record(str(protected_setup["record_path"]))
        report = ProvenanceVerifier().verify(record, model_path=str(tampered_model))
        assert finding_codes(report) == {TAMPERED_MODEL}

    def test_modified_image_file_reports_tampered_input(self, protected_setup, tmp_path):
        tampered_image = tmp_path / "tampered_image.jpg"
        shutil.copy(SAMPLE_IMAGE, tampered_image)
        with open(tampered_image, "ab") as f:
            f.write(b"\x00sneaky-pixels")

        record = load_record(str(protected_setup["record_path"]))
        report = ProvenanceVerifier().verify(record, image_path=str(tampered_image))
        assert finding_codes(report) == {TAMPERED_INPUT}

    def test_modified_config_reports_tampered_config(self, protected_setup):
        record = load_record(str(protected_setup["record_path"]))
        record["payload"]["inference_config"]["confidence_threshold"] = 0.99
        report = ProvenanceVerifier().verify(record)
        assert TAMPERED_CONFIG in finding_codes(report)

        record2 = load_record(str(protected_setup["record_path"]))
        record2["payload"]["preprocessing_config"]["color_space"] = "BGR"
        report2 = ProvenanceVerifier().verify(record2)
        assert TAMPERED_CONFIG in finding_codes(report2)

    def test_replayed_record_detected_on_second_submission(self, protected_setup, tmp_path):
        record = load_record(str(protected_setup["record_path"]))
        registry_path = str(tmp_path / "registry.json")
        assert ProvenanceVerifier(
            replay_registry=ReplayRegistry(registry_path)
        ).verify(record).is_valid
        second = ProvenanceVerifier(
            replay_registry=ReplayRegistry(registry_path)
        ).verify(record)
        assert not second.is_valid
        assert second.replay_status == "replayed"

    def test_tampered_audit_log_detected(self, protected_setup):
        log = protected_setup["log"]
        with open(log.path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        entry = json.loads(lines[0])
        entry["record_digest"] = "0" * 64
        lines[0] = json.dumps(entry, sort_keys=True)
        with open(log.path, "w") as f:
            f.write("\n".join(lines) + "\n")

        result = log.verify_chain()
        assert not result.valid

    def test_two_runs_produce_distinct_chained_entries(self, live_inference, tmp_path):
        """A second inference record extends the same chain verifiably."""
        config, model_metadata, image_result = live_inference
        log = AuditLog(str(tmp_path / "audit_log.jsonl"))

        for _ in range(2):
            payload = payload_from_inference(
                image_result, model_metadata, config,
                sequence_number=log.next_sequence_number(),
            )
            log.append_record(protect_payload(payload))

        entries = log.read_entries()
        assert [e["entry_index"] for e in entries] == [0, 1]
        assert entries[0]["record_id"] != entries[1]["record_id"]
        assert entries[1]["previous_entry_hash"] == entries[0]["entry_hash"]
        assert log.verify_chain().valid
