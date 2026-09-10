"""
test_provenance_cli.py
-----------------------
Exercises the ``audit_inference.py`` CLI end-to-end through real
subprocess invocations: keygen -> create -> verify (VALID and tampered)
-> verify-log (valid and tampered chain). Exit codes are part of the
contract (0 = VALID, 1 = anything else).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "audit_inference.py"

from .fixtures import SAMPLE_IMAGES_DIR, SAMPLE_MODEL_PATH  # noqa: E402

SAMPLE_IMAGE = SAMPLE_IMAGES_DIR / "image_001.jpg"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory):
    """One keygen + create run shared by the whole module (subprocesses
    are slow because each re-imports torch)."""
    ws = tmp_path_factory.mktemp("cli_ws")
    key = ws / "signing.key"
    record = ws / "inference_record.json"
    log = ws / "audit_log.jsonl"
    prediction = ws / "prediction.json"

    keygen = run_cli("keygen", "--key", str(key))
    assert keygen.returncode == 0, keygen.stderr

    create = run_cli(
        "create",
        "--image", str(SAMPLE_IMAGE),
        "--model", str(SAMPLE_MODEL_PATH),
        "--output", str(prediction),
        "--record", str(record),
        "--log", str(log),
        "--key", str(key),
        "--detector-score-thresh", "0.0",
    )
    assert create.returncode == 0, create.stderr
    return {"key": key, "record": record, "log": log, "prediction": prediction}


class TestCreate:
    def test_create_writes_all_artifacts(self, cli_workspace):
        assert cli_workspace["record"].exists()
        assert cli_workspace["log"].exists()
        assert cli_workspace["prediction"].exists()

        record = json.loads(cli_workspace["record"].read_text())
        assert record["protection"]["record_digest"]
        assert record["protection"]["signature"]["scheme"] == "hmac-sha256"

        prediction = json.loads(cli_workspace["prediction"].read_text())
        assert prediction["results"]  # Phase 1 schema preserved


class TestVerify:
    def test_verify_valid_record_exit_0(self, cli_workspace, tmp_path):
        report_path = tmp_path / "report.json"
        result = run_cli(
            "verify",
            "--record", str(cli_workspace["record"]),
            "--image", str(SAMPLE_IMAGE),
            "--model", str(SAMPLE_MODEL_PATH),
            "--key", str(cli_workspace["key"]),
            "--output", str(report_path),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VALID" in result.stdout
        report = json.loads(report_path.read_text())
        assert report["summary"]["overall_status"] == "VALID"
        assert report["summary"]["signature_status"] == "valid"

    def test_verify_tampered_record_exit_1(self, cli_workspace, tmp_path):
        record = json.loads(cli_workspace["record"].read_text())
        preds = record["payload"]["output"]["predictions"]
        if preds:
            preds[0]["class"] = "person"
        else:
            preds.append({"class": "person", "class_id": 1,
                          "confidence": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]})
        tampered_path = tmp_path / "tampered_record.json"
        tampered_path.write_text(json.dumps(record, indent=2))

        result = run_cli("verify", "--record", str(tampered_path),
                         "--key", str(cli_workspace["key"]))
        assert result.returncode == 1
        assert "TAMPERED_OUTPUT" in result.stdout
        assert "INVALID_SIGNATURE" in result.stdout  # content changed after signing

    def test_verify_replay_detection(self, cli_workspace, tmp_path):
        registry = tmp_path / "registry.json"
        first = run_cli("verify", "--record", str(cli_workspace["record"]),
                        "--check-replay", "--replay-registry", str(registry))
        assert first.returncode == 0, first.stdout + first.stderr
        second = run_cli("verify", "--record", str(cli_workspace["record"]),
                         "--check-replay", "--replay-registry", str(registry))
        assert second.returncode == 1
        assert "REPLAY_DETECTED" in second.stdout

    def test_verify_corrupted_record_exit_1(self, cli_workspace, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text(cli_workspace["record"].read_text()[:100])
        result = run_cli("verify", "--record", str(broken))
        assert result.returncode == 1
        assert "CORRUPTED_RECORD" in result.stdout


class TestVerifyLog:
    def test_verify_log_valid_exit_0(self, cli_workspace):
        result = run_cli("verify-log", "--log", str(cli_workspace["log"]))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VALID" in result.stdout

    def test_verify_log_tampered_exit_1(self, cli_workspace, tmp_path):
        lines = [ln for ln in cli_workspace["log"].read_text().splitlines() if ln.strip()]
        entry = json.loads(lines[0])
        entry["record_digest"] = "0" * 64
        lines[0] = json.dumps(entry, sort_keys=True)
        tampered_log = tmp_path / "tampered_log.jsonl"
        tampered_log.write_text("\n".join(lines) + "\n")

        result = run_cli("verify-log", "--log", str(tampered_log))
        assert result.returncode == 1
        assert "ENTRY_HASH_MISMATCH" in result.stdout
