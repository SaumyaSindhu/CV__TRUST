import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv_auditor.model_audit.fingerprint import compare_fingerprint, compute_fingerprint


def test_compute_fingerprint_is_deterministic(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"some model bytes")
    assert compute_fingerprint(str(f)) == compute_fingerprint(str(f))


def test_compute_fingerprint_differs_for_different_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"model bytes A")
    b.write_bytes(b"model bytes B")
    assert compute_fingerprint(str(a)) != compute_fingerprint(str(b))


def test_compare_fingerprint_no_reference_available(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"bytes")
    result = compare_fingerprint(str(f), reference_sha256=None)
    assert result.reference_sha256 is None
    assert result.changed is None  # nothing to compare against -- not a guess


def test_compare_fingerprint_matches_reference(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"identical bytes")
    reference_hash = compute_fingerprint(str(f))

    result = compare_fingerprint(str(f), reference_sha256=reference_hash)
    assert result.changed is False


def test_compare_fingerprint_detects_change(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"original bytes")
    reference_hash = compute_fingerprint(str(f))

    f.write_bytes(b"modified bytes")  # simulate the file changing
    result = compare_fingerprint(str(f), reference_sha256=reference_hash)

    assert result.changed is True
    assert result.candidate_sha256 != result.reference_sha256


def test_fingerprint_result_serializes_cleanly(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"bytes")
    result = compare_fingerprint(str(f), reference_sha256="deadbeef")
    d = result.to_dict()
    assert set(d.keys()) == {"candidate_path", "candidate_sha256", "reference_sha256", "changed"}
