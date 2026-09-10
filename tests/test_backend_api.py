"""
test_backend_api.py
--------------------
Backend tests: every endpoint exercised with the small deterministic
fixtures, against a temporary report store (the app's store paths are
pointed at tmp dirs so tests never touch results/).

The routes must stay thin wrappers -- these tests verify they return the
SAME report structures the underlying engines produce, plus store /
findings / audit-log behavior and the served frontend.
"""

import importlib
import json
from pathlib import Path

import pytest

from .distribution_helpers import BASELINE_STYLE, WINTER_STYLE, make_batch
from .provenance_helpers import make_record

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = ROOT / "data" / "sample_dataset"
SAMPLE_MODEL = ROOT / "data" / "sample_model" / "sample_model.pth"

pytestmark = pytest.mark.skipif(
    not SAMPLE_DATASET.is_dir() or not SAMPLE_MODEL.exists(),
    reason="sample assets missing",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    tmp = tmp_path_factory.mktemp("api_store")

    import backend.app as app_module
    importlib.reload(app_module)
    app_module.STORE_DIR = tmp
    app_module.AUDIT_LOG_PATH = tmp / "assurance_audit_log.jsonl"
    return fastapi_testclient.TestClient(app_module.app)


@pytest.fixture(scope="module")
def phase5_assets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api_fixtures")
    make_batch(tmp / "reference", n=12, seed=3, style=BASELINE_STYLE)
    make_batch(tmp / "incoming", n=8, seed=4, style=WINTER_STYLE)
    from cv_auditor.distribution import DistributionConfig, build_baseline
    baseline = build_baseline(str(tmp / "reference"), DistributionConfig())
    baseline_path = tmp / "baseline.json"
    baseline.save(str(baseline_path))

    record = make_record(tmp)
    record_path = tmp / "record.json"
    record_path.write_text(json.dumps(record))
    return {"incoming": tmp / "incoming", "baseline": baseline_path,
            "record": record_path}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "governance" in body["phases"]


def test_dataset_audit_endpoint(client):
    r = client.post("/api/datasets/audit",
                    json={"dataset_dir": str(SAMPLE_DATASET)})
    assert r.status_code == 200
    env = r.json()
    assert env["kind"] == "dataset_audit"
    summary = env["report"]["summary"]
    assert "overall_dataset_finding_level" in summary
    assert summary["total_samples"] > 0


def test_model_audit_endpoint(client):
    r = client.post("/api/models/audit", json={
        "candidate_model_path": str(SAMPLE_MODEL),
        "reference_model_path": str(SAMPLE_MODEL),
        "detector_score_thresh": 0.0,
    })
    assert r.status_code == 200
    env = r.json()
    assert env["report"]["summary"]["category"] == "clean"
    assert env["report"]["comparison"]["agreement_rate"] == 1.0


def test_inference_verify_endpoint(client, phase5_assets):
    r = client.post("/api/inference/verify",
                    json={"record_path": str(phase5_assets["record"])})
    assert r.status_code == 200
    assert r.json()["report"]["summary"]["overall_status"] == "VALID"


def test_distribution_audit_endpoint(client, phase5_assets):
    r = client.post("/api/distribution/audit", json={
        "incoming_dir": str(phase5_assets["incoming"]),
        "baseline_path": str(phase5_assets["baseline"]),
        "metadata": {"season": "winter"},
    })
    assert r.status_code == 200
    rep = r.json()["report"]
    assert rep["risk"] in ("MEDIUM", "HIGH")
    assert rep["current_batch"]["metadata"] == {"season": "winter"}


def test_assurance_run_findings_reports_and_audit_log(client, phase5_assets):
    r = client.post("/api/assurance/run", json={
        "incoming_dir": str(phase5_assets["incoming"]),
        "distribution_baseline_path": str(phase5_assets["baseline"]),
        "inference_record_path": str(phase5_assets["record"]),
    })
    assert r.status_code == 200
    env = r.json()
    assessment_id = env["report_id"]
    assert env["report"]["governance"]["recommendation"] in (
        "ACCEPT", "REVIEW", "QUARANTINE")

    # stored & retrievable
    listed = client.get("/api/reports").json()["reports"]
    assert any(x["report_id"] == assessment_id for x in listed)
    fetched = client.get(f"/api/reports/{assessment_id}").json()
    assert fetched["report"]["assessment_id"] == assessment_id

    # aggregated findings include this assessment's findings
    body = client.get("/api/findings").json()
    assert body["count"] >= 1
    assert any(f["report_id"] == assessment_id for f in body["findings"])

    # audit log recorded the assessment; chain verifies
    log = client.get("/api/audit-log").json()
    assert log["chain_valid"] is True
    assert any(e["record_id"] == assessment_id for e in log["entries"])


def test_analyst_decision_endpoint(client, phase5_assets):
    run = client.post("/api/assurance/run", json={
        "incoming_dir": str(phase5_assets["incoming"]),
        "distribution_baseline_path": str(phase5_assets["baseline"]),
    }).json()
    rid = run["report_id"]
    r = client.post(f"/api/reports/{rid}/decision", json={
        "decision": "REVIEW", "analyst": "api.tester", "note": "via api",
    })
    assert r.status_code == 200
    stored = client.get(f"/api/reports/{rid}").json()
    ad = stored["report"]["governance"]["analyst_decision"]
    assert ad["decision"] == "REVIEW" and ad["analyst"] == "api.tester"
    log = client.get("/api/audit-log").json()
    assert any(e["record_id"] == f"{rid}-decision" for e in log["entries"])
    assert log["chain_valid"] is True


def test_error_paths(client):
    assert client.get("/api/reports/nope").status_code == 404
    r = client.post("/api/datasets/audit",
                    json={"dataset_dir": "/definitely/not/here"})
    assert r.status_code == 400


def test_frontend_served_with_all_seven_panels(client):
    """The production React build is served at '/' by FastAPI; the seven
    required panels and the real API endpoints it renders from must be
    present in the served bundle (built JSX text survives minification)."""
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "/static/assets/" in html  # built bundle, not a placeholder page

    static_dir = Path(__file__).resolve().parent.parent / "backend" / "static"
    bundles = list((static_dir / "assets").glob("index-*.js"))
    assert bundles, "React production build missing — run `npm run build` in frontend/"
    js = "".join(p.read_text(encoding="utf-8") for p in bundles)

    for marker in ("overall risk",            # overall assurance visualization
                   "Dataset Integrity", "Model Integrity",
                   "Provenance hash chain", "Distribution Shift",
                   "Analyst decision", "Audit Trail"):
        assert marker in js, marker
    # the dashboard renders backend data only -- these are its real sources
    for endpoint in ("/api/health", "/api/reports", "/api/findings",
                     "/api/audit-log"):
        assert endpoint in js, endpoint

    # bundle assets must be served by the same backend (air-gapped: no CDN)
    asset_path = "/static/assets/" + bundles[0].name
    assert client.get(asset_path).status_code == 200
    assert "http://" not in html and "https://" not in html  # no external resources


def test_fs_browse_rooted_listing(client):
    """The picker's read-only browse endpoint: lists within the root,
    navigates children, refuses to escape the root."""
    root = client.get("/api/fs/browse").json()
    assert root["path"] == root["root"]
    names = {e["name"] for e in root["entries"]}
    assert "data" in names and "backend" in names
    data_dir = next(e for e in root["entries"] if e["name"] == "data")
    sub = client.get("/api/fs/browse", params={"path": data_dir["path"]}).json()
    assert sub["parent"] == root["path"]
    assert any(e["name"] == "sample_images" and e["is_dir"] for e in sub["entries"])
    assert client.get("/api/fs/browse", params={"path": "/etc"}).status_code == 403


def test_assurance_dataset_only(client):
    """UI scenario 1: only a Dataset selected -> only the dataset layer runs."""
    r = client.post("/api/assurance/run", json={"dataset_dir": str(SAMPLE_DATASET)})
    assert r.status_code == 200
    sections = r.json()["report"]["sections"]
    assert sections["dataset"] is not None
    assert sections["model"] is None
    assert sections["provenance"] is None
    assert sections["distribution"] is None


def test_assurance_model_only(client):
    """UI scenario 2: Candidate + Reference Model -> only Phase 3 runs."""
    r = client.post("/api/assurance/run", json={
        "candidate_model_path": str(SAMPLE_MODEL),
        "reference_model_path": str(SAMPLE_MODEL),
        "detector_score_thresh": 0.0,
    })
    assert r.status_code == 200
    rep = r.json()["report"]
    assert rep["sections"]["model"]["status"] == "clean"
    assert rep["sections"]["dataset"] is None
    assert rep["sections"]["provenance"] is None
    assert rep["sections"]["distribution"] is None


def test_assurance_inference_only(client, phase5_assets):
    """UI scenario 4: only an Inference Record -> only Phase 4 runs."""
    r = client.post("/api/assurance/run",
                    json={"inference_record_path": str(phase5_assets["record"])})
    assert r.status_code == 200
    sections = r.json()["report"]["sections"]
    assert sections["provenance"]["status"] == "VALID"
    assert sections["dataset"] is None and sections["model"] is None
    assert sections["distribution"] is None


def test_assurance_full_all_inputs(client, phase5_assets):
    """UI scenario 5: every input -> all four evidence layers execute."""
    r = client.post("/api/assurance/run", json={
        "dataset_dir": str(SAMPLE_DATASET),
        "candidate_model_path": str(SAMPLE_MODEL),
        "reference_model_path": str(SAMPLE_MODEL),
        "detector_score_thresh": 0.0,
        "inference_record_path": str(phase5_assets["record"]),
        "incoming_dir": str(phase5_assets["incoming"]),
        "distribution_baseline_path": str(phase5_assets["baseline"]),
    })
    assert r.status_code == 200
    rep = r.json()["report"]
    assert all(rep["sections"][k] is not None
               for k in ("dataset", "model", "provenance", "distribution"))
    assert rep["governance"]["recommendation"] in ("ACCEPT", "REVIEW", "QUARANTINE")


def test_new_assessment_workflow_shipped_in_bundle(client):
    """UI scenarios 3/6/7 are enforced client-side (the API keeps every
    field optional by design): the shipped bundle must contain the modal,
    the dependency validations, and the empty-submission message."""
    static_dir = Path(__file__).resolve().parent.parent / "backend" / "static"
    js = "".join(p.read_text(encoding="utf-8")
                 for p in (static_dir / "assets").glob("index-*.js"))
    for marker in ("NEW ASSURANCE ASSESSMENT", "RUN ASSURANCE",
                   "Layers that will run", "/api/fs/browse",
                   "requires a Reference Model for the Phase 3",
                   "requires a Distribution Baseline for the Phase 5",
                   "Select at least one input"):
        assert marker in js, marker
