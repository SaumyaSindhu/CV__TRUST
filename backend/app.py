
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from cv_auditor import __version__ as project_version  # noqa: E402
from cv_auditor.governance import AssuranceConfig, AssuranceEngine  # noqa: E402
from cv_auditor.provenance import AuditLog  # noqa: E402

APP_TITLE = "CV Assurance Backend"
STORE_DIR = PROJECT_ROOT / "results" / "api_store"
AUDIT_LOG_PATH = PROJECT_ROOT / "results" / "api_store" / "assurance_audit_log.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title=APP_TITLE, version=project_version)

class DatasetAuditRequest(BaseModel):
    dataset_dir: str
    dataset_format: str = "yolo_folder"
    device: str = "cpu"


class ModelAuditRequest(BaseModel):
    candidate_model_path: str
    reference_model_path: Optional[str] = None
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    test_images_dir: str = "data/sample_images"
    confidence_threshold: float = 0.5
    detector_score_thresh: Optional[float] = None
    device: str = "cpu"


class InferenceVerifyRequest(BaseModel):
    record_path: str
    key_path: Optional[str] = None
    image_path: Optional[str] = None
    model_path: Optional[str] = None


class DistributionAuditRequest(BaseModel):
    incoming_dir: str
    baseline_path: str
    metadata: Optional[Dict] = None
    device: str = "cpu"


class AssuranceRunRequest(BaseModel):
    dataset_dir: Optional[str] = None
    dataset_format: str = "yolo_folder"
    candidate_model_path: Optional[str] = None
    reference_model_path: Optional[str] = None
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    test_images_dir: str = "data/sample_images"
    confidence_threshold: float = 0.5
    detector_score_thresh: Optional[float] = None
    inference_record_path: Optional[str] = None
    provenance_key_path: Optional[str] = None
    provenance_image_path: Optional[str] = None
    provenance_model_path: Optional[str] = None
    incoming_dir: Optional[str] = None
    distribution_baseline_path: Optional[str] = None
    incoming_metadata: Optional[Dict] = None
    device: str = "cpu"


class AnalystDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(ACCEPT|REVIEW|QUARANTINE)$")
    analyst: str
    note: str = ""

def _store(kind: str, report_id: str, data: dict) -> dict:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    envelope = {"report_id": report_id, "kind": kind, "report": data}
    with open(STORE_DIR / f"{report_id}.json", "w") as f:
        json.dump(envelope, f, indent=2)
    return envelope


def _load(report_id: str) -> dict:
    path = STORE_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown report id '{report_id}'")
    with open(path) as f:
        return json.load(f)


def _list_reports() -> List[dict]:
    if not STORE_DIR.exists():
        return []
    out = []
    for p in sorted(STORE_DIR.glob("*.json")):
        try:
            with open(p) as f:
                env = json.load(f)
            out.append({"report_id": env["report_id"], "kind": env["kind"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _wrap_errors(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/health")
def health():
    return {"status": "ok", "service": APP_TITLE, "version": project_version,
            "phases": ["dataset", "model", "provenance", "distribution",
                       "governance"]}

import os as _os

BROWSE_ROOT = Path(_os.environ.get("CV_ASSURANCE_BROWSE_ROOT",
                                   str(PROJECT_ROOT))).resolve()
_BROWSE_SKIP = {"__pycache__", "node_modules", ".git", ".pytest_cache"}


@app.get("/api/fs/browse")
def fs_browse(path: Optional[str] = None):
    base = Path(path).resolve() if path else BROWSE_ROOT
    try:
        base.relative_to(BROWSE_ROOT)
    except ValueError:
        raise HTTPException(status_code=403,
                            detail=f"Browsing is restricted to {BROWSE_ROOT}")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {base}")
    entries = []
    for child in sorted(base.iterdir(),
                        key=lambda c: (not c.is_dir(), c.name.lower())):
        if child.name.startswith(".") or child.name in _BROWSE_SKIP:
            continue
        entries.append({"name": child.name, "path": str(child),
                        "is_dir": child.is_dir()})
        if len(entries) >= 500:
            break
    return {
        "root": str(BROWSE_ROOT),
        "path": str(base),
        "parent": str(base.parent) if base != BROWSE_ROOT else None,
        "entries": entries,
    }


@app.post("/api/datasets/audit")
def dataset_audit(req: DatasetAuditRequest):
    def run():
        from cv_auditor.dataset import DatasetAuditConfig, DatasetAuditor
        report_id = f"dataset-{uuid.uuid4().hex[:12]}"
        report = DatasetAuditor(DatasetAuditConfig(
            dataset_dir=req.dataset_dir, dataset_format=req.dataset_format,
            device=req.device, embedding_cache_dir=None,
            output_path=str(STORE_DIR / f"{report_id}_native.json"),
        )).run()
        return _store("dataset_audit", report_id, report.to_dict())
    return _wrap_errors(run)


@app.post("/api/models/audit")
def model_audit(req: ModelAuditRequest):
    def run():
        from cv_auditor.model_audit import ModelAuditConfig, ModelAuditor
        report_id = f"model-{uuid.uuid4().hex[:12]}"
        report = ModelAuditor(ModelAuditConfig(
            candidate_model_path=req.candidate_model_path,
            reference_model_path=req.reference_model_path,
            reference_behavior_path=str(STORE_DIR / f"{report_id}_refbehavior.json"),
            regenerate_reference_behavior=True,
            architecture=req.architecture,
            test_images_dir=req.test_images_dir,
            confidence_threshold=req.confidence_threshold,
            detector_score_thresh=req.detector_score_thresh,
            device=req.device,
            output_path=str(STORE_DIR / f"{report_id}_native.json"),
        )).run()
        return _store("model_audit", report_id, report.to_dict())
    return _wrap_errors(run)


@app.post("/api/inference/verify")
def inference_verify(req: InferenceVerifyRequest):
    def run():
        from cv_auditor.provenance import ProvenanceVerifier, load_key, load_record
        report_id = f"provenance-{uuid.uuid4().hex[:12]}"
        key = load_key(req.key_path) if req.key_path else None
        record = load_record(req.record_path)
        report = ProvenanceVerifier(key=key).verify(
            record, record_path=req.record_path,
            image_path=req.image_path, model_path=req.model_path,
        )
        return _store("inference_verification", report_id, report.to_dict())
    return _wrap_errors(run)


@app.post("/api/distribution/audit")
def distribution_audit(req: DistributionAuditRequest):
    def run():
        from cv_auditor.distribution import (
            DistributionAuditor, DistributionBaseline, DistributionConfig,
        )
        report_id = f"distribution-{uuid.uuid4().hex[:12]}"
        baseline = DistributionBaseline.load(req.baseline_path)
        report = DistributionAuditor(DistributionConfig(
            baseline_path=req.baseline_path, device=req.device,
        )).run(req.incoming_dir, baseline=baseline,
               current_metadata=req.metadata)
        return _store("distribution_audit", report_id, report.to_dict())
    return _wrap_errors(run)


@app.post("/api/assurance/run")
def assurance_run(req: AssuranceRunRequest):
    def run():
        engine = AssuranceEngine(AssuranceConfig(
            dataset_dir=req.dataset_dir, dataset_format=req.dataset_format,
            candidate_model_path=req.candidate_model_path,
            reference_model_path=req.reference_model_path,
            architecture=req.architecture,
            test_images_dir=req.test_images_dir,
            confidence_threshold=req.confidence_threshold,
            detector_score_thresh=req.detector_score_thresh,
            inference_record_path=req.inference_record_path,
            provenance_key_path=req.provenance_key_path,
            provenance_image_path=req.provenance_image_path,
            provenance_model_path=req.provenance_model_path,
            incoming_dir=req.incoming_dir,
            distribution_baseline_path=req.distribution_baseline_path,
            incoming_metadata=req.incoming_metadata,
            device=req.device,
            workdir=str(STORE_DIR / "assurance_work"),
            audit_log_path=str(AUDIT_LOG_PATH),
        ))
        report = engine.run()
        report.save(str(STORE_DIR / f"{report.assessment_id}_native.json"))
        return _store("assurance_assessment", report.assessment_id,
                      report.to_dict())
    return _wrap_errors(run)


@app.post("/api/reports/{report_id}/decision")
def record_decision(report_id: str, req: AnalystDecisionRequest):
    def run():
        env = _load(report_id)
        if env["kind"] != "assurance_assessment":
            raise HTTPException(
                status_code=400,
                detail="Analyst decisions attach to assurance assessments only.",
            )
        native = STORE_DIR / f"{report_id}_native.json"
        entry = AssuranceEngine.record_analyst_decision(
            str(native), req.decision, req.analyst,
            audit_log_path=str(AUDIT_LOG_PATH), note=req.note,
        )
        with open(native) as f:
            env["report"] = json.load(f)
        _store("assurance_assessment", report_id, env["report"])
        return {"report_id": report_id, "analyst_decision": entry}
    return _wrap_errors(run)


@app.get("/api/findings")
def findings():
    """All governance findings across stored assurance assessments."""
    out = []
    for meta in _list_reports():
        if meta["kind"] != "assurance_assessment":
            continue
        env = _load(meta["report_id"])
        for f in env["report"].get("findings", []):
            out.append({"report_id": meta["report_id"], **f})
    return {"count": len(out), "findings": out}


@app.get("/api/reports")
def reports():
    return {"reports": _list_reports()}


@app.get("/api/reports/{report_id}")
def report(report_id: str):
    return _load(report_id)


@app.get("/api/audit-log")
def audit_log():
    log = AuditLog(str(AUDIT_LOG_PATH))
    entries = log.read_entries()
    chain = log.verify_chain()
    return {
        "log_path": str(AUDIT_LOG_PATH),
        "num_entries": len(entries),
        "chain_valid": chain.valid,
        "chain_issues": [i.to_dict() for i in chain.issues],
        "entries": entries,
    }

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
