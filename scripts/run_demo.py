#!/usr/bin/env python3
"""
scripts/run_demo.py
--------------------
Reproducible, fully offline end-to-end demo of the complete assurance
system. Twelve steps, small deterministic fixtures, no external
dependencies (TRACE remains only an external Phase-3 validation
benchmark documented in the README -- nothing here depends on it).

    python scripts/run_demo.py            # everything under results/demo/

Each step prints what was done, the key numbers, and where the artifact
lives, so the run doubles as a live walkthrough for a jury/evaluator.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

DEMO = ROOT / "results" / "demo"

STEP = 0


def step(title: str):
    global STEP
    STEP += 1
    print(f"\n{'=' * 70}\nSTEP {STEP:02d} -- {title}\n{'=' * 70}")


def main() -> int:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True)

    import torch

    from cv_auditor.dataset import DatasetAuditConfig, DatasetAuditor
    from cv_auditor.distribution import (
        DistributionAuditor, DistributionConfig, build_baseline,
    )
    from cv_auditor.governance import AssuranceConfig, AssuranceEngine
    from cv_auditor.model_audit import ModelAuditConfig, ModelAuditor
    from cv_auditor.provenance import (
        AuditLog, ProvenanceVerifier, load_record, protect_payload,
    )
    from distribution_helpers import (
        ANOMALY_STYLE, BASELINE_STYLE, SMALL_DRIFT_STYLE, WINTER_STYLE,
        make_batch, make_corrupt_file,
    )
    from model_audit_helpers import add_noise_checkpoint
    from provenance_helpers import make_payload

    sample_dataset = ROOT / "data" / "sample_dataset"
    sample_model = ROOT / "data" / "sample_model" / "sample_model.pth"
    sample_images = ROOT / "data" / "sample_images"

    # ------------------------------------------------------------------ 1
    step("Clean dataset -- Phase 2 dataset integrity audit")
    clean_ds = DEMO / "clean_dataset" / "images"
    make_batch(clean_ds, n=10, seed=100, style=BASELINE_STYLE)
    (DEMO / "clean_dataset" / "labels").mkdir(parents=True)
    for img in sorted(clean_ds.iterdir()):
        (DEMO / "clean_dataset" / "labels" / f"{img.stem}.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n")
    rep = DatasetAuditor(DatasetAuditConfig(
        dataset_dir=str(DEMO / "clean_dataset"), embedding_cache_dir=None,
        output_path=str(DEMO / "step01_clean_dataset.json"))).run()
    rep.save(str(DEMO / "step01_clean_dataset.json"))
    print(f"valid={rep.valid_samples} invalid={rep.invalid_samples} "
          f"level={rep.overall_finding_level}")

    # ------------------------------------------------------------------ 2
    step("Dataset anomaly -- duplicates + corrupt file are surfaced")
    bad_ds = DEMO / "anomalous_dataset" / "images"
    make_batch(bad_ds, n=8, seed=100, style=BASELINE_STYLE)  # same seed batch
    shutil.copy(sorted(bad_ds.iterdir())[0], bad_ds / "copy_of_first.png")
    make_corrupt_file(bad_ds, "broken.png")
    (DEMO / "anomalous_dataset" / "labels").mkdir(parents=True)
    for img in sorted(bad_ds.iterdir()):
        (DEMO / "anomalous_dataset" / "labels" / f"{img.stem}.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n")
    rep = DatasetAuditor(DatasetAuditConfig(
        dataset_dir=str(DEMO / "anomalous_dataset"), embedding_cache_dir=None,
        output_path=str(DEMO / "step02_anomalous_dataset.json"))).run()
    rep.save(str(DEMO / "step02_anomalous_dataset.json"))
    print(f"invalid={rep.invalid_samples} exact_dup_groups="
          f"{rep.exact_duplicate_groups} level={rep.overall_finding_level}")

    # ------------------------------------------------------------------ 3
    step("Clean/reference model -- identical candidate is 'clean'")
    rep = ModelAuditor(ModelAuditConfig(
        candidate_model_path=str(sample_model),
        reference_model_path=str(sample_model),
        reference_behavior_path=str(DEMO / "reference_behavior.json"),
        regenerate_reference_behavior=True,
        test_images_dir=str(sample_images), detector_score_thresh=0.0,
        output_path=str(DEMO / "step03_clean_model.json"))).run()
    rep.save(str(DEMO / "step03_clean_model.json"))
    print(f"fingerprint_changed={rep.fingerprint['changed']} "
          f"agreement={rep.comparison['agreement_rate']} category={rep.category}")

    # ------------------------------------------------------------------ 4
    step("Modified model -- small weight noise: file changed, behavior compared")
    modified = DEMO / "modified_model.pth"
    add_noise_checkpoint(str(sample_model), str(modified), std=1e-4)
    rep = ModelAuditor(ModelAuditConfig(
        candidate_model_path=str(modified),
        reference_model_path=str(sample_model),
        reference_behavior_path=str(DEMO / "reference_behavior.json"),
        test_images_dir=str(sample_images), detector_score_thresh=0.0,
        output_path=str(DEMO / "step04_modified_model.json"))).run()
    rep.save(str(DEMO / "step04_modified_model.json"))
    print(f"fingerprint_changed={rep.fingerprint['changed']} "
          f"agreement={rep.comparison['agreement_rate']} category={rep.category}")

    # ------------------------------------------------------------------ 5
    step("Backdoor-like model -- targeted head bias: trojan-style indicator")
    backdoor = DEMO / "backdoorlike_model.pth"
    # crank one class's classification bias -- the detector floods that
    # class with high-confidence detections (trojan-style signature)
    ckpt = torch.load(str(sample_model), map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    bias_key = next(k for k in sd if k.endswith("cls_score.bias"))
    sd[bias_key] = sd[bias_key].clone()
    sd[bias_key][5] += 30.0  # class id 5
    torch.save(sd, str(backdoor))
    rep = ModelAuditor(ModelAuditConfig(
        candidate_model_path=str(backdoor),
        reference_model_path=str(sample_model),
        reference_behavior_path=str(DEMO / "reference_behavior.json"),
        test_images_dir=str(sample_images), detector_score_thresh=0.0,
        output_path=str(DEMO / "step05_backdoorlike_model.json"))).run()
    rep.save(str(DEMO / "step05_backdoorlike_model.json"))
    print(f"category={rep.category} level={rep.overall_finding_level} "
          f"added={rep.comparison['total_added']} "
          f"dominant_class={rep.comparison['dominant_added_class_id']}")
    print("NOTE: reported as 'consistent with backdoor/trojan-style "
          "modification' -- an indicator, never a confirmation.")

    # ------------------------------------------------------------------ 6
    step("Valid inference record -- protected payload verifies cleanly")
    payload = make_payload(DEMO)
    record = protect_payload(payload)
    record_path = DEMO / "record_valid.json"
    record_path.write_text(json.dumps(record, indent=2))
    vrep = ProvenanceVerifier(key=None).verify(
        load_record(str(record_path)), record_path=str(record_path))
    vrep.save(str(DEMO / "step06_valid_record.json"))
    print(f"overall={vrep.overall_status} hash={vrep.hash_integrity}")

    # ------------------------------------------------------------------ 7
    step("Tampered inference output -- edited prediction breaks the digest")
    tampered = json.loads(record_path.read_text())
    tampered["payload"]["output"]["predictions"][0]["class"] = "person"
    tampered["payload"]["output"]["predictions"][0]["confidence"] = 0.11
    tpath = DEMO / "record_tampered_output.json"
    tpath.write_text(json.dumps(tampered, indent=2))
    vrep = ProvenanceVerifier(key=None).verify(
        load_record(str(tpath)), record_path=str(tpath))
    vrep.save(str(DEMO / "step07_tampered_output.json"))
    print(f"overall={vrep.overall_status} hash={vrep.hash_integrity} "
          f"codes={sorted({f.code for f in vrep.findings})}")

    # ------------------------------------------------------------------ 8
    step("Modified input/model/config -- re-hashed files no longer match record")
    image_path = DEMO / "image_001.jpg"
    image_path.write_bytes(image_path.read_bytes() + b" tampered")
    vrep = ProvenanceVerifier(key=None).verify(
        load_record(str(record_path)), record_path=str(record_path),
        image_path=str(image_path))
    vrep.save(str(DEMO / "step08_modified_input.json"))
    print(f"overall={vrep.overall_status} "
          f"codes={sorted({f.code for f in vrep.findings})}")
    cfg_tampered = json.loads(record_path.read_text())
    cfg_tampered["payload"]["inference_config"]["confidence_threshold"] = 0.01
    cpath = DEMO / "record_tampered_config.json"
    cpath.write_text(json.dumps(cfg_tampered, indent=2))
    vrep = ProvenanceVerifier(key=None).verify(
        load_record(str(cpath)), record_path=str(cpath))
    print(f"config-tamper overall={vrep.overall_status}")

    # ------------------------------------------------------------------ 9
    step("Normal distribution drift -- small change stays LOW/MEDIUM")
    ref_dir = DEMO / "distribution_reference"
    make_batch(ref_dir, n=16, seed=1, style=BASELINE_STYLE)
    baseline = build_baseline(str(ref_dir), DistributionConfig())
    baseline_path = DEMO / "baseline.json"
    baseline.save(str(baseline_path))
    drift_dir = DEMO / "incoming_drift"
    make_batch(drift_dir, n=10, seed=42, style=SMALL_DRIFT_STYLE)
    rep = DistributionAuditor(DistributionConfig()).run(
        str(drift_dir), baseline=baseline)
    rep.save(str(DEMO / "step09_normal_drift.json"))
    print(f"risk={rep.risk} shift_score={rep.comparison.shift_score} "
          f"recommendation={rep.recommendation}")

    # ----------------------------------------------------------------- 10
    step("Strong distribution anomaly -- winter batch + injected outliers")
    strong_dir = DEMO / "incoming_strong"
    make_batch(strong_dir, n=8, seed=5, style=WINTER_STYLE)
    make_batch(strong_dir, n=2, seed=999, style=ANOMALY_STYLE, prefix="odd")
    rep = DistributionAuditor(DistributionConfig()).run(
        str(strong_dir), baseline=baseline,
        current_metadata={"season": "winter"})
    rep.save(str(DEMO / "step10_strong_anomaly.json"))
    print(f"risk={rep.risk} pattern={rep.comparison.shift_pattern} "
          f"anomalous_fraction={rep.comparison.anomalous_fraction} "
          f"recommendation={rep.recommendation}")

    # ----------------------------------------------------------------- 11
    step("Final unified assurance report -- all four evidence layers")
    engine = AssuranceEngine(AssuranceConfig(
        dataset_dir=str(DEMO / "anomalous_dataset"),
        candidate_model_path=str(backdoor),
        reference_model_path=str(sample_model),
        test_images_dir=str(sample_images),
        detector_score_thresh=0.0,
        inference_record_path=str(tpath),
        incoming_dir=str(strong_dir),
        distribution_baseline_path=str(baseline_path),
        incoming_metadata={"season": "winter"},
        workdir=str(DEMO / "assurance_work"),
        audit_log_path=str(DEMO / "assurance_audit_log.jsonl"),
    ))
    report = engine.run()
    report_path = DEMO / "step11_unified_assurance_report.json"
    report.save(str(report_path))
    report.print_summary()
    AssuranceEngine.record_analyst_decision(
        str(report_path), "QUARANTINE", analyst="demo.analyst",
        audit_log_path=str(DEMO / "assurance_audit_log.jsonl"),
        note="Demo walkthrough decision.")
    print("\nAnalyst decision QUARANTINE recorded (audit-logged).")

    # ----------------------------------------------------------------- 12
    step("Audit trail -- chain verifies; tampering with history is detected")
    log = AuditLog(str(DEMO / "assurance_audit_log.jsonl"))
    chain = log.verify_chain()
    print(f"entries={chain.num_entries} chain_valid={chain.valid}")
    lines = (DEMO / "assurance_audit_log.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["record_digest"] = "0" * 64
    lines[0] = json.dumps(first, sort_keys=True)
    tampered_log = DEMO / "assurance_audit_log_TAMPERED.jsonl"
    tampered_log.write_text("\n".join(lines) + "\n")
    tchain = AuditLog(str(tampered_log)).verify_chain()
    print(f"after tampering entry 0: chain_valid={tchain.valid} "
          f"issues={len(tchain.issues)} (tamper-evidence demonstrated)")

    print(f"\n{'=' * 70}\nDemo complete -- all artifacts under {DEMO}\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
