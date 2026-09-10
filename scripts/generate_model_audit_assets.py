"""
generate_model_audit_assets.py
---------------------------------
Creates the SYNTHETIC / TEST model checkpoints used to validate the
Phase 3 Model Integrity Auditor, under ``data/sample_models/``.

Reuses Phase 1's existing ``data/sample_model/sample_model.pth`` as the
"reference" model (copied here, unmodified) and derives three candidate
checkpoints from it with controlled, documented differences:

  * ``candidate_identical.pth``   -- byte-for-byte copy of the reference.
                                      Expected: no findings at all.
  * ``candidate_minor_noise.pth`` -- reference weights + tiny Gaussian
                                      noise (std=1e-4) added to every
                                      parameter. The file hash changes,
                                      but behavior should stay close to
                                      the reference. Expected: a single
                                      low-severity "model_file_changed"
                                      finding.
  * ``candidate_backdoor_like.pth`` -- reference weights, but with the
                                      classification head's bias for one
                                      target class (COCO class_id=5,
                                      "airplane") pushed sharply upward.
                                      This makes the candidate confidently
                                      predict "airplane" on almost every
                                      internal region proposal, regardless
                                      of image content -- a textbook
                                      symptom of a classifier-level
                                      backdoor (a change that makes a
                                      model confidently favor one target
                                      class everywhere). Expected: a
                                      high-severity
                                      "possible_trojan_indicator" finding.

IMPORTANT LIMITATION (documented also in the README): because this
sandboxed environment has no internet access to torchvision's official
pretrained-weights server, the *reference* model itself has randomly
initialized (untrained) weights, same as Phase 1. The auditor code and
these test fixtures are fully real; only the semantic *meaningfulness*
of "airplane" as a concept is unaffected by that -- the classification
head's structure and the bias-injection mechanism are exactly what a
real backdoor targeting a real trained model would look like at the
parameter level.

This script only needs to be run once; its output is shipped with the
project. Re-run it if you want to regenerate the synthetic checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SOURCE = ROOT / "data" / "sample_model" / "sample_model.pth"
OUT_DIR = ROOT / "data" / "sample_models"

TARGET_CLASS_ID = 5  # COCO "airplane"
BIAS_BOOST = 40.0     # large enough to dominate the softmax for every proposal
MINOR_NOISE_STD = 1e-4


def generate() -> None:
    if not REFERENCE_SOURCE.exists():
        raise FileNotFoundError(
            f"Expected Phase 1's sample checkpoint at {REFERENCE_SOURCE}. "
            f"Run scripts/generate_sample_assets.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_state = torch.load(str(REFERENCE_SOURCE), map_location="cpu")

    reference_path = OUT_DIR / "reference_model.pth"
    torch.save(reference_state, reference_path)
    print(f"Wrote {reference_path}")

    # --- candidate_identical.pth: byte-for-byte copy ---
    identical_path = OUT_DIR / "candidate_identical.pth"
    identical_path.write_bytes(reference_path.read_bytes())
    print(f"Wrote {identical_path}")

    # --- candidate_minor_noise.pth: tiny noise on every parameter ---
    torch.manual_seed(1234)
    minor_state = {}
    for key, tensor in reference_state.items():
        if torch.is_floating_point(tensor):
            minor_state[key] = tensor + torch.randn_like(tensor) * MINOR_NOISE_STD
        else:
            minor_state[key] = tensor.clone()
    minor_path = OUT_DIR / "candidate_minor_noise.pth"
    torch.save(minor_state, minor_path)
    print(f"Wrote {minor_path}")

    # --- candidate_backdoor_like.pth: bias boost on one target class ---
    backdoor_state = {k: v.clone() for k, v in reference_state.items()}
    bias_key = "roi_heads.box_predictor.cls_score.bias"
    if bias_key not in backdoor_state:
        raise KeyError(
            f"Expected key '{bias_key}' in the checkpoint's state_dict -- "
            f"the architecture may have changed; update this script."
        )
    backdoor_state[bias_key][TARGET_CLASS_ID] += BIAS_BOOST
    backdoor_path = OUT_DIR / "candidate_backdoor_like.pth"
    torch.save(backdoor_state, backdoor_path)
    print(f"Wrote {backdoor_path} (class_id={TARGET_CLASS_ID} bias boosted by +{BIAS_BOOST})")

    manifest_path = OUT_DIR / "SYNTHETIC_MANIFEST.md"
    manifest_path.write_text(
        "# Synthetic Test Model Checkpoints Manifest\n\n"
        "**These checkpoints are synthetic test fixtures**, generated by "
        "`scripts/generate_model_audit_assets.py` for validating the Phase 3 "
        "Model Integrity Auditor. The reference weights themselves are the same "
        "randomly-initialized (untrained) weights used in Phase 1 -- see "
        "README \"Known Limitations\".\n\n"
        "| file | derived from reference by | expected audit outcome |\n"
        "|---|---|---|\n"
        "| reference_model.pth | (is the reference) | n/a |\n"
        "| candidate_identical.pth | byte-for-byte copy | no findings |\n"
        f"| candidate_minor_noise.pth | + N(0, {MINOR_NOISE_STD}) noise on every param | "
        "low-severity 'model_file_changed' only |\n"
        f"| candidate_backdoor_like.pth | class_id={TARGET_CLASS_ID} bias += {BIAS_BOOST} | "
        "high-severity 'possible_trojan_indicator' |\n"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    generate()
