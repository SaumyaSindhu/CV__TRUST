"""
generate_sample_dataset.py
----------------------------
Creates the SYNTHETIC / TEST dataset used to validate the Phase 2
Dataset Integrity Auditor, at ``data/sample_dataset/``.

This is not real-world data. It is deliberately constructed with known,
controlled cases so the test suite can verify the auditor actually
detects what it claims to detect:

  * "normal" images       -- ordinary-looking generated images, several
                              per synthetic class, that should NOT trigger
                              any detector.
  * exact duplicates       -- a byte-for-byte copy of an existing image.
  * near duplicates        -- a slightly modified copy (recompressed /
                              minor pixel jitter) of an existing image.
  * label inconsistency    -- an image visually similar to one class'
                              cluster but labeled with a different class.
  * a visually unusual/OOD sample -- an image with a very different
                              color/structure profile from everything else.

The detectors' outputs are NOT hard-coded anywhere in the test suite --
tests run the real detectors against this data and check the results.

This script only needs to be run once; its output is shipped with the
project. Re-run it if you want to regenerate the synthetic dataset.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "sample_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Synthetic class ids used throughout this dataset. 0=square, 1=circle,
# 2=triangle -- shapes are used (rather than photographic content) so the
# dataset is small, dependency-free, and clearly synthetic.
CLASS_NAMES = ["square", "circle", "triangle"]


def _blank_canvas(rng: random.Random, bg_low=245, bg_high=255) -> Image.Image:
    color = (rng.randint(bg_low, bg_high), rng.randint(bg_low, bg_high), rng.randint(bg_low, bg_high))
    return Image.new("RGB", (256, 256), color=color)


def _draw_shape(img: Image.Image, shape: str, rng: random.Random, color) -> None:
    draw = ImageDraw.Draw(img)
    cx, cy = rng.randint(100, 156), rng.randint(100, 156)
    r = rng.randint(70, 95)
    if shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def _save(img: Image.Image, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality)


def _write_label(path: Path, class_id: int, cx=0.5, cy=0.5, w=0.5, h=0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")


def _jitter_copy(img: Image.Image, rng: random.Random) -> Image.Image:
    """Return a near-duplicate: same content, minor pixel-level changes
    from a JPEG re-encode at a different quality plus a 1px shift."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    buf.seek(0)
    jittered = Image.open(buf).convert("RGB")
    return jittered.crop((1, 1, 255, 255)).resize((256, 256))


def generate(n_normal_per_class: int = 5, seed: int = 7) -> None:
    rng = random.Random(seed)

    if IMAGES_DIR.exists():
        for f in IMAGES_DIR.glob("*"):
            f.unlink()
    if LABELS_DIR.exists():
        for f in LABELS_DIR.glob("*"):
            f.unlink()

    manifest = []  # (filename, class_id, note)

    # --- 1. Normal images: several per class, consistent color palette per class ---
    class_colors = {
        0: (200, 60, 60),   # square -> reddish
        1: (60, 120, 200),  # circle -> blueish
        2: (60, 180, 90),   # triangle -> greenish
    }
    counter = 1
    normal_images_by_class = {0: [], 1: [], 2: []}
    for class_id, shape in enumerate(CLASS_NAMES):
        for _ in range(n_normal_per_class):
            img = _blank_canvas(rng)
            base_color = class_colors[class_id]
            jittered_color = tuple(
                max(0, min(255, c + rng.randint(-8, 8))) for c in base_color
            )
            _draw_shape(img, shape, rng, jittered_color)
            name = f"image_{counter:03d}.jpg"
            _save(img, IMAGES_DIR / name)
            _write_label(LABELS_DIR / f"image_{counter:03d}.txt", class_id)
            normal_images_by_class[class_id].append(name)
            manifest.append((name, class_id, "normal"))
            counter += 1

    # --- 2. Exact duplicate: byte-for-byte copy of an existing "square" image ---
    source_name = normal_images_by_class[0][0]
    dup_name = f"image_{counter:03d}.jpg"
    (IMAGES_DIR / dup_name).write_bytes((IMAGES_DIR / source_name).read_bytes())
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 0)
    manifest.append((dup_name, 0, f"exact_duplicate_of:{source_name}"))
    counter += 1

    # --- 3. Near duplicate: recompressed/shifted copy of a "circle" image ---
    source_name = normal_images_by_class[1][0]
    with Image.open(IMAGES_DIR / source_name) as src:
        near = _jitter_copy(src.convert("RGB"), rng)
    near_name = f"image_{counter:03d}.jpg"
    _save(near, IMAGES_DIR / near_name, quality=60)
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 1)
    manifest.append((near_name, 1, f"near_duplicate_of:{source_name}"))
    counter += 1

    # --- 4. Label inconsistency: a "square"-colored/shaped image mislabeled as circle ---
    img = _blank_canvas(rng)
    base_color = class_colors[0]
    jittered_color = tuple(max(0, min(255, c + rng.randint(-8, 8))) for c in base_color)
    _draw_shape(img, "square", rng, jittered_color)
    mislabeled_name = f"image_{counter:03d}.jpg"
    _save(img, IMAGES_DIR / mislabeled_name)
    # Deliberately wrong label (class 1 "circle") for a visually square-like image.
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 1)
    manifest.append((mislabeled_name, 1, "intentionally_inconsistent_label(visual=square)"))
    counter += 1

    # --- 5. Visually unusual / OOD sample: high-frequency noise, unlike anything else ---
    noise_img = Image.effect_noise((256, 256), 60).convert("RGB")
    ood_name = f"image_{counter:03d}.jpg"
    _save(noise_img, IMAGES_DIR / ood_name)
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 2)
    manifest.append((ood_name, 2, "visually_unusual_ood_sample"))
    counter += 1

    # --- 6. A couple of intentionally invalid samples for the validator ---
    # 6a. Corrupted "image" (not actually valid image bytes).
    corrupt_name = f"image_{counter:03d}.jpg"
    (IMAGES_DIR / corrupt_name).write_bytes(b"not a real jpeg file")
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 0)
    manifest.append((corrupt_name, None, "corrupted_image"))
    counter += 1

    # 6b. Malformed label line (wrong number of fields) on an otherwise-fine image.
    img = _blank_canvas(rng)
    _draw_shape(img, "triangle", rng, class_colors[2])
    malformed_name = f"image_{counter:03d}.jpg"
    _save(img, IMAGES_DIR / malformed_name)
    (LABELS_DIR / f"image_{counter:03d}.txt").write_text("2 0.5 0.5\n")  # missing w/h fields
    manifest.append((malformed_name, None, "malformed_label_line"))
    counter += 1

    # 6c. Invalid bounding box (out of [0,1] range) on an otherwise-fine image.
    img = _blank_canvas(rng)
    _draw_shape(img, "square", rng, class_colors[0])
    bad_box_name = f"image_{counter:03d}.jpg"
    _save(img, IMAGES_DIR / bad_box_name)
    _write_label(LABELS_DIR / f"image_{counter:03d}.txt", 0, cx=1.5, cy=0.5, w=0.3, h=0.3)
    manifest.append((bad_box_name, None, "invalid_bounding_box"))
    counter += 1

    manifest_path = DATASET_DIR / "SYNTHETIC_MANIFEST.md"
    lines = [
        "# Synthetic Test Dataset Manifest",
        "",
        "**This entire dataset directory is synthetic test data**, generated by",
        "`scripts/generate_sample_dataset.py` for validating the Phase 2 Dataset",
        "Integrity Auditor. It contains no real-world images.",
        "",
        "| file | ground-truth class | note |",
        "|---|---|---|",
    ]
    for name, class_id, note in manifest:
        cname = CLASS_NAMES[class_id] if class_id is not None else "n/a"
        lines.append(f"| {name} | {cname} | {note} |")
    manifest_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {counter - 1} image/label pairs under {DATASET_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    generate()
