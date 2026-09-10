"""
generate_sample_assets.py
--------------------------
One-off utility script that creates the sample data shipped with Phase 1:

  * data/sample_images/*.jpg   - a few small synthetic RGB images
  * data/sample_model/sample_model.pth - a torchvision Faster R-CNN
    (MobileNetV3 backbone) checkpoint

IMPORTANT LIMITATION (documented also in the README):
This environment has no internet access to torchvision's official
pretrained-weights server, so the sample checkpoint is saved with
RANDOMLY INITIALIZED weights, not weights trained on COCO. The pipeline
that loads and runs this checkpoint is fully real and correct; only the
*quality* of its predictions is affected (they will look like noise).
When you run this project somewhere with normal internet access, set
`pretrained=True` in the config (or pass --pretrained to the demo) to
download and use real COCO-trained weights, which will produce
meaningful detections.

This script only needs to be run once; the outputs it produces are
already included in the shipped project.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "sample_images"
MODEL_DIR = ROOT / "data" / "sample_model"


def make_sample_images(n: int = 3) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    for i in range(1, n + 1):
        img = Image.new("RGB", (640, 480), color=(rng.randint(180, 255),
                                                    rng.randint(180, 255),
                                                    rng.randint(180, 255)))
        draw = ImageDraw.Draw(img)
        # Draw a couple of simple rectangles ("objects") so the image has
        # some visual structure, purely for a realistic-looking demo input.
        for _ in range(rng.randint(2, 4)):
            x0 = rng.randint(0, 500)
            y0 = rng.randint(0, 350)
            x1 = x0 + rng.randint(60, 130)
            y1 = y0 + rng.randint(60, 130)
            color = (rng.randint(0, 150), rng.randint(0, 150), rng.randint(0, 150))
            draw.rectangle([x0, y0, x1, y1], fill=color)
        out_path = IMAGES_DIR / f"image_{i:03d}.jpg"
        img.save(out_path, quality=90)
        print(f"Wrote {out_path}")


def make_sample_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=91
    )
    model.eval()
    out_path = MODEL_DIR / "sample_model.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Wrote {out_path} (randomly initialized weights, see module docstring)")


if __name__ == "__main__":
    make_sample_images()
    make_sample_model()
