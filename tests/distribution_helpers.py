"""
distribution_helpers.py
------------------------
Small deterministic synthetic image batches for Phase 5 tests. Every
generator is seeded, so the same call always produces byte-stable
statistics (PNG output; no JPEG nondeterminism).

The generator paints a noisy "terrain" background plus a centered
foreground blob, giving both the Phase 2 foreground descriptor and the
Phase 5 global-statistics block real signal to work with.
"""

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 96

# canonical "domains". bg_jitter / brightness_jitter give every batch a
# natural per-image spread (real reference sets are not perfectly uniform),
# which is what makes "small drift" small RELATIVE to baseline variability.
BASELINE_STYLE = dict(bg=(100, 140, 90), fg=(150, 110, 70), noise=18.0,
                      brightness=1.0, bg_jitter=14.0, brightness_jitter=0.10)
SMALL_DRIFT_STYLE = dict(bg=(105, 144, 95), fg=(154, 114, 74), noise=19.0,
                         brightness=1.03, bg_jitter=14.0, brightness_jitter=0.10)
WINTER_STYLE = dict(bg=(228, 230, 240), fg=(190, 195, 205), noise=10.0,
                    brightness=1.15, bg_jitter=8.0, brightness_jitter=0.05)
ANOMALY_STYLE = dict(bg=(250, 20, 210), fg=(10, 250, 40), noise=60.0,
                     brightness=1.0, bg_jitter=5.0, brightness_jitter=0.02)


def make_image(path: Path, seed: int, bg, fg, noise: float, brightness: float,
               bg_jitter: float = 0.0, brightness_jitter: float = 0.0):
    rng = np.random.default_rng(seed)
    bg = np.array(bg, dtype=np.float64) + rng.uniform(-bg_jitter, bg_jitter, 3)
    brightness = brightness * (1.0 + rng.uniform(-brightness_jitter, brightness_jitter))
    img = np.ones((SIZE, SIZE, 3), dtype=np.float64) * bg
    img += rng.normal(0.0, noise, size=img.shape)

    # centered foreground blob (ellipse) with its own jitter
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cy, cx = SIZE / 2 + rng.uniform(-8, 8), SIZE / 2 + rng.uniform(-8, 8)
    mask = (((yy - cy) / 22.0) ** 2 + ((xx - cx) / 16.0) ** 2) <= 1.0
    img[mask] = np.array(fg, dtype=np.float64) + rng.normal(0.0, noise / 2, size=(mask.sum(), 3))

    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(path, format="PNG")


def make_batch(directory, n: int, seed: int, style=None, prefix: str = "img"):
    """Write ``n`` deterministic images of a given style into ``directory``."""
    style = style or BASELINE_STYLE
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = directory / f"{prefix}_{i:03d}.png"
        make_image(p, seed=seed * 10_000 + i, **style)
        paths.append(p)
    return paths


def make_corrupt_file(directory, name: str = "broken.jpg"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_bytes(b"\xff\xd8\xff\xe0 this is definitely not a decodable jpeg")
    return p
