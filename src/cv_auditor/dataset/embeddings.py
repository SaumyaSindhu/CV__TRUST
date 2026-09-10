"""
embeddings.py
-------------
Turns images into fixed-length numeric vectors ("embeddings") using a
pretrained TorchVision CNN, for use by the label-anomaly and OOD/anomaly
detectors later in the pipeline.

Design notes
------------
* Modular on purpose: everything downstream (``ood.py``,
  ``label_anomaly.py``) only depends on "image_id -> numpy vector",
  never on which network produced it. Swapping ``resnet18`` for a
  different backbone later is a one-line change here.
* Cached on purpose: embeddings are cached to disk keyed by the *image's
  SHA-256 content hash* (not its path), so identical images -- including
  ones already flagged as exact/near duplicates -- automatically share a
  cache entry, and renaming/moving a file doesn't invalidate its cache.
* Offline-safe on purpose, matching the precedent set in Phase 1's
  ``ModelLoader``: this environment has no internet access to
  TorchVision's pretrained-weights server. ``EmbeddingExtractor`` always
  *attempts* to download official pretrained (ImageNet) weights for the
  TorchVision backbone first. If that fails, it does NOT fall back to a
  randomly-initialized CNN -- untrained conv features pooled globally
  turn out to be nearly indistinguishable from one another in practice
  (cosine similarity collapses toward 1.0 for arbitrary inputs), which
  would silently make every downstream similarity/anomaly score
  meaningless while looking like it still works.

  Instead, the offline fallback is a small, fully explainable
  *handcrafted* descriptor (foreground color statistics computed after a
  simple median-background subtraction, plus a whole-image gradient-
  magnitude texture signal -- see ``_handcrafted_embedding`` below).
  It captures coarse color/texture structure using nothing but PIL and
  numpy, is deterministic, and is honest about what it can/can't
  distinguish. Which backend produced a given embedding is always
  recorded (``EmbeddingExtractor.backend`` / ``pretrained``) and surfaced
  in the audit report, so results can be interpreted accordingly. See
  README "Known Limitations".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..utils import get_logger

logger = get_logger(__name__)

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmbeddingExtractor:
    """Extracts and caches image embeddings using a TorchVision backbone."""

    def __init__(
        self,
        model_name: str = "resnet18",
        device: str = "cpu",
        cache_dir: Optional[str] = "results/embedding_cache",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.pretrained = True
        self.backend = "resnet18_pretrained"
        self._model: Optional[torch.nn.Module] = self._build_model()

    # -- model construction -------------------------------------------------

    def _build_model(self) -> Optional[torch.nn.Module]:
        import torchvision.models as tv_models

        if self.model_name != "resnet18":
            raise ValueError(
                f"Unsupported embedding model '{self.model_name}'. "
                f"Only 'resnet18' is wired up in Phase 2; add new backbones "
                f"here as needed."
            )

        try:
            backbone = tv_models.resnet18(weights="DEFAULT")
            self.pretrained = True
            self.backend = "resnet18_pretrained"
            logger.info("Loaded pretrained ImageNet weights for resnet18 embeddings.")
        except Exception as exc:  # noqa: BLE001 - no internet access, HTTP errors, etc.
            logger.warning(
                "Could not download pretrained resnet18 weights (%s). Falling back to "
                "a handcrafted color/texture-histogram embedding instead of a randomly "
                "initialized CNN (untrained pooled CNN features are not discriminative "
                "enough to be useful here) -- see README 'Known Limitations'.",
                exc,
            )
            self.pretrained = False
            self.backend = "handcrafted_color_texture"
            return None

        # Drop the final classification layer; keep the pooled feature
        # vector (512-d for resnet18) as the embedding.
        feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
        feature_extractor.to(self.device)
        feature_extractor.eval()
        return feature_extractor

    # -- caching --------------------------------------------------------

    def _cache_path(self, content_hash: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self.backend}_{content_hash}.json"

    def _load_from_cache(self, content_hash: str) -> Optional[np.ndarray]:
        path = self._cache_path(content_hash)
        if not path or not path.exists():
            return None
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            return np.array(data["embedding"], dtype=np.float32)
        except Exception:  # noqa: BLE001 - corrupt cache entry, ignore and recompute
            return None

    def _save_to_cache(self, content_hash: str, vector: np.ndarray) -> None:
        path = self._cache_path(content_hash)
        if not path:
            return
        with open(path, "w") as fh:
            json.dump({"embedding": vector.tolist()}, fh)

    # -- public API -------------------------------------------------------

    @torch.no_grad()
    def extract(self, image_path: str, content_hash: Optional[str] = None) -> np.ndarray:
        """Return the embedding vector for one image, using the cache if possible.

        ``content_hash`` should be the image's SHA-256 (already computed
        by the dataset loader) when available -- passing it avoids
        re-hashing the file just for cache lookup. If omitted, it is
        computed from the file on demand.
        """
        content_hash = content_hash or _sha256_of_file(image_path)

        cached = self._load_from_cache(content_hash)
        if cached is not None:
            return cached

        if self._model is not None:
            with Image.open(image_path) as img:
                tensor = _PREPROCESS(img.convert("RGB")).unsqueeze(0).to(self.device)
            features = self._model(tensor)  # [1, C, 1, 1]
            vector = features.flatten().cpu().numpy().astype(np.float32)
        else:
            vector = _handcrafted_embedding(image_path)

        self._save_to_cache(content_hash, vector)
        return vector

    def extract_batch(self, samples: List, show_progress: bool = False) -> Dict[str, np.ndarray]:
        """Extract embeddings for a list of ``dataset_loader.Sample`` objects.

        Unreadable samples are silently skipped (they have no valid image
        to embed); callers should intersect any downstream results with
        the set of keys actually returned here.
        """
        vectors: Dict[str, np.ndarray] = {}
        for i, sample in enumerate(samples):
            if not sample.readable:
                continue
            try:
                vectors[sample.image_id] = self.extract(sample.image_path, sample.sha256)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not embed '%s': %s", sample.image_id, exc)
            if show_progress and (i + 1) % 50 == 0:
                logger.info("Embedded %d/%d images", i + 1, len(samples))
        return vectors


def _handcrafted_embedding(image_path: str, size: int = 64, bg_distance_threshold: float = 30.0) -> np.ndarray:
    """Offline-safe fallback embedding: foreground color/texture descriptor.

    Deliberately simple and fully explainable, in three steps:

      1. Estimate the background color from the image's four corner
         patches (small 8x8 blocks). Corners are a common, robust place
         to sample "background" for typically-composed images (subject
         roughly centered) without assuming what fraction of the frame
         the subject occupies -- unlike a whole-image median/mode, which
         breaks down once the subject covers close to (or more than)
         half the frame. This is a documented simplifying assumption,
         not a general-purpose foreground segmentation method; see
         README "Known Limitations".
      2. Mask "foreground" pixels as those whose Euclidean distance from
         the background estimate exceeds ``bg_distance_threshold`` --
         i.e. pixels that visually stand out from the majority surface.
      3. Describe the image via the foreground pixels' mean and std RGB
         color, the foreground area fraction, and a whole-image gradient-
         magnitude mean (coarse texture/edge-sharpness signal).

    Why foreground-relative rather than a plain whole-image histogram:
    a plain histogram is dominated by whatever covers the most pixels
    (typically background/sky/floor/table), which can vary a lot between
    otherwise similar images and drown out the more class-relevant
    foreground content. Recentering on the foreground gives a much more
    stable, class-discriminative signal for the coarse similarity use
    cases this fallback needs to support (near-duplicate-adjacent
    clustering for label-anomaly detection, and kNN-based OOD scoring).

    The result is L2-normalized so cosine distance behaves sensibly.
    """
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32)

    corner = size // 8 or 1
    corner_pixels = np.concatenate([
        rgb[:corner, :corner].reshape(-1, 3),
        rgb[:corner, -corner:].reshape(-1, 3),
        rgb[-corner:, :corner].reshape(-1, 3),
        rgb[-corner:, -corner:].reshape(-1, 3),
    ])
    background = corner_pixels.mean(axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    mask = distance > bg_distance_threshold

    if mask.sum() < 5:  # near-uniform image (or nothing stands out from the background)
        fg_mean = background
        fg_std = np.zeros(3, dtype=np.float32)
        fg_fraction = 0.0
    else:
        fg_pixels = rgb[mask]
        fg_mean = fg_pixels.mean(axis=0)
        fg_std = fg_pixels.std(axis=0)
        fg_fraction = float(mask.mean())

    gray = rgb.mean(axis=2)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient_mean = float(np.sqrt(gx**2 + gy**2).mean())

    vector = np.concatenate([
        fg_mean / 255.0,
        fg_std / 255.0,
        [fg_fraction],
        [gradient_mean / 255.0],
    ]).astype(np.float32)

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _sha256_of_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
