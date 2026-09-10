"""
perceptual_hash.py
-------------------
Near-duplicate detection using a difference hash ("dHash").

Why dHash
---------
dHash is a small, well-understood, fully explainable perceptual hash:

    1. Shrink the image to a tiny (hash_size+1) x hash_size grayscale
       thumbnail. Shrinking discards high-frequency detail (JPEG noise,
       resaving artifacts, minor crops/recolors) while preserving coarse
       visual structure.
    2. For each row, compare each pixel to its right-hand neighbor: bit=1
       if the pixel is brighter, bit=0 otherwise. This encodes the
       *gradient* of the image, which is robust to small brightness/
       contrast shifts.
    3. Two images are "near-duplicates" if their hashes differ in only a
       few bits (small Hamming distance).

This intentionally avoids pulling in a dedicated perceptual-hashing or
deep-learning dependency for something a ~30 line, easily-audited
function can do (see "avoid unnecessary dependencies" in the project
principles). It also avoids conflating this step with the embedding-based
similarity used later for OOD/label-anomaly detection -- dHash answers
"are these two files near-identical images" specifically, not "are these
two images semantically similar".

Limitations (see README for the full list):
  * Not rotation/large-crop invariant.
  * Two genuinely different images can rarely collide (birthday-paradox
    style) for a small hash size -- this is why a similarity *threshold*
    is used rather than requiring an exact hash match (that's what
    ``duplicates.py`` is for).
"""

from __future__ import annotations

from typing import Dict, List

from PIL import Image
import numpy as np


def compute_dhash(image_path: str, hash_size: int = 8) -> int:
    """Compute a dHash for an image, returned as a single Python int bitmask.

    Raises
    ------
    ValueError
        If the image cannot be opened/decoded.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            pixels = np.asarray(img).flatten().tolist()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not compute perceptual hash for '{image_path}': {exc}") from exc

    width = hash_size + 1
    bits = 0
    bit_index = 0
    for row in range(hash_size):
        row_pixels = pixels[row * width:(row + 1) * width]
        for col in range(hash_size):
            if row_pixels[col] > row_pixels[col + 1]:
                bits |= (1 << bit_index)
            bit_index += 1
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity_from_hashes(a: int, b: int, hash_size: int = 8) -> float:
    """Convert a Hamming distance between two dHashes into a [0, 1] similarity."""
    total_bits = hash_size * hash_size
    return 1.0 - (hamming_distance(a, b) / total_bits)


class _UnionFind:
    """Minimal union-find used to group images into similarity clusters."""

    def __init__(self, items: List[str]):
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def compute_hashes(image_paths_by_id: Dict[str, str], hash_size: int = 8) -> Dict[str, int]:
    """Compute a dHash for every readable image. Unreadable images are skipped."""
    hashes: Dict[str, int] = {}
    for image_id, path in image_paths_by_id.items():
        try:
            hashes[image_id] = compute_dhash(path, hash_size=hash_size)
        except ValueError:
            continue
    return hashes


def find_near_duplicate_groups(
    image_paths_by_id: Dict[str, str],
    similarity_threshold: float = 0.90,
    hash_size: int = 8,
) -> List[Dict]:
    """Group images whose dHash similarity is >= ``similarity_threshold``.

    Returns a list of group dicts, each with ``members`` (list of
    image_ids) and ``min_pairwise_similarity`` (the weakest link in the
    group, useful context for how tight the group actually is).

    O(n^2) pairwise comparison. Fine for the dataset sizes this component
    targets (hundreds-to-low-thousands of images); a spatial index
    (e.g. LSH) would be the natural upgrade for much larger datasets and
    is noted as a limitation in the README rather than implemented here.
    """
    hashes = compute_hashes(image_paths_by_id, hash_size=hash_size)
    ids = list(hashes.keys())

    uf = _UnionFind(ids)
    pairwise_sim: Dict[frozenset, float] = {}

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            sim = similarity_from_hashes(hashes[id_a], hashes[id_b], hash_size=hash_size)
            if sim >= similarity_threshold:
                uf.union(id_a, id_b)
                pairwise_sim[frozenset((id_a, id_b))] = sim

    raw_groups = [members for members in uf.groups().values() if len(members) > 1]

    groups = []
    for members in raw_groups:
        sims = [
            pairwise_sim[frozenset((a, b))]
            for i, a in enumerate(members)
            for b in members[i + 1:]
            if frozenset((a, b)) in pairwise_sim
        ]
        groups.append({
            "members": sorted(members),
            "min_pairwise_similarity": min(sims) if sims else similarity_threshold,
        })
    return groups
