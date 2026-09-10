"""
record.py (provenance)
-----------------------
Builds the canonical *inference provenance record*: one self-contained,
deterministic JSON document binding together

    input image  <->  model  <->  preprocessing config  <->
    inference config  <->  prediction/output  <->  run metadata

and the *protection block* (per-component SHA-256 digests + one overall
record digest, optionally an HMAC signature -- see ``signing.py``).

Record layout
-------------
::

    {
      "record_format_version": "1.0",
      "payload": {
        "record_id": "...",            # unique per record (uuid4 hex)
        "created_at": "...",           # ISO-8601 UTC
        "sequence_number": 7,          # monotonically increasing per log
        "nonce": "...",                # random per record (replay detection)
        "inference_timestamp": "...",  # when the model actually ran
        "software": {...},             # package/framework versions
        "input":  {...},               # image name/path/sha256/dimensions
        "model":  {...},               # identifier/architecture/weights sha256
        "preprocessing_config": {...},
        "inference_config": {...},
        "output": {"predictions": [...]}
      },
      "protection": {
        "hash_algorithm": "sha256",
        "canonicalization": "json/sorted-keys/compact/utf-8/v1",
        "component_digests": { "<component>": "<sha256 hex>", ... },
        "record_digest": "<sha256 hex over payload + component_digests>",
        "signature": null | {"scheme": "hmac-sha256", "key_id": "...", "value": "..."}
      }
    }

What the digests mean (and don't mean)
--------------------------------------
* ``component_digests[c]`` lets a verifier attribute a change to a specific
  component (input vs. model vs. config vs. output vs. metadata).
* ``record_digest`` covers the entire payload AND the component-digest
  table, so an attacker cannot silently fix up a component digest.
* Hashing alone is **integrity detection only**: anyone can recompute all
  digests after modifying the record. Detecting *that kind* of consistent
  re-forging requires the optional signature (``signing.py``), i.e. a
  secret the attacker does not have. The verifier reports these two layers
  separately ("hash integrity" vs. "signature") on purpose.
"""

from __future__ import annotations

import json
import platform
import secrets
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import ensure_parent_dir, get_logger, utc_timestamp
from .canonical import CANONICALIZATION_ID, HASH_ALGORITHM, sha256_of_obj

logger = get_logger(__name__)

RECORD_FORMAT_VERSION = "1.0"

# Payload keys that make up each verifiable component. "metadata" is the
# set of scalar bookkeeping fields; the rest are nested sections.
COMPONENT_SECTIONS = (
    "input",
    "model",
    "preprocessing_config",
    "inference_config",
    "output",
)
METADATA_FIELDS = (
    "record_id",
    "created_at",
    "sequence_number",
    "nonce",
    "inference_timestamp",
    "software",
)
ALL_COMPONENTS = COMPONENT_SECTIONS + ("metadata",)


def default_software_info() -> Dict[str, Any]:
    """Best-effort software/version metadata for the running environment.

    torch/torchvision are imported lazily and tolerated if missing so that
    provenance verification (which never needs them) stays importable in
    minimal environments.
    """
    from .. import __version__ as cv_auditor_version

    info: Dict[str, Any] = {
        "cv_auditor": cv_auditor_version,
        "python": platform.python_version(),
    }
    try:  # pragma: no cover - depends on environment
        import torch

        info["torch"] = str(torch.__version__)
    except Exception:
        info["torch"] = None
    try:  # pragma: no cover - depends on environment
        import torchvision

        info["torchvision"] = str(torchvision.__version__)
    except Exception:
        info["torchvision"] = None
    return info


def new_nonce() -> str:
    """128-bit random nonce, hex encoded."""
    return secrets.token_hex(16)


def new_record_id() -> str:
    """Unique record identifier (uuid4, hex)."""
    return uuid.uuid4().hex


def build_payload(
    *,
    input_info: Dict[str, Any],
    model_info: Dict[str, Any],
    preprocessing_config: Dict[str, Any],
    inference_config: Dict[str, Any],
    output: Dict[str, Any],
    sequence_number: int,
    record_id: Optional[str] = None,
    created_at: Optional[str] = None,
    nonce: Optional[str] = None,
    inference_timestamp: Optional[str] = None,
    software: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble an (unprotected) provenance payload dict.

    All arguments must already be JSON-native values; the caller is
    responsible for extracting them from live objects (see
    ``payload_from_inference``). Randomized/derived fields (record id,
    nonce, timestamps, software info) are filled in when not supplied,
    which keeps unit tests fully deterministic when they pass everything
    explicitly.
    """
    return {
        "record_id": record_id if record_id is not None else new_record_id(),
        "created_at": created_at if created_at is not None else utc_timestamp(),
        "sequence_number": int(sequence_number),
        "nonce": nonce if nonce is not None else new_nonce(),
        "inference_timestamp": inference_timestamp if inference_timestamp is not None else "",
        "software": software if software is not None else default_software_info(),
        "input": input_info,
        "model": model_info,
        "preprocessing_config": preprocessing_config,
        "inference_config": inference_config,
        "output": output,
    }


def payload_from_inference(
    image_result,
    model_metadata,
    config,
    *,
    sequence_number: int,
    record_id: Optional[str] = None,
    created_at: Optional[str] = None,
    nonce: Optional[str] = None,
    software: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a payload from live Phase 1 objects.

    Parameters
    ----------
    image_result:
        A Phase 1 ``ImageResult`` (one image's structured predictions).
    model_metadata:
        The Phase 1 ``ModelMetadata`` returned by ``ModelLoader.load()``.
    config:
        The ``PipelineConfig`` the run used.
    """
    input_info = {
        "image": image_result.image,
        "image_path": image_result.image_path,
        "image_sha256": image_result.image_sha256,
        "width": image_result.width,
        "height": image_result.height,
    }
    model_info = {
        "model_name": model_metadata.model_name,
        "architecture": model_metadata.architecture,
        "source": model_metadata.source,
        "weights_path": model_metadata.weights_path,
        "model_sha256": model_metadata.weights_sha256,
        "num_parameters": model_metadata.num_parameters,
        "num_classes": model_metadata.num_classes,
        "framework": model_metadata.framework,
        "framework_version": model_metadata.framework_version,
    }
    # Phase 1's ImageLoader applies exactly this preprocessing (see
    # image_loader.py): decode -> RGB -> float32 tensor in [0, 1]; any
    # resizing happens inside the torchvision model's own transform.
    preprocessing_config = {
        "color_space": "RGB",
        "tensor_dtype": "float32",
        "value_range": [0.0, 1.0],
        "resize": "model-internal (torchvision GeneralizedRCNNTransform)",
    }
    inference_config = {
        "confidence_threshold": config.confidence_threshold,
        "detector_score_thresh": config.detector_score_thresh,
        "device": config.device,
    }
    output = {
        "predictions": [p.to_dict() for p in image_result.predictions],
    }
    return build_payload(
        input_info=input_info,
        model_info=model_info,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config,
        output=output,
        sequence_number=sequence_number,
        record_id=record_id,
        created_at=created_at,
        nonce=nonce,
        inference_timestamp=image_result.inference_timestamp,
        software=software,
    )


def metadata_component(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the "metadata" component dict from a payload."""
    return {name: payload.get(name) for name in METADATA_FIELDS}


def compute_component_digests(payload: Dict[str, Any]) -> Dict[str, str]:
    """Per-component SHA-256 digests over canonical serialization."""
    digests = {
        section: sha256_of_obj(payload.get(section)) for section in COMPONENT_SECTIONS
    }
    digests["metadata"] = sha256_of_obj(metadata_component(payload))
    return digests


def compute_record_digest(payload: Dict[str, Any], component_digests: Dict[str, str]) -> str:
    """Overall record digest.

    Covers BOTH the payload and the component-digest table so that
    tampering with either is detectable, and binds the format version so
    records cannot be replayed across incompatible format revisions.
    """
    return sha256_of_obj(
        {
            "record_format_version": RECORD_FORMAT_VERSION,
            "payload": payload,
            "component_digests": component_digests,
        }
    )


def protect_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a payload into a full, hash-protected (unsigned) record."""
    component_digests = compute_component_digests(payload)
    record_digest = compute_record_digest(payload, component_digests)
    return {
        "record_format_version": RECORD_FORMAT_VERSION,
        "payload": payload,
        "protection": {
            "hash_algorithm": HASH_ALGORITHM,
            "canonicalization": CANONICALIZATION_ID,
            "component_digests": component_digests,
            "record_digest": record_digest,
            "signature": None,
        },
    }


def save_record(record: Dict[str, Any], path: str) -> str:
    """Write a record to disk as (pretty-printed) JSON.

    On-disk formatting is irrelevant to integrity -- all digests are over
    the *canonical* serialization, which is recomputed from the parsed
    structure at verification time.
    """
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Saved provenance record %s to %s", record["payload"].get("record_id", "?"), path)
    return path


def load_record(path: str) -> Dict[str, Any]:
    """Load a record JSON file. Raises on missing file or invalid JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Provenance record not found: {path}")
    with open(p, "r") as f:
        return json.load(f)
