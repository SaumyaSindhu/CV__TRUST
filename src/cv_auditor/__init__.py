"""
cv_auditor - Phase 1: CV Model Ingestion & Inference Foundation
=================================================================

This package is the foundational layer of a larger, future "AI Auditor"
system for computer-vision pipelines (SIH26228).

Phase 1 scope (this package):
    MODEL + IMAGE(S) -> INFERENCE -> STRUCTURED, SAVED RESULTS

Later phases (NOT implemented here) will consume the structured results
produced by this package to perform integrity/trust auditing such as
poisoning detection, backdoor detection, OOD detection, provenance
verification, and risk scoring.
"""

__version__ = "0.1.0"

from .config import PipelineConfig
from .model_loader import ModelLoader, ModelMetadata
from .image_loader import ImageLoader
from .inference import InferenceEngine
from .result_formatter import ResultFormatter, DetectionResult, ImageResult

__all__ = [
    "PipelineConfig",
    "ModelLoader",
    "ModelMetadata",
    "ImageLoader",
    "InferenceEngine",
    "ResultFormatter",
    "DetectionResult",
    "ImageResult",
]
