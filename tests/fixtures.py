"""Shared helpers/fixtures for the test suite."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MODEL_PATH = ROOT / "data" / "sample_model" / "sample_model.pth"
SAMPLE_IMAGES_DIR = ROOT / "data" / "sample_images"
SAMPLE_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
