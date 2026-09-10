"""
test_yolov5_support.py
-----------------------
Focused tests for the Phase 3 YOLOv5 integration:

- checkpoint loading (normal Ultralytics ``.pt`` format, via ModelLoader)
- CPU inference (and CUDA when available)
- normalized detection output (torchvision-style dicts + Phase 1
  DetectionResult schema, YOLO's own 80-class COCO label space)
- identical YOLOv5 reference/candidate through the FULL ModelAuditor ->
  no behavioral anomaly, category "clean"
- existing torchvision architectures untouched

Everything runs offline against the bundled yolov5n.pt checkpoint, the
vendored YOLOv5 implementation in third_party/yolov5, and the real test
photos that ship inside it (bus.jpg / zidane.jpg -- the synthetic Phase 1
sample images contain no COCO objects a real detector would report).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cv_auditor import ImageLoader, InferenceEngine, ModelLoader, PipelineConfig
from cv_auditor.config import SUPPORTED_ARCHITECTURES
from cv_auditor.model_audit import ModelAuditConfig, ModelAuditor
from cv_auditor.utils import sha256_of_file

ROOT = Path(__file__).resolve().parent.parent
YOLO_CKPT = ROOT / "data" / "sample_model" / "yolov5n.pt"
YOLO_REPO = ROOT / "third_party" / "yolov5"
REAL_IMAGES = YOLO_REPO / "data" / "images"  # bus.jpg, zidane.jpg
BUS_IMAGE = REAL_IMAGES / "bus.jpg"

pytestmark = pytest.mark.skipif(
    not YOLO_CKPT.exists() or not (YOLO_REPO / "models" / "yolo.py").exists(),
    reason="yolov5n.pt checkpoint or third_party/yolov5 implementation missing",
)


@pytest.fixture(scope="module")
def loaded_yolo():
    """Load the YOLOv5 model once for the whole module (CPU)."""
    cfg = PipelineConfig(
        model_path=str(YOLO_CKPT), architecture="yolov5",
        model_name="yolo_test_model", confidence_threshold=0.25, device="cpu",
    )
    model, metadata = ModelLoader(cfg).load()
    return model, metadata


@pytest.fixture(scope="module")
def bus_result(loaded_yolo):
    """One real CPU inference on bus.jpg, shared across tests."""
    model, _ = loaded_yolo
    image = ImageLoader(str(REAL_IMAGES)).load_one(str(BUS_IMAGE))
    engine = InferenceEngine(
        model, "yolo_test_model", model.class_names,
        torch.device("cpu"), confidence_threshold=0.25,
    )
    return image, engine.run_one(image)


class TestArchitectureRegistration:
    def test_yolov5_is_supported(self):
        assert "yolov5" in SUPPORTED_ARCHITECTURES

    def test_existing_architectures_untouched(self):
        for arch in (
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "retinanet_resnet50_fpn",
        ):
            assert arch in SUPPORTED_ARCHITECTURES

    def test_unknown_architecture_still_rejected(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            PipelineConfig(architecture="yolov9000")
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ModelAuditConfig(candidate_model_path=str(YOLO_CKPT),
                             architecture="yolov9000")

    def test_yolov5_requires_checkpoint_path(self):
        cfg = PipelineConfig(architecture="yolov5", model_path=None)
        with pytest.raises(ValueError, match="requires model_path"):
            ModelLoader(cfg).load()


class TestCheckpointLoading:
    def test_metadata_reflects_yolov5(self, loaded_yolo):
        model, metadata = loaded_yolo
        assert metadata.architecture == "yolov5"
        assert metadata.source == "yolov5_checkpoint"
        assert metadata.weights_sha256 == sha256_of_file(str(YOLO_CKPT))
        assert metadata.num_parameters > 1_000_000  # yolov5n ~1.87M
        assert metadata.framework == "pytorch"

    def test_coco_class_configuration_is_yolos_own(self, loaded_yolo):
        """80 classes, 0-based, no background slot -- NOT the 91-entry
        torchvision list."""
        model, metadata = loaded_yolo
        assert metadata.num_classes == 80
        assert len(model.class_names) == 80
        assert model.class_names[0] == "person"      # id 0 is a real class
        assert "__background__" not in model.class_names
        assert model.class_names[5] == "bus"

    def test_missing_checkpoint_file_raises(self, tmp_path):
        cfg = PipelineConfig(model_path=str(tmp_path / "nope.pt"),
                             architecture="yolov5")
        with pytest.raises(FileNotFoundError):
            ModelLoader(cfg).load()


class TestCpuInference:
    def test_real_objects_detected_on_cpu(self, bus_result):
        _, result = bus_result
        assert len(result.predictions) >= 2
        detected = {p.class_name for p in result.predictions}
        assert "person" in detected  # bus.jpg contains several people

    def test_normalized_detection_representation(self, bus_result):
        """Every prediction uses the SAME internal representation the
        rest of ModelAuditor consumes: class/class_id/confidence/bbox."""
        image, result = bus_result
        for pred in result.predictions:
            d = pred.to_dict()
            assert set(d.keys()) == {"class", "class_id", "confidence", "bbox"}
            assert 0 <= d["class_id"] < 80
            assert 0.0 < d["confidence"] <= 1.0
            x1, y1, x2, y2 = d["bbox"]
            assert 0.0 <= x1 <= x2 <= image.width
            assert 0.0 <= y1 <= y2 <= image.height

    def test_class_names_map_ids_correctly(self, loaded_yolo, bus_result):
        model, _ = loaded_yolo
        _, result = bus_result
        for pred in result.predictions:
            assert pred.class_name == model.class_names[pred.class_id]

    def test_adapter_output_follows_torchvision_api(self, loaded_yolo):
        model, _ = loaded_yolo
        image = ImageLoader(str(REAL_IMAGES)).load_one(str(BUS_IMAGE))
        out = model([image.tensor])
        assert isinstance(out, list) and len(out) == 1
        assert set(out[0].keys()) == {"boxes", "labels", "scores"}
        assert out[0]["boxes"].shape[1] == 4
        assert out[0]["labels"].dtype == torch.int64
        assert len(out[0]["boxes"]) == len(out[0]["labels"]) == len(out[0]["scores"])

    def test_detector_score_thresh_is_honored(self, tmp_path):
        """Lowering the internal confidence threshold must yield at least
        as many raw detections (mirrors box_score_thresh semantics)."""
        counts = {}
        for thresh in (0.25, 0.001):
            cfg = PipelineConfig(
                model_path=str(YOLO_CKPT), architecture="yolov5",
                confidence_threshold=0.0, detector_score_thresh=thresh,
            )
            model, _ = ModelLoader(cfg).load()
            image = ImageLoader(str(REAL_IMAGES)).load_one(str(BUS_IMAGE))
            counts[thresh] = len(model([image.tensor])[0]["scores"])
        assert counts[0.001] >= counts[0.25] >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaInference:
    def test_cuda_inference_matches_cpu_classes(self, bus_result):
        cfg = PipelineConfig(
            model_path=str(YOLO_CKPT), architecture="yolov5",
            model_name="yolo_cuda", confidence_threshold=0.25, device="cuda",
        )
        model, metadata = ModelLoader(cfg).load()
        assert metadata.device.startswith("cuda")
        assert next(model.parameters()).is_cuda

        image = ImageLoader(str(REAL_IMAGES)).load_one(str(BUS_IMAGE))
        engine = InferenceEngine(model, "yolo_cuda", model.class_names,
                                 torch.device("cuda"), confidence_threshold=0.25)
        cuda_result = engine.run_one(image)
        assert len(cuda_result.predictions) >= 2

        _, cpu_result = bus_result
        cpu_classes = {p.class_name for p in cpu_result.predictions}
        cuda_classes = {p.class_name for p in cuda_result.predictions}
        # Same model, same image: detected class sets should overlap heavily.
        assert "person" in cuda_classes
        assert cpu_classes & cuda_classes


class TestModelAuditorIntegration:
    def test_identical_yolo_reference_and_candidate_is_clean(self, tmp_path):
        """The required no-anomaly check: same yolov5n.pt as reference and
        candidate through the FULL Phase 3 pipeline (fingerprint +
        behavioral comparison + suspicion analysis)."""
        cfg = ModelAuditConfig(
            candidate_model_path=str(YOLO_CKPT),
            reference_model_path=str(YOLO_CKPT),
            reference_behavior_path=str(tmp_path / "reference_behavior.json"),
            architecture="yolov5",
            test_images_dir=str(REAL_IMAGES),
            confidence_threshold=0.25,
            output_path=str(tmp_path / "report.json"),
        )
        report = ModelAuditor(cfg).run()

        # SHA-256 fingerprinting: identical file -> unchanged.
        assert report.fingerprint["changed"] is False
        # Behavioral comparison: deterministic identical models agree fully.
        assert report.comparison is not None
        assert report.comparison["agreement_rate"] == 1.0
        assert report.comparison["total_class_flips"] == 0
        assert report.comparison["total_missing"] == 0
        assert report.comparison["total_added"] == 0
        # No anomaly of any kind.
        assert report.findings == []
        assert report.category == "clean"
        assert report.overall_finding_level == "LOW"

    def test_yolo_report_uses_yolo_label_space(self, tmp_path):
        """Per-image reference behavior must be recorded with YOLOv5's own
        class names (id 0 = person), not the Faster R-CNN list."""
        cfg = ModelAuditConfig(
            candidate_model_path=str(YOLO_CKPT),
            reference_model_path=str(YOLO_CKPT),
            reference_behavior_path=str(tmp_path / "reference_behavior.json"),
            architecture="yolov5",
            test_images_dir=str(REAL_IMAGES),
            confidence_threshold=0.25,
            output_path=str(tmp_path / "report.json"),
        )
        ModelAuditor(cfg).run()
        behavior = json.loads(Path(cfg.reference_behavior_path).read_text())
        classes = {
            p["class"]
            for img in behavior["image_results"]
            for p in img["predictions"]
        }
        assert "person" in classes
        assert "__background__" not in classes


class TestCli:
    def test_cli_yolov5_audit_end_to_end(self, tmp_path):
        """The exact CLI shape from the requirements (CPU here; --device
        cuda takes the same path and is covered by TestCudaInference)."""
        report_path = tmp_path / "report.json"
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "audit_model.py"),
                "--candidate", str(YOLO_CKPT),
                "--reference", str(YOLO_CKPT),
                "--architecture", "yolov5",
                "--test-images", str(REAL_IMAGES),
                "--device", "cpu",
                "--confidence-threshold", "0.25",
                "--reference-behavior", str(tmp_path / "reference_behavior.json"),
                "--output", str(report_path),
            ],
            capture_output=True, text=True, cwd=str(ROOT), timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(report_path.read_text())
        assert report["summary"]["category"] == "clean"
        assert report["fingerprint"]["changed"] is False
