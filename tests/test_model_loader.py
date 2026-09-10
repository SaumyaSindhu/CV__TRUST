import pytest
import torch

from cv_auditor.config import PipelineConfig
from cv_auditor.model_loader import ModelLoader, ModelMetadata

from .fixtures import SAMPLE_MODEL_PATH, SAMPLE_ARCHITECTURE


def test_model_loads_successfully_from_checkpoint():
    config = PipelineConfig(
        model_path=str(SAMPLE_MODEL_PATH),
        architecture=SAMPLE_ARCHITECTURE,
        pretrained=False,
    )
    model, metadata = ModelLoader(config).load()

    assert isinstance(metadata, ModelMetadata)
    assert model is not None
    assert metadata.source == "checkpoint"
    assert metadata.num_parameters > 0
    assert metadata.weights_sha256 is not None
    # Model should be in eval mode after loading.
    assert model.training is False


def test_model_loads_with_random_init_when_no_checkpoint_given():
    config = PipelineConfig(
        model_path=None,
        architecture=SAMPLE_ARCHITECTURE,
        pretrained=False,
    )
    model, metadata = ModelLoader(config).load()
    assert metadata.source == "random_init"
    assert model is not None


def test_missing_model_file_raises_clear_error():
    config = PipelineConfig(
        model_path="data/sample_model/does_not_exist.pth",
        architecture=SAMPLE_ARCHITECTURE,
    )
    with pytest.raises(FileNotFoundError):
        ModelLoader(config).load()


def test_unsupported_architecture_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(architecture="not_a_real_architecture")


def test_cuda_request_falls_back_to_cpu_when_unavailable():
    config = PipelineConfig(
        model_path=str(SAMPLE_MODEL_PATH),
        architecture=SAMPLE_ARCHITECTURE,
        device="cuda",
    )
    model, metadata = ModelLoader(config).load()
    if not torch.cuda.is_available():
        assert metadata.device == "cpu"
