import pytest
import torch

from cv_auditor.image_loader import ImageLoader, LoadedImage

from .fixtures import SAMPLE_IMAGES_DIR


def test_image_directory_discovers_all_images():
    loader = ImageLoader(str(SAMPLE_IMAGES_DIR))
    paths = loader.discover()
    assert len(paths) >= 1
    assert all(p.lower().endswith((".jpg", ".jpeg", ".png")) for p in paths)


def test_single_image_loads_successfully():
    loader = ImageLoader(str(SAMPLE_IMAGES_DIR))
    first_path = loader.discover()[0]
    image = loader.load_one(first_path)

    assert isinstance(image, LoadedImage)
    assert isinstance(image.tensor, torch.Tensor)
    assert image.tensor.shape[0] == 3  # RGB channels
    assert image.width > 0 and image.height > 0
    assert len(image.sha256) == 64  # sha256 hex digest length


def test_load_all_returns_one_result_per_image():
    loader = ImageLoader(str(SAMPLE_IMAGES_DIR))
    images = loader.load_all()
    assert len(images) == len(loader.discover())


def test_missing_image_path_raises_error():
    loader = ImageLoader("data/sample_images/does_not_exist.jpg")
    with pytest.raises(FileNotFoundError):
        loader.load_one("data/sample_images/does_not_exist.jpg")


def test_nonexistent_directory_raises_error():
    with pytest.raises(FileNotFoundError):
        ImageLoader("data/no_such_directory").discover()
