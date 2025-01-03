import pytest

from pathlib import Path
from microwink import SegModel


@pytest.fixture(scope="package")
def seg_model() -> SegModel:
    path = Path("./models/seg_model.onnx")
    assert path.exists()
    return SegModel.from_path(path)
