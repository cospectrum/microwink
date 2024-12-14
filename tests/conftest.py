import pytest
from microwink import SegModel


@pytest.fixture
def seg_model() -> SegModel:
    return SegModel.from_path("./models/seg_model.onnx")
