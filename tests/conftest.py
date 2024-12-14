import pytest
import os

from pathlib import Path
from microwink import SegModel

KiB = 2**10
MiB = (2**10) * KiB


@pytest.fixture
def seg_model() -> SegModel:
    path = Path("./models/seg_model.onnx")
    assert path.exists()
    size = os.path.getsize(path)
    assert 10 * MiB < size < 15 * MiB
    return SegModel.from_path(path)
