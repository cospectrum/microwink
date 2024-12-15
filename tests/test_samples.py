import pytest

from pathlib import Path
from PIL import Image

from microwink import SegModel
from microwink.common import Box, draw_mask


DATA_ROOT = Path("./assets/data/")
TRUTH_ROOT = Path("./tests/truth/seg_model/")
BIN_THRESHOLD = 0.5


@pytest.mark.parametrize(
    ["sample_filename", "boxes"],
    [
        ("us_card.png", [Box(x=27, y=95, h=299, w=420)]),
        ("mklovin.png", [Box(x=84, y=194, h=448, w=715)]),
    ],
)
def test_samples(
    seg_model: SegModel,
    sample_filename: str,
    boxes: list[Box],
) -> None:
    img_path = DATA_ROOT / sample_filename
    truth_path = TRUTH_ROOT / sample_filename
    img = Image.open(img_path).convert("RGB")
    truth = Image.open(truth_path).convert("RGB")

    cards = seg_model.apply(img)
    assert len(cards) == len(boxes)
    actual = img.copy()
    for card, box in zip(cards, boxes):
        assert round_box(card.box) == box
        binary_mask = card.mask > BIN_THRESHOLD
        actual = draw_mask(actual, binary_mask)
    assert truth == actual


def round_box(box: Box) -> Box:
    return Box(x=int(box.x), y=int(box.y), w=int(box.w), h=int(box.h))
