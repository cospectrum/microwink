import pytest

from pathlib import Path
from PIL import Image

from microwink import SegModel
from microwink.common import Box, draw_box, draw_mask

from .utils import round_box


DATA_ROOT = Path("./assets/data/")
TRUTH_ROOT = Path("./tests/truth/seg_model/")
BIN_THRESHOLD = 0.5


@pytest.mark.parametrize(
    ["sample_filename", "boxes"],
    [
        ("us_card.png", [Box(x=27, y=94, h=298, w=420)]),
        ("us_card_rotated.png", [Box(x=96, y=252, h=419, w=297)]),
        ("mklovin.png", [Box(x=84, y=194, h=448, w=715)]),
        (
            "two_ids.png",
            [Box(x=159, y=33, h=246, w=400), Box(x=653, y=32, h=240, w=384)],
        ),
        (
            "three_ids.png",
            [
                Box(x=62, y=97, h=245, w=391),
                Box(x=751, y=92, h=245, w=393),
                Box(x=479, y=23, h=390, w=246),
            ],
        ),
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
        actual = draw_box(actual, card.box)
        actual = draw_mask(actual, card.mask > BIN_THRESHOLD)
    assert truth == actual
