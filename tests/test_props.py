from microwink import SegModel, Threshold
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from PIL.Image import Image as PILImage

from .utils.proptest import arbitrary_rgb_image as arb_img


@settings(
    max_examples=200,
)
@given(img=arb_img((1, 2000), (1, 2000)))
def test_preprocess(img: PILImage, seg_model: SegModel) -> None:
    B = 1
    CH = 3
    H = seg_model.model_height
    W = seg_model.model_width

    blob, *_ = seg_model.preprocess(img)
    assert blob.shape == (B, CH, H, W)
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


@settings(
    deadline=2 * 1000,
)
@given(
    img=arb_img((1, 2000), (1, 2000)),
    iou=st.none() | st.floats(0.01, 1.0),
    score=st.none() | st.floats(0.01, 1.0),
)
def test_apply(
    img: PILImage,
    iou: float | None,
    score: float | None,
    seg_model: SegModel,
) -> None:
    default = Threshold.default()
    threshold = Threshold(
        confidence=score or default.confidence,
        iou=iou or default.iou,
    )

    objects = seg_model.apply(img, threshold=threshold)
    for obj in objects:
        assert obj.score >= threshold.confidence
        assert obj.mask.min() >= 0.0
        assert obj.mask.max() <= 1.0
