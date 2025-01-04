from microwink import SegModel, Threshold
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from PIL.Image import Image as PILImage

from .utils.proptest import arbitrary_rgb_image as arb_img


@settings(
    deadline=1 * 1000,
    max_examples=50,
)
@given(
    img=arb_img((1, 1000), (1, 1000)),
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
