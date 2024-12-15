import numpy as np

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage


@dataclass
class Box:
    x: float
    y: float
    h: float
    w: float

    @staticmethod
    def from_xyxy(box: Iterable[Any]) -> "Box":
        x1, y1, x2, y2 = [float(t) for t in box]
        h = y2 - y1
        w = x2 - x1
        assert h >= 0.0
        assert w >= 0.0
        return Box(x=x1, y=y1, w=w, h=h)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def draw_box(
    image: PILImage,
    box: Box,
    *,
    color: tuple[int, ...] | str | float = (255, 0, 0),
    width: int = 3,
) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    points = [(box.x, box.y), (box.x + box.w, box.y + box.h)]
    draw.rectangle(points, outline=color, width=width)
    return image


def draw_mask(
    image: PILImage,
    binary_mask: np.ndarray,
    *,
    color: Sequence[Any] = (0, 255, 0),
    alpha: float = 0.5,
) -> PILImage:
    assert 0.0 <= alpha <= 1.0
    assert (image.height, image.width) == binary_mask.shape
    img = np.array(image)
    assert len(img.shape) == len(color)
    overlay = np.zeros_like(img)
    overlay[binary_mask] = color
    assert overlay.shape == img.shape

    img[binary_mask] = (1.0 - alpha) * img[binary_mask] + alpha * overlay[binary_mask]
    return Image.fromarray(img)
