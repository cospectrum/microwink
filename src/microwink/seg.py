import os
import onnxruntime as ort  # type: ignore # missing stubs
import numpy as np

from typing import Sequence
from dataclasses import dataclass
from PIL import Image
from PIL.Image import Image as PILImage

from . import common


@dataclass
class Threshold:
    confidence: float = 0.6
    iou: float = 0.5

    @staticmethod
    def default() -> "Threshold":
        return Threshold()


@dataclass
class SegResult:
    box: common.Box
    score: float
    mask: np.ndarray  # heat map with values from 0.0 to 1.0


@dataclass
class Result:
    boxes: np.ndarray
    scores: np.ndarray
    mask_maps: np.ndarray


H = int
W = int


@dataclass
class InputShape:
    batch: int
    ch: int
    h: H
    w: W


@dataclass
class SegModel:
    """SegModel is thread-safe if and only if the `session` is thread-safe"""

    session: ort.InferenceSession
    input_names: list[str]
    output_names: list[str]
    input_shape: InputShape

    @staticmethod
    def from_path(
        path: str | os.PathLike, *, providers: Sequence[str] | None = None
    ) -> "SegModel":
        return SegModel.from_session(ort.InferenceSession(path, providers=providers))

    @staticmethod
    def from_session(session: ort.InferenceSession) -> "SegModel":
        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]
        batch, ch, h, w = inputs[0].shape
        input_shape = InputShape(batch=batch, ch=ch, h=h, w=w)

        outputs = session.get_outputs()
        output_names = [out.name for out in outputs]
        return SegModel(
            session=session,
            input_shape=input_shape,
            input_names=input_names,
            output_names=output_names,
        )

    def apply(
        self,
        image: PILImage,
        threshold: Threshold = Threshold.default(),
    ) -> list[SegResult]:
        assert image.mode == "RGB"
        tensor = self.preprocess(image)
        outs = self.forward(tensor)
        img_size = (image.height, image.width)
        result = self.postprocess(outs, img_size, threshold)
        if result is None:
            return []

        out = []
        assert len(result.boxes) == len(result.scores) == len(result.mask_maps)
        for box, score, mask in zip(result.boxes, result.scores, result.mask_maps):
            assert len(mask.shape) == 2
            assert mask.dtype == np.float64
            out.append(
                SegResult(
                    box=common.Box.from_xyxy(box),
                    score=float(score),
                    mask=mask,
                )
            )
        return out

    def postprocess(
        self,
        outs: list[np.ndarray],
        img_size: tuple[H, W],
        threshold: Threshold,
    ) -> Result | None:
        NUM_MASKS = 32
        box_out, mask_out = outs

        preds = np.squeeze(box_out).T
        num_classes = box_out.shape[1] - NUM_MASKS - 4
        assert num_classes >= 0
        split_at = 4 + num_classes

        scores = np.max(preds[:, 4:split_at], axis=1)
        preds = preds[scores > threshold.confidence, :]
        scores = scores[scores > threshold.confidence]
        if len(scores) == 0:
            return None

        box_preds = preds[..., :split_at]
        mask_preds = preds[..., split_at:]

        boxes = self.extract_boxes(box_preds, img_size)
        indexes = nms(boxes, scores, threshold.iou)
        if len(indexes) == 0:
            return None
        final_boxes = boxes[indexes]
        final_scores = scores[indexes]
        final_mask_maps = self.postprocess_mask(
            mask_preds[indexes],
            mask_out,
            final_boxes,
            img_size,
        )
        assert len(final_boxes) == len(final_scores) == len(final_mask_maps)
        return Result(
            boxes=final_boxes,
            scores=final_scores,
            mask_maps=final_mask_maps,
        )

    def postprocess_mask(
        self,
        mask_preds: np.ndarray,
        mask_out: np.ndarray,
        boxes: np.ndarray,
        img_size: tuple[H, W],
    ) -> np.ndarray:
        assert len(mask_preds) > 0
        mask_out = np.squeeze(mask_out)
        num_mask, mask_height, mask_width = mask_out.shape
        masks = mask_preds @ mask_out.reshape((num_mask, -1))
        masks = masks.reshape((-1, mask_height, mask_width))

        ih, iw = img_size
        scaled_boxes = self.rescale_boxes(
            boxes,
            (ih, iw),
            (mask_height, mask_width),
        )
        mask_maps = np.zeros(
            (
                len(scaled_boxes),
                ih,
                iw,
            )
        )
        assert len(scaled_boxes) == len(masks)
        assert len(scaled_boxes) == len(boxes)
        for i, (box, scaled_box, mask) in enumerate(zip(boxes, scaled_boxes, masks)):
            assert 2 == len(mask.shape)

            scale_x1 = int(scaled_box[0])
            scale_y1 = int(scaled_box[1])
            scale_x2 = 1 + int(scaled_box[2])
            scale_y2 = 1 + int(scaled_box[3])

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = 1 + int(box[2])
            y2 = 1 + int(box[3])

            ow, oh = (x2 - x1, y2 - y1)
            assert ow >= 0
            assert oh >= 0
            resized_mask = resize(
                mask[scale_y1:scale_y2, scale_x1:scale_x2],
                (ow, oh),
            )
            assert resized_mask.shape == (oh, ow)
            mask_maps[i, y1:y2, x1:x2] = common.sigmoid(resized_mask).clip(0.0, 1.0)

        return mask_maps

    def forward(self, tensor: np.ndarray) -> list[np.ndarray]:
        outs = self.session.run(
            self.output_names,
            {self.input_names[0]: tensor},
        )
        return outs

    def preprocess(self, image: PILImage) -> np.ndarray:
        size = (self.input_shape.w, self.input_shape.h)
        if image.size != size:
            image = image.resize(size)
        img = np.array(image).astype(np.float32)
        assert len(img.shape) == 3
        img /= 255.0
        img = img.transpose(2, 0, 1)
        tensor = img[np.newaxis, :, :, :]
        return tensor

    @staticmethod
    def rescale_boxes(
        boxes: np.ndarray,
        div_shape: tuple[H, W],
        mul_shape: tuple[H, W],
    ) -> np.ndarray:
        shape = np.array(
            [
                div_shape[1],
                div_shape[0],
                div_shape[1],
                div_shape[0],
            ]
        )
        boxes = np.divide(boxes, shape, dtype=np.float32)
        boxes *= np.array(
            [
                mul_shape[1],
                mul_shape[0],
                mul_shape[1],
                mul_shape[0],
            ]
        )
        return boxes

    def extract_boxes(self, box_preds: np.ndarray, img_size: tuple[H, W]) -> np.ndarray:
        h, w = img_size
        boxes = box_preds[:, :4]
        boxes = self.rescale_boxes(
            boxes,
            (self.input_shape.h, self.input_shape.w),
            (h, w),
        )
        boxes = xywh2xyxy(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        assert len(boxes) == len(box_preds)
        return boxes


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.copy(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = int(sorted_indices[0])
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

        keep_boxes.append(box_id)
    return keep_boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area
    return iou


def resize(buf: np.ndarray, size: tuple[W, H]) -> np.ndarray:
    img = Image.fromarray(buf).resize(size)
    out = np.array(img)
    assert out.dtype == buf.dtype
    assert len(out.shape) == len(buf.shape)
    return out
