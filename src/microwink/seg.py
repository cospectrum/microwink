import os
import numpy as np
import onnxruntime as ort  # type: ignore

from numpy.typing import DTypeLike
from typing import Any, NewType
from dataclasses import dataclass
from PIL import Image
from PIL.Image import Image as PILImage, Resampling

from . import common


Dtype = DTypeLike
Color = tuple[int, int, int]

H = NewType("H", int)
W = NewType("W", int)
RgbBuf = NewType("RgbBuf", np.ndarray)


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
class RawResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: np.ndarray


class SegModel:
    session: ort.InferenceSession
    dtype: Dtype
    model_width: int
    model_height: int
    input_: Any

    @staticmethod
    def from_path(
        path: str | os.PathLike, providers: list[str] | None = None
    ) -> "SegModel":
        session = ort.InferenceSession(
            path,
            providers=providers or ["CPUExecutionProvider"],
        )
        return SegModel.from_session(session)

    @staticmethod
    def from_session(session: ort.InferenceSession) -> "SegModel":
        return SegModel(session)

    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        inputs = self.session.get_inputs()
        assert len(inputs) == 1, len(inputs)
        self.input_ = inputs[0]
        if self.input_.type == "tensor(float16)":
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        B, _, H, W = self.input_.shape
        assert B == 1, "batching is not supported"
        self.model_height = H
        self.model_width = W

    def apply(
        self, image: PILImage, threshold: Threshold = Threshold()
    ) -> list[SegResult]:
        CLASS_ID = 0
        assert image.mode == "RGB"
        buf = RgbBuf(np.array(image))
        raw = self._run(buf, threshold.confidence, threshold.iou)
        if raw is None:
            return []

        results = []
        it = zip(raw.scores, raw.boxes, raw.masks, raw.class_ids)
        for score, raw_box, mask, class_id in it:
            assert class_id == CLASS_ID, class_id
            assert 0.0 <= score <= 1.0
            box = common.Box.from_xyxy(raw_box)
            results.append(
                SegResult(
                    score=score,
                    mask=mask,
                    box=box,
                )
            )
        return results

    def _run(
        self, img: RgbBuf, conf_threshold: float, iou_threshold: float
    ) -> RawResult | None:
        NM = 32
        ih, iw, _ = img.shape

        blob, ratio, (pad_w, pad_h) = self.preprocess(img)
        assert blob.ndim == 4
        preds = self.session.run(None, {self.input_.name: blob})
        return self.postprocess(
            preds,
            img_size=(ih, iw),
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=NM,
        )

    def preprocess(
        self, img_buf: RgbBuf
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        BORDER_COLOR = (114, 114, 114)
        EPS = 0.1
        img = np.array(img_buf)
        ih, iw, _ = img.shape
        oh, ow = self.model_height, self.model_width
        r = min(oh / ih, ow / iw)
        rw, rh = round(iw * r), round(ih * r)

        pad_w, pad_h = [
            (ow - rw) / 2,
            (oh - rh) / 2,
        ]
        if (iw, ih) != (rw, rh):
            img = resize(img, (W(rw), H(rh)))
        top, bottom = round(pad_h - EPS), round(pad_h + EPS)
        left, right = round(pad_w - EPS), round(pad_w + EPS)
        img = self.with_border(img, top, bottom, left, right, BORDER_COLOR)
        assert img.ndim == 3
        blob = (1 / 255.0) * np.ascontiguousarray(
            np.einsum("HWC->CHW", img),  # type: ignore
            dtype=self.dtype,
        )
        assert blob.ndim == 3
        blob = blob[None]
        return blob, r, (pad_w, pad_h)

    def postprocess(
        self,
        preds: list[np.ndarray],
        img_size: tuple[H, W],
        ratio: float,
        pad_w: float,
        pad_h: float,
        conf_threshold: float,
        iou_threshold: float,
        nm: int,
    ) -> RawResult | None:
        B = 1
        NM, MH, MW = (nm, 160, 160)
        NUM_CLASSES = 1
        C = 4 + NUM_CLASSES + NM

        x, protos = preds
        assert len(x) == len(protos) == B
        protos = protos[0]
        x = x[0].T
        assert protos.shape == (NM, MH, MW), protos.shape
        assert x.shape == (len(x), C)

        likely = x[:, 4 : 4 + NUM_CLASSES].max(axis=1) > conf_threshold
        x = x[likely]

        scores = x[:, 4 : 4 + NUM_CLASSES].max(axis=1)
        boxes = x[:, :4]
        boxes = self.postprocess_boxes(boxes, img_size, ratio, pad_w=pad_w, pad_h=pad_h)
        keep = self.nms(
            boxes,
            scores,
            iou_threshold=iou_threshold,
        )
        N = len(keep)
        if N == 0:
            return None
        class_ids = x[:, 4 : 4 + NUM_CLASSES].argmax(axis=1)
        masks_in = x[:, 4 + NUM_CLASSES :]

        scores = scores[keep]
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        masks_in = masks_in[keep]

        ih, iw = img_size
        masks = self.postprocess_masks(protos, masks_in, boxes, (ih, iw))

        assert masks.shape == (N, ih, iw)
        assert boxes.shape == (N, 4)
        assert scores.shape == (N,)
        assert class_ids.shape == (N,)
        return RawResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            masks=masks,
        )

    def postprocess_boxes(
        self,
        boxes: np.ndarray,
        img_size: tuple[H, W],
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> np.ndarray:
        boxes = boxes.copy()
        boxes[:, [0, 1]] -= boxes[:, [2, 3]] / 2
        boxes[:, [2, 3]] += boxes[:, [0, 1]]

        boxes -= [pad_w, pad_h, pad_w, pad_h]
        boxes /= ratio

        ih, iw = img_size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih)
        return boxes

    def postprocess_masks(
        self,
        protos: np.ndarray,
        masks_in: np.ndarray,
        boxes: np.ndarray,
        img_size: tuple[H, W],
    ) -> np.ndarray:
        N = len(masks_in)
        nm, mh, mw = protos.shape
        assert boxes.shape == (N, 4)
        assert masks_in.shape == (N, nm)

        masks = np.matmul(masks_in, protos.reshape((nm, -1))).reshape((N, mh, mw))
        ih, iw = img_size
        masks = self._scale_masks(np.ascontiguousarray(masks), (ih, iw))
        assert masks.shape == (N, ih, iw)
        return common.sigmoid(self._crop_masks(masks, boxes))

    @staticmethod
    def _scale_masks(masks: np.ndarray, img_size: tuple[H, W]) -> np.ndarray:
        EPS = 0.1
        ih, iw = img_size
        N, mh, mw = masks.shape

        gain = min(mh / ih, mw / iw)
        pad_w = (mw - iw * gain) / 2
        pad_h = (mh - ih * gain) / 2

        top = round(pad_h - EPS)
        bottom = round(mh - pad_h + EPS)

        left = round(pad_w - EPS)
        right = round(mw - pad_w + EPS)

        masks = masks[:, top:bottom, left:right]
        masks_out = np.zeros((N, ih, iw))
        for i, mask in enumerate(masks):
            resized_mask = resize(mask, (iw, ih))
            assert resized_mask.shape == (ih, iw)
            masks_out[i] = resized_mask
        return masks_out

    @staticmethod
    def _crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        N, mh, mw = masks.shape
        assert boxes.shape == (N, 4)
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(mw, dtype=x1.dtype)[None, None, :]
        c = np.arange(mh, dtype=x1.dtype)[None, :, None]
        assert r.shape == (1, 1, mw)
        assert c.shape == (1, mh, 1)
        masks_out = masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
        assert masks_out.shape == (N, mh, mw)
        return masks_out

    @staticmethod
    def nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        *,
        iou_threshold: float,
    ) -> list[int]:
        sorted_indices = np.argsort(scores)[::-1]
        N = len(boxes)
        assert boxes.shape == (N, 4)
        assert scores.shape == (N,)
        assert sorted_indices.shape == (N,)

        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = int(sorted_indices[0])
            ious = SegModel._compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

            keep_boxes.append(box_id)
        return keep_boxes

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        assert box.shape == (4,)
        assert boxes.shape == (len(boxes), 4)
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

    @staticmethod
    def with_border(
        img: np.ndarray,
        top: int,
        bottom: int,
        left: int,
        right: int,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        import cv2

        assert img.ndim == 3
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )


def resize(buf: np.ndarray, size: tuple[W, H]) -> np.ndarray:
    w, h = size
    assert w > 0
    assert h > 0
    img = Image.fromarray(buf).resize(size, Resampling.LANCZOS)
    out = np.array(img).astype(buf.dtype)
    assert out.dtype == buf.dtype
    assert out.ndim == buf.ndim
    return out
