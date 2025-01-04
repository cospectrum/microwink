import os
import cv2
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
        assert len(raw.boxes) == len(raw.masks)

        results = []
        for raw_bbox, raw_mask in zip(raw.boxes, raw.masks):
            x1, y1, x2, y2, score, class_id = raw_bbox
            assert class_id == CLASS_ID, class_id
            assert 0.0 <= score <= 1.0

            box = common.Box.from_xyxy([x1, y1, x2, y2])
            results.append(
                SegResult(
                    score=score,
                    mask=raw_mask,
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
        out = self.postprocess(
            preds,
            img_size=(ih, iw),
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=NM,
        )
        if out is None:
            return None
        boxes, masks = out
        N = len(boxes)
        assert boxes.shape == (N, 6)
        assert masks.shape == (N, ih, iw), masks.shape
        masks = common.sigmoid(masks)
        return RawResult(boxes=boxes, masks=masks)

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
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BORDER_COLOR
        )

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
    ) -> tuple[np.ndarray, np.ndarray] | None:
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

        x = x[x[..., 4 : 4 + NUM_CLASSES].max(axis=1) > conf_threshold]
        L = len(x)
        assert x.shape == (L, C)
        x = np.c_[
            x[:, :4],  # boxes
            x[:, 4 : 4 + NUM_CLASSES].max(axis=1),  # scores
            x[:, 4 : 4 + NUM_CLASSES].argmax(axis=1),  # class_ids
            x[:, 4 + NUM_CLASSES :],  # masks
        ]
        assert x.shape == (L, 1 + C)
        keep = self.nms(
            x[:, :4],
            x[:, 4],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        N = len(keep)
        x = x[keep]
        assert x.shape == (N, 1 + C)
        if N == 0:
            return None

        x[:, [0, 1]] -= x[:, [2, 3]] / 2
        x[:, [2, 3]] += x[:, [0, 1]]

        x[:, :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[:, :4] /= ratio

        ih, iw = img_size
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, iw)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, ih)

        boxes = x[:, :4]
        masks = self.process_mask(protos, x[:, 6:], boxes, (ih, iw))
        dets = x[:, :6]
        assert masks.shape == (N, ih, iw)
        assert dets.shape == (N, 6)
        return dets, masks

    @staticmethod
    def nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        *,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[int]:
        N = len(boxes)
        assert boxes.shape == (N, 4)
        assert scores.shape == (N,)
        keep = cv2.dnn.NMSBoxes(
            boxes,  # type: ignore
            scores,  # type: ignore
            conf_threshold,
            iou_threshold,
        )
        return list(keep)

    def process_mask(
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
        masks = self.scale_masks(np.ascontiguousarray(masks), (ih, iw))
        assert masks.shape == (N, ih, iw)
        return self.crop_masks(masks, boxes)

    @staticmethod
    def scale_masks(masks: np.ndarray, img_size: tuple[H, W]) -> np.ndarray:
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
    def crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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


def resize(buf: np.ndarray, size: tuple[W, H]) -> np.ndarray:
    w, h = size
    assert w > 0
    assert h > 0
    img = Image.fromarray(buf).resize(size, Resampling.LANCZOS)
    out = np.array(img).astype(buf.dtype)
    assert out.dtype == buf.dtype
    assert out.ndim == buf.ndim
    return out
