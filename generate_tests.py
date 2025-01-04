from microwink import SegModel
from microwink.common import draw_mask, draw_box

from PIL import Image
from pathlib import Path
from tests.utils import round_box

MODEL_PATH = "./models/seg_model.onnx"
SAVE_TO = Path("./generated")


def main() -> None:
    SAVE_TO.mkdir(exist_ok=True)
    seg_model = SegModel.from_path(MODEL_PATH)

    for img_path in Path("./assets/data/").iterdir():
        img = Image.open(img_path).convert("RGB")
        cards = seg_model.apply(img)

        print(img_path.name)
        for card in cards:
            img = draw_box(img, card.box)
            img = draw_mask(img, card.mask > 0.5)
            print(round_box(card.box))
        print()
        img.save(SAVE_TO / img_path.name)


if __name__ == "__main__":
    main()
