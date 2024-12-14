# microwink
Lightweight instance segmentation for card IDs

## Install
```sh
pip install microwink
```

## Usage
```python
from microwink import SegModel
from microwink.common import draw_mask, draw_box
from PIL import Image

seg_model = SegModel.from_path("./models/seg_model.onnx")

img = Image.open("./tests/data/us_card.png").convert("RGB")
cards = seg_model.apply(img)

for card in cards:
    print(f"score={card.score}, box={card.box}")
    img = draw_mask(img, card.mask > 0.5)
    img = draw_box(img, card.box)

img.save("./assets/result.png")
```

<p align="middle">
  <img src="./assets/result.png">
</p>
