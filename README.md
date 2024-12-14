# microwink
Lightweight instance segmentation model for card IDs

## Usage

```python
from microwink import SegModel
from microwink.common import draw_mask
from PIL import Image

seg_model = SegModel.from_path("./models/seg_model.onnx")

img = Image.open("...").convert("RGB")
cards = seg_model.apply(img)

for card in cards:
    print(f"{card.score=}, {card.box=}")
    img = draw_mask(img, card.mask > 0.5)
img.save("result.png")
```
