# microwink
[![github]](https://github.com/cospectrum/microwink)
[![ci]](https://github.com/cospectrum/microwink/actions)

[github]: https://img.shields.io/badge/github-cospectrum/microwink-8da0cb?logo=github
[ci]: https://github.com/cospectrum/microwink/workflows/ci/badge.svg

Lightweight instance segmentation of card IDs.

<p>
  <img src="assets/data/us_card.png" width="49%">
  <img src="assets/us_card.result.png" width="49%">
</p>

## Usage

### Python
```sh
pip install microwink
```
```python
from microwink import SegModel
from microwink.common import draw_mask, draw_box
from PIL import Image

seg_model = SegModel.from_path("./models/seg_model.onnx")

img = Image.open("./assets/data/us_card.png").convert("RGB")
cards = seg_model.apply(img)

for card in cards:
    print(f"score={card.score}, box={card.box}")
    img = draw_box(img, card.box)
    img = draw_mask(img, card.mask > 0.5)
img.save("./output.png")
```

## License
Apache-2.0
