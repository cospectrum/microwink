from microwink.common import Box as Box


def round_box(box: Box) -> Box:
    return Box(x=int(box.x), y=int(box.y), w=int(box.w), h=int(box.h))
