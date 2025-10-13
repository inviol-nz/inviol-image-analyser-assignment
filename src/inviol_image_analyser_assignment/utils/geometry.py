from __future__ import annotations
from typing import Tuple
from inviol_image_analyser_assignment.models import BoundingBox

def bbox_to_pixels(box: BoundingBox, W: int, H: int) -> Tuple[int,int,int,int]:
    return (int(box.x1 * W), int(box.y1 * H), int(box.x2 * W), int(box.y2 * H))

def bottom_center(box: BoundingBox, W: int, H: int, offset: float = 0.0) -> Tuple[int,int]:
    x1, y1, x2, y2 = bbox_to_pixels(box, W, H)
    cx = (x1 + x2) // 2
    by = y2 - int(offset * H)
    return cx, max(0, min(H-1, by))

def torso_crop(box: BoundingBox, W: int, H: int, y1f: float, y2f: float) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bbox_to_pixels(box, W, H)
    h = y2 - y1
    ty1 = y1 + int(h * y1f)
    ty2 = y1 + int(h * y2f)
    return x1, ty1, x2, ty2
