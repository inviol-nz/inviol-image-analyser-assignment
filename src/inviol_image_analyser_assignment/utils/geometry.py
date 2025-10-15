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

def torso_region_norm(box: BoundingBox, y1f: float = 0.35, y2f: float = 0.75, cx_frac: float = 0.60) -> BoundingBox:
    # returns a normalized sub-box inside `box`
    h = box.y2 - box.y1
    w = box.x2 - box.x1
    ty1 = box.y1 + h * y1f
    ty2 = box.y1 + h * y2f
    cx = (box.x1 + box.x2) / 2.0
    half = (w * cx_frac) / 2.0
    tx1 = max(0.0, cx - half)
    tx2 = min(1.0, cx + half)
    return BoundingBox(x1=tx1, y1=ty1, x2=tx2, y2=ty2)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-9
    return inter / union

def frac_inside(inner: BoundingBox, outer: BoundingBox) -> float:
    # fraction of INNER area that lies inside OUTER
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_inner = max(1e-9, (inner.x2 - inner.x1) * (inner.y2 - inner.y1))
    return inter / area_inner

