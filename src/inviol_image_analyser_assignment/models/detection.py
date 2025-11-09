from __future__ import annotations

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box in image pixel coordinates.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    """
    A single object detected in the image.
    """

    id: str
    label: str
    confidence: float
    bbox: BoundingBox
