"""Local YOLOv8 backend used for base detection (people, vehicles, etc.).
"""

from __future__ import annotations
import io
from typing import List
import numpy as np
from PIL import Image

from inviol_image_analyser_assignment.models import Detection, BoundingBox
from .base import DetectionBackend

# Classes we use  for now
_ALLOWED = {"person", "car", "truck", "bus"}


def _norm_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> BoundingBox:
    """Clamp and normalize absolute XYXY to [0,1] BoundingBox."""
    x1 = max(0.0, min(float(x1), W))
    y1 = max(0.0, min(float(y1), H))
    x2 = max(0.0, min(float(x2), W))
    y2 = max(0.0, min(float(y2), H))
    return BoundingBox(x1=x1 / W, y1=y1 / H, x2=x2 / W, y2=y2 / H)


class YOLOv8Backend(DetectionBackend):
    """Thin wrapper around `ultralytics.YOLO` with lazy loading."""

    _model = None
    _names = None

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640):
        """
        Args:
            model_name: Weight file to load (downloaded on first run if missing).
            conf: Confidence threshold used by YOLO.
            imgsz: Inference image size.
        """
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz

    def load(self) -> None:
        """Load YOLO model once (idempotent)."""
        if YOLOv8Backend._model is not None:
            return
        try:
            # Import only inside container where `ultralytics` is installed.
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics not available. Build/run in Docker or install 'ultralytics' in your env."
            ) from exc

        model = YOLO(self.model_name)  # downloads weights on first run
        YOLOv8Backend._model = model
        YOLOv8Backend._names = model.names  # id -> label mapping

    def predict(self, image_bytes: bytes) -> List[Detection]:
        """Run YOLO on the provided image and return filtered detections."""
        if YOLOv8Backend._model is None:
            self.load()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size

        res = YOLOv8Backend._model.predict(
            source=np.array(img),  # RGB ndarray
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        dets: List[Detection] = []
        if res.boxes is None:
            return dets

        for b in res.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten()  # [x1, y1, x2, y2]
            score = float(b.conf.cpu().numpy().item())
            cls_id = int(b.cls.cpu().numpy().item())
            label = YOLOv8Backend._names.get(cls_id, str(cls_id))
            if label not in _ALLOWED:
                continue
            dets.append(
                Detection(
                    label=label,
                    confidence=score,
                    box=_norm_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H),
                )
            )
        return dets
