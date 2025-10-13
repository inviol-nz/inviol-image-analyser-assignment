from __future__ import annotations
import os
import io
from typing import List
import requests
from PIL import Image

from inviol_image_analyser_assignment.models import Detection, BoundingBox
from .base import DetectionBackend

class RoboflowVestBackend(DetectionBackend):
    """
    Calls Roboflow Hosted Inference for a 'vest' model.
    Expects env var ROBOFLOW_API_KEY.
    """

    def __init__(self, model_id: str, version: int | str = "1", conf: float = 0.25,
                 api_key_env: str = "ROBOFLOW_API_KEY"):
        # model_id examples:
        #   "vest-qf3av"                       (model in your default workspace)
        #   "aa-sutfb/vest-qf3av"              (explicit workspace/model)
        self.model_id = model_id.strip("/")  # <-- only strip leading/trailing slashes
        self.version = str(version)  # e.g., "1"
        self.conf = conf
        self.api_key_env = api_key_env
        self._url = f"https://detect.roboflow.com/{self.model_id}/{self.version}"

    def load(self) -> None:
        # nothing to load for HTTP
        return

    def predict(self, image_bytes: bytes) -> List[Detection]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Roboflow API key not set; export {self.api_key_env}=...")

        # Get image size for normalization
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size

        params = {
            "api_key": api_key,
            "confidence": int(self.conf * 100),  # Roboflow expects 0-100
            "format": "json",
        }
        files = {"file": ("image.jpg", image_bytes, "application/octet-stream")}
        r = requests.post(self._url, params=params, files=files, timeout=25)
        r.raise_for_status()
        data = r.json()

        preds = data.get("predictions", [])
        out: List[Detection] = []
        for p in preds:
            cls_name = p.get("class", "").lower()
            if cls_name not in {"vest"}:
                continue
            # Roboflow returns x,y,w,h in *pixels* (center-based) by default
            x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
            x1, y1 = x - w / 2.0, y - h / 2.0
            x2, y2 = x + w / 2.0, y + h / 2.0
            conf = float(p.get("confidence", 0.0))

            # normalize to 0..1
            box = BoundingBox(x1=max(0.0, x1 / W), y1=max(0.0, y1 / H),
                              x2=min(1.0, x2 / W), y2=min(1.0, y2 / H))
            out.append(Detection(label="vest", confidence=conf, box=box))
        return out
