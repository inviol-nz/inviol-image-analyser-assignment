# src/inviol_image_analyser_assignment/services/detection/roboflow_vest.py
from __future__ import annotations
import io
import os
from typing import List
from PIL import Image
from inference_sdk import InferenceHTTPClient

from inviol_image_analyser_assignment.models import Detection, BoundingBox
from .base import DetectionBackend

class RoboflowVestBackend(DetectionBackend):
    def __init__(self, model_id: str, version: int | str = "1", conf: float = 0.25,
                 api_key_env: str = "ROBOFLOW_API_KEY"):
        self.model_id = model_id.strip("/")            # e.g. "vest-qf3av" or "aa-sutfb/vest-qf3av"
        self.version = str(version)                    # e.g. "1"
        self.conf = conf
        self.api_key_env = api_key_env
        self._client = None

    def load(self) -> None:
        if self._client is not None:
            return
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Roboflow API key not set; export {self.api_key_env}=...")
        self._client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )

    def predict(self, image_bytes: bytes) -> List[Detection]:
        if self._client is None:
            self.load()

        # The SDK expects a path or a file-like; write to a tiny in-memory file
        # or to a NamedTemporaryFile. Easiest: pass a BytesIO-like path via temp file.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            # ensure RGB JPEG for consistency
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(tmp.name, format="JPEG", quality=90)

            result = self._client.infer(
                tmp.name,
                model_id=f"{self.model_id}/{self.version}",
                # SDK supports confidence as 0–1 or 0–100 depending on version—force 0–1:
                confidence=self.conf,
            )

        preds = result.get("predictions", [])
        W = result.get("image", {}).get("width", img.width)
        H = result.get("image", {}).get("height", img.height)

        out: List[Detection] = []
        for p in preds:
            if str(p.get("class", "")).lower() != "vest":
                continue
            x, y = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            x1, y1 = x - w/2.0, y - h/2.0
            x2, y2 = x + w/2.0, y + h/2.0
            conf = float(p.get("confidence", 0.0))
            out.append(Detection(
                label="vest",
                confidence=conf,
                box=BoundingBox(
                    x1=max(0.0, x1 / W), y1=max(0.0, y1 / H),
                    x2=min(1.0, x2 / W), y2=min(1.0, y2 / H)
                )
            ))
        return out
