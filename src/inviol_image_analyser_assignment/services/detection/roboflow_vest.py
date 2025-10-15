"""
Roboflow-hosted detector for safety vests.

"""

from __future__ import annotations
import os
import io
import base64
from typing import List, Tuple
import requests
from PIL import Image

from inviol_image_analyser_assignment.models import Detection, BoundingBox
from .base import DetectionBackend


def _prepare_for_roboflow(
    image_bytes: bytes,
    max_side: int = 1024,
    quality: int = 85,
) -> tuple[str, int, int]:
    """Resize + JPEG-encode the image and return (base64_jpeg, width, height).
    Args:
        image_bytes: Raw bytes of the source image.
        max_side: Max length of the longest image side after resize.
        quality: JPEG quality (trade-off between size and fidelity).

    Returns:
        (b64, W, H): base64 string (no newlines), and the resized image size.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        new_w, new_h = int(round(w / scale)), int(round(h / scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h

    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, w, h


class RoboflowVestBackend(DetectionBackend):
    """Calls Roboflow to detect `vest` objects and normalizes results."""

    def __init__(
        self,
        model_id: str,
        version: int | str | None = None,
        api_key_env: str = "ROBOFLOW_API_KEY",
    ):
        """Create the backend.

        Args:
            model_id: Either "project/version" or "project" (then provide `version`).
            version: Version number if not included in `model_id`.
            api_key_env: Environment variable from which to read the RF API key.
        """
        model_id = model_id.strip("/")
        self.model_full = model_id if "/" in model_id else f"{model_id}/{version}"
        self.api_key_env = api_key_env
        self.base = "https://detect.roboflow.com"
        self.api_key: str | None = None

    def load(self) -> None:
        """Read and validate the API key once."""
        if self.api_key is not None:
            return
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Roboflow API key not set; export {self.api_key_env}=...")
        if not api_key.startswith("rf_"):
            raise RuntimeError(
                f"{self.api_key_env} looks invalid (missing 'rf_' prefix). Got: {api_key[:6]}…"
            )
        self.api_key = api_key

    def predict(self, image_bytes: bytes) -> List[Detection]:
        """Detect vests using Roboflow Hosted API and return normalized boxes."""
        if self.api_key is None:
            self.load()

        # Downsize + JPEG encode before sending (prevents 413)
        b64, W, H = _prepare_for_roboflow(image_bytes, max_side=1024, quality=85)

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        params = {"api_key": self.api_key, "format": "json"}
        url = f"{self.base}/{self.model_full}"

        r = requests.post(url, params=params, data={"image": b64}, headers=headers, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise RuntimeError(f"Roboflow error {r.status_code}: {msg}") from e

        data = r.json()

        out: List[Detection] = []
        for p in data.get("predictions", []):
            if str(p.get("class", "")).lower() != "vest":
                continue
            x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
            x1, y1 = x - w / 2.0, y - h / 2.0
            x2, y2 = x + w / 2.0, y + h / 2.0
            out.append(
                Detection(
                    label="vest",
                    confidence=float(p.get("confidence", 0.0)),
                    box=BoundingBox(
                        x1=max(0.0, x1 / W),
                        y1=max(0.0, y1 / H),
                        x2=min(1.0, x2 / W),
                        y2=min(1.0, y2 / H),
                    ),
                )
            )
        return out
