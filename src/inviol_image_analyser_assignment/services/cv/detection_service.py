from __future__ import annotations

from typing import Tuple

from PIL import Image
from ultralytics import YOLO  # type: ignore[import]

from inviol_image_analyser_assignment.models.detection import BoundingBox, Detection


class ObjectDetector:
    """
    Thin wrapper around a pretrained YOLO object detection model.

    - Uses ultralytics YOLOv8 (COCO-pretrained) for simplicity.
    - Loaded once at startup and reused for all requests.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold

        self.model_name: str = "YOLOv8n (COCO)"
        try:
            import ultralytics  # type: ignore[import]

            self.model_version: str | None = ultralytics.__version__
        except Exception:  # pragma: no cover - optional metadata
            self.model_version = None

    def detect(self, image: Image.Image) -> Tuple[list[Detection], int, int]:
        """
        Run object detection on an image.

        Returns:
            - list of Detection models
            - image width
            - image height
        """
        results = self._model(image)
        result = results[0]

        height, width = result.orig_shape[:2]
        detections: list[Detection] = []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

        for idx, (xyxy, conf, cls_id) in enumerate(
            zip(boxes_xyxy, confidences, class_ids, strict=False),
        ):
            confidence = float(conf)
            if confidence < self._confidence_threshold:
                continue

            x_min, y_min, x_max, y_max = [float(v) for v in xyxy]
            class_index = int(cls_id)
            label = str(self._model.names.get(class_index, f"class_{class_index}"))

            detection = Detection(
                id=f"det-{idx}",
                label=label,
                confidence=confidence,
                bbox=BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
            )
            detections.append(detection)

        return detections, width, height
