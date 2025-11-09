from __future__ import annotations

from io import BytesIO
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inviol_image_analyser_assignment.api.dependencies import require_api_key
from inviol_image_analyser_assignment.main import create_app
from inviol_image_analyser_assignment.models.detection import BoundingBox, Detection


class FakeDetector:
    """
    ObjectDetector used in unit tests.

    It ignores the actual image content and always returns a person near a truck,
    so the proximity rule fires in a deterministic way.
    """

    model_name = "fake-detector"
    model_version = "test-1.0"

    def detect(self, image: Image.Image):
        width, height = image.size

        # Person on the right
        person = Detection(
            id="det-person",
            label="person",
            confidence=0.99,
            bbox=BoundingBox(
                x_min=0.6 * width,
                y_min=0.3 * height,
                x_max=0.8 * width,
                y_max=0.9 * height,
            ),
        )

        # Truck on the left, reasonably close to the person
        truck = Detection(
            id="det-truck",
            label="truck",
            confidence=0.95,
            bbox=BoundingBox(
                x_min=0.2 * width,
                y_min=0.3 * height,
                x_max=0.7 * width,
                y_max=0.9 * height,
            ),
        )

        return [person, truck], width, height


class FakeDetectorNoDetections:
    """
    Detector that returns no detections at all.
    Useful for testing edge cases.
    """

    model_name = "fake-detector-empty"
    model_version = "test-1.0"

    def detect(self, image: Image.Image):
        width, height = image.size
        return [], width, height


@pytest.fixture()
def client(monkeypatch) -> Iterator[TestClient]:
    """
    TestClient with the ML model mocked out and API key disabled.

    - ObjectDetector return FakeDetector().
    - require_api_key is overridden to do nothing.
    """

    monkeypatch.setattr(
        "inviol_image_analyser_assignment.main.ObjectDetector",
        lambda: FakeDetector(),
    )

    app = create_app()
    app.dependency_overrides[require_api_key] = lambda: None

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def empty_detector_client(monkeypatch) -> Iterator[TestClient]:
    """
    TestClient wired with a detector that produces no detections.
    """

    from inviol_image_analyser_assignment.services import cv as cv_module

    monkeypatch.setattr(
        cv_module,
        "ObjectDetector",
        lambda: FakeDetectorNoDetections(),
    )

    app = create_app()
    app.dependency_overrides[require_api_key] = lambda: None

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def tiny_png_bytes() -> bytes:
    """
    Returns a small in-memory PNG image, used as a dummy upload.
    """
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
