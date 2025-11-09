import io

from fastapi.testclient import TestClient
from PIL import Image

from inviol_image_analyser_assignment.main import app
from inviol_image_analyser_assignment.models.detection import BoundingBox, Detection
from inviol_image_analyser_assignment.services.rule_engine import RuleEngine


class FakeDetector:
    model_name = "fake-detector"
    model_version = "test-1.0"

    def detect(self, image: Image.Image):
        width, height = image.size
        bbox = BoundingBox(
            x_min=width / 2 - 10,
            y_min=height / 2 - 10,
            x_max=width / 2 + 10,
            y_max=height / 2 + 10,
        )
        detection = Detection(
            id="det-0",
            label="person",
            confidence=0.99,
            bbox=bbox,
        )
        return [detection], width, height


def create_test_image() -> bytes:
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_analyse_returns_result():
    client = TestClient(app)

    # Override heavy services with test doubles
    app.state.detector = FakeDetector()
    app.state.rule_engine = RuleEngine.default()
    app.state.cache = None

    image_bytes = create_test_image()
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}

    response = client.post("/analyse", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "risk_rating" in data
    assert data["model_name"] == "fake-detector"
    assert len(data["detections"]) == 1
    assert data["detections"][0]["label"] == "person"


def test_invalid_content_type_rejected():
    client = TestClient(app)

    response = client.post(
        "/analyse",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "invalid_content_type"
