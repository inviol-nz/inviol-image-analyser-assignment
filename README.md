
# Inviol Image Analyser

FastAPI-based computer vision service for workplace safety risk detection.
It accepts image uploads, performs object detection using a pretrained model, applies configurable safety rules, and returns a structured JSON risk report.

---

## Quick Start

### 1. Install dependencies

The project uses [**uv**](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### 2. Environment setup

Copy the example env file and set your variables:

```bash
cp .env.example .env
```

Key values:

```bash
APP_ENV=dev
API_KEY=your_api_key_here
CACHE_ENABLED=true
CACHE_SIZE=128
```

Optional environment-specific overrides can go in `.env.dev` or `.env.prod`.
The active one is selected by `APP_ENV`.

### 3. Run the API

```bash
uv run uvicorn inviol_image_analyser_assignment.main:create_app --factory --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Features

| Area             | Description                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Model**        | Uses YOLOv8 pretrained on COCO for person/vehicle/equipment detection.                                              |
| **Rules Engine** | Evaluates proximity, crowding, and other safety conditions. Rules and thresholds are loaded from JSON config files. |
| **Config**       | `.env` files for environment setup; `rules.json` for dynamic rule weights.                                          |
| **Validation**   | Accepts JPEG/PNG images via multipart/form-data, validates file type and size, and returns clear error messages.    |
| **Security**     | Simple API-key middleware for protected endpoints.                                                                  |
| **Caching**      | Optional in-memory cache for repeated image analyses.                                                               |
| **Tests**        | Unit and API tests using `pytest` with mocked detectors.                                       |

---

## Testing

Run all tests (unit + API):

```bash
uv run pytest
```

Mocked detectors are used in tests for speed and isolation—no real YOLO inference runs.

---

## Design Choices

* **YOLOv8** chosen for speed and simplicity; swappable via `ObjectDetector` abstraction.
* **RuleEngine** decoupled from detection → new safety logic can be added via config without touching core code.
* **.env + JSON configs** allow easy tuning of thresholds, risk weights, and environment modes.

In the sample_images folder was added few truck_*.jpg images because it can be detected by YOLOv8 better than forklift.

---

## Future / Production Considerations

* Switch to GPU-based inference or model service for high-load environments.
* Add job queue (Celery / Redis) for async, large-batch image processing.
* Support Detectron2 or fine-tuned YOLO models for custom safety gear detection (PPE, helmets, vests).
* Add streaming/video analysis and real-time dashboard.
* Adding video processing using Decord / FFMPEG

---

## Example Request

```bash
curl -X POST "http://localhost:8000/analyse" \
  -H "x-api-key: your_api_key_here" \
  -F "file=@sample_images/truck_1.jpg"
```

Example response:

```json
{
  "risk_rating": 3,
  "overall_risk_score": 3.0,
  "risk_level": "medium",
  "violations": [
    {
      "rule_id": "person_machine_proximity",
      "severity": "medium",
      "description": "Person near moving vehicle."
    }
  ]
}
```
