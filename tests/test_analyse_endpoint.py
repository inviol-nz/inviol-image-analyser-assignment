from __future__ import annotations

from io import BytesIO

from fastapi import status


def test_analyse_returns_result(client, tiny_png_bytes: bytes) -> None:
    """
    Basic happy-path test: mocked detector returns some detections,
    and the endpoint returns a structured AnalysisResult.
    """
    files = {
        "file": ("test.png", BytesIO(tiny_png_bytes), "image/png"),
    }

    resp = client.post("/analyse", files=files)
    assert resp.status_code == status.HTTP_200_OK

    data = resp.json()
    assert "risk_rating" in data
    assert "overall_risk_score" in data
    assert data["model_name"] == "fake-detector"
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["label"] in {"person", "truck"}


def test_invalid_content_type_rejected(client) -> None:
    """
    The endpoint should reject non-image content types with a clear error.
    """
    resp = client.post(
        "/analyse",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST

    data = resp.json()
    assert data["code"] == "invalid_content_type"


def test_analyse_returns_proximity_violation(
    client,
    tiny_png_bytes: bytes,
) -> None:
    """
    With FakeDetector returning a person and a truck close together,
    /analyse should report a person_machine_proximity violation.
    """
    files = {
        "file": ("dummy.png", BytesIO(tiny_png_bytes), "image/png"),
    }

    resp = client.post("/analyse", files=files)
    assert resp.status_code == status.HTTP_200_OK

    data = resp.json()
    assert data["risk_rating"] >= 1
    assert data["overall_risk_score"] > 0.0

    violations = data.get("violations", [])
    assert any(
        v.get("rule_id") == "person_machine_proximity" for v in violations
    ), f"Expected person_machine_proximity violation, got: {violations}"


def test_analyse_no_detections_results_in_low_risk(
    empty_detector_client,
    tiny_png_bytes: bytes,
) -> None:
    """
    Edge case: when the detector finds no objects, overall risk should be low/zero.
    """
    files = {
        "file": ("dummy.png", BytesIO(tiny_png_bytes), "image/png"),
    }

    resp = empty_detector_client.post("/analyse", files=files)
    assert resp.status_code == status.HTTP_200_OK

    data = resp.json()
    assert data["risk_rating"] == 0
    assert data["overall_risk_score"] == 0.0
    assert data["risk_level"] == "low"
    assert data.get("violations", []) == []
