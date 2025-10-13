from __future__ import annotations

import time
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from inviol_image_analyser_assignment.models import (
    AnalysisResult,
    AnalysisMeta,
    RiskScore,
    Detection,
    BoundingBox,
    RuleBreach,
)

# Use the Ultralytics backend inside Docker (Torch CPU wheels installed there)
from inviol_image_analyser_assignment.services.detection.yolo import YOLOv8Backend
# If you went with ONNX instead, import that backend and instantiate it similarly.

app = FastAPI(title="Image Analyser Assignment - by raffaele", version="0.1.0")

# ---- Detection backend singleton (lazy-loaded) ----
_detector = YOLOv8Backend(model_name="yolov8n.pt", conf=0.25, imgsz=640)

@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}

@app.post("/debug/detect", response_model=List[Detection])
async def debug_detect(file: UploadFile = File(...)) -> List[Detection]:
    """
    Phase-2 checkpoint: run the detector only and return raw detections.
    This lets you sanity-check the model wiring and performance in isolation.
    """
    try:
        image_bytes = await file.read()
        t0 = time.perf_counter()
        detections = _detector.predict(image_bytes)  # lazy-loads model on first call
        _inference_ms = (time.perf_counter() - t0) * 1000.0  # you can log this later
        return detections
    except Exception as e:
        # If ultralytics/torch is missing (not the case in Docker), you’ll see it here.
        raise HTTPException(status_code=500, detail=str(e))

# Keep /analyse as your Phase 1 stub for now (we’ll make it real in Phase 3)
@app.post("/analyse", response_model=AnalysisResult)
async def post_analyse(file: UploadFile = File(...)) -> AnalysisResult:
    detections = [
        Detection(
            label="person",
            confidence=0.93,
            box=BoundingBox(x1=0.12, y1=0.15, x2=0.42, y2=0.60),
        ),
        Detection(
            label="helmet",
            confidence=0.88,
            box=BoundingBox(x1=0.18, y1=0.10, x2=0.30, y2=0.22),
        ),
    ]
    breaches = [
        RuleBreach(
            rule_id="prox_veh_person",
            severity="medium",
            message="Person within 1.5m of moving vehicle",
            subjects=[0],
        )
    ]
    risk = RiskScore(overall=5, by_rule={"prox_veh_person": 3, "ppe_missing_helmet": 2})
    meta = AnalysisMeta(model="yolov8n@phase2", inference_ms=1)
    return AnalysisResult(risk=risk, detections=detections, breaches=breaches, meta=meta)
