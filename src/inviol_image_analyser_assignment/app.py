import io, time
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List


from inviol_image_analyser_assignment.services.detection.roboflow_vest import RoboflowVestBackend
from inviol_image_analyser_assignment.services.detection.yolo import YOLOv8Backend
from inviol_image_analyser_assignment.services.rules.ppe_vest import VestRule
from inviol_image_analyser_assignment.services.rules.restricted import RestrictedRule
from inviol_image_analyser_assignment.core.config import CFG
from inviol_image_analyser_assignment.models import (
    AnalysisResult, AnalysisMeta, RiskScore, Detection, RuleBreach
)

app = FastAPI(title="Image Analyser Assignment - by raffaele ciao", version="0.1.0")

# ---- Detection backend singleton (lazy-loaded) ----
_detector = YOLOv8Backend(model_name="yolov8n.pt", conf=0.25, imgsz=640)
_rules = [VestRule(), RestrictedRule()]
_ppe = RoboflowVestBackend(
    model_id="aa-sutfb/vest-qf3av",
    version="1",
    conf=0.25,
)

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

@app.post("/analyse", response_model=AnalysisResult)
async def post_analyse(file: UploadFile = File(...)) -> AnalysisResult:
    try:
        img_bytes = await file.read()

        t0 = time.perf_counter()
        base_dets = _detector.predict(img_bytes)         # has "person", "car", "truck", ...
        ppe_dets  = _ppe.predict(img_bytes)              # has "vest" (and maybe "novest")
        detections = base_dets + ppe_dets                # merge so the response includes vest boxes too
        inf_ms = int((time.perf_counter() - t0) * 1000.0)

        # decode once for rules that need pixels (restricted rule uses pixels)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        breaches: list[RuleBreach] = []
        for rule in _rules:
            breaches.extend(rule.evaluate(detections, bgr))

        weight = {"low": CFG.risk.low, "medium": CFG.risk.med, "high": CFG.risk.high}
        by_rule: dict[str, int] = {}
        for br in breaches:
            by_rule[br.rule_id] = by_rule.get(br.rule_id, 0) + weight.get(br.severity, 1)
        overall = min(10, sum(by_rule.values()))

        return AnalysisResult(
            risk=RiskScore(overall=overall, by_rule=by_rule),
            detections=detections,
            breaches=breaches,
            meta=AnalysisMeta(model="yolov8n + roboflow-vest", inference_ms=inf_ms),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
