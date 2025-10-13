import io, time
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

# Roboflow backend removed
# from inviol_image_analyser_assignment.services.detection.roboflow_vest import RoboflowVestBackend
from inviol_image_analyser_assignment.services.detection.yolo import YOLOv8Backend
from inviol_image_analyser_assignment.services.rules.ppe_vest_color import VestColorRule
from inviol_image_analyser_assignment.services.rules.restricted import RestrictedRule
from inviol_image_analyser_assignment.core.config import CFG
from inviol_image_analyser_assignment.models import (
    AnalysisResult, AnalysisMeta, RiskScore, Detection, RuleBreach
)

app = FastAPI(title="Image Analyser Assignment - by raffaele ciaoxx9", version="0.2.0")

# ---- Detection backend singleton (lazy-loaded) ----
_detector = YOLOv8Backend(model_name="yolov8n.pt", conf=0.25, imgsz=640)

# Use only your local color-based vest rule + restricted area rule
_rules = [VestColorRule(coverage_threshold=0.1), RestrictedRule()]

# Roboflow vest backend disabled
# _ppe = RoboflowVestBackend(model_id="vests-287rc/1")


@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}


@app.post("/debug/detect", response_model=List[Detection])
async def debug_detect(file: UploadFile = File(...)) -> List[Detection]:
    """
    Debug endpoint: run YOLO detector only and return raw detections.
    """
    try:
        image_bytes = await file.read()
        t0 = time.perf_counter()
        detections = _detector.predict(image_bytes)
        _inference_ms = (time.perf_counter() - t0) * 1000.0
        return detections
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyse", response_model=AnalysisResult)
async def post_analyse(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Full analysis pipeline using YOLO detections + color-based vest rule.
    """
    try:
        img_bytes = await file.read()

        #  Run person/object detection
        t0 = time.perf_counter()
        detections = _detector.predict(img_bytes)
        inf_ms = int((time.perf_counter() - t0) * 1000.0)

        # Decode for rules needing pixel access (e.g. vest color, restricted)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Evaluate all safety rules
        breaches: list[RuleBreach] = []
        for rule in _rules:
            breaches.extend(rule.evaluate(detections, bgr))

        #  Compute risk score
        weight = {"low": CFG.risk.low, "medium": CFG.risk.med, "high": CFG.risk.high}
        by_rule: dict[str, int] = {}
        for br in breaches:
            by_rule[br.rule_id] = by_rule.get(br.rule_id, 0) + weight.get(br.severity, 1)
        overall = min(10, sum(by_rule.values()))

        # Return structured response
        return AnalysisResult(
            risk=RiskScore(overall=overall, by_rule=by_rule),
            detections=detections,
            breaches=breaches,
            meta=AnalysisMeta(model="yolov8n + color-vest", inference_ms=inf_ms),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
