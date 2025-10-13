from fastapi import FastAPI, UploadFile
from inviol_image_analyser_assignment.models import AnalysisResult, RiskScore, Detection, BoundingBox, RuleBreach, AnalysisMeta

app = FastAPI(title="Image Analyser Assignment", version="0.1.0")



@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}


@app.post("/analyse")
async def post_analyse(file: UploadFile) -> AnalysisResult:
    # TODO: Implement the actual image analysis logic

    # Phase 1: return a well-formed, static payload to lock the contract.
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
    meta = AnalysisMeta(model="stub@phase1", inference_ms=1)

    return AnalysisResult(risk=risk, detections=detections, breaches=breaches, meta=meta)