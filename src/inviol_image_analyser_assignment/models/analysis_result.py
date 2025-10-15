from __future__ import annotations
from typing import Annotated, Dict, List, Literal
from pydantic import BaseModel, Field

# Type aliases that satisfy Pyright
NormFloat = Annotated[float, Field(ge=0.0, le=1.0)]
ScoreInt = Annotated[int, Field(ge=0, le=10)]

class BoundingBox(BaseModel):
    """Normalized axis-aligned bounding box in [0,1] coordinates."""
    x1: NormFloat = Field(..., description="Left (0–1)")
    y1: NormFloat = Field(..., description="Top (0–1)")
    x2: NormFloat = Field(..., description="Right (0–1)")
    y2: NormFloat = Field(..., description="Bottom (0–1)")

    model_config = {
        "json_schema_extra": {
            "example": {"x1": 0.12, "y1": 0.15, "x2": 0.42, "y2": 0.60}
        }
    }

class Detection(BaseModel):
    """A single model detection with class label, confidence, and bounding box."""
    label: str = Field(..., description="e.g. 'person', 'helmet'")
    confidence: NormFloat = Field(..., description="Score [0,1]")
    box: BoundingBox

    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "person",
                "confidence": 0.93,
                "box": {"x1": 0.12, "y1": 0.15, "x2": 0.42, "y2": 0.60},
            }
        }
    }

Severity = Literal["low", "medium", "high", "critical"]

class RuleBreach(BaseModel):
    """
    A safety rule breach referencing detections by index.
    `subjects` contains indices into the `detections` array.
    """
    rule_id: str = Field(..., description="Stable id, e.g. 'prox_veh_person'")
    severity: Severity
    message: str
    subjects: List[int] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "rule_id": "ppe_missing_helmet",
                "severity": "high",
                "message": "Person without helmet in construction zone",
                "subjects": [0],
            }
        }
    }

class RiskScore(BaseModel):
    """Overall risk plus per-rule contributions (0–10)."""
    overall: ScoreInt = Field(..., description="Overall risk 0–10")
    by_rule: Dict[str, ScoreInt] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {"example": {"overall": 6, "by_rule": {"ppe_missing_helmet": 4}}}
    }

class AnalysisMeta(BaseModel):
    """Debug/observability metadata."""
    model: str = Field(..., description="Model name/version")
    inference_ms: int = Field(..., ge=0, description="Model time in ms")

    model_config = {"json_schema_extra": {"example": {"model": "stub@phase1", "inference_ms": 1}}}

class AnalysisResult(BaseModel):
    """Top-level response returned by /analyse."""
    risk: RiskScore
    detections: List[Detection] = Field(default_factory=list)
    breaches: List[RuleBreach] = Field(default_factory=list)
    meta: AnalysisMeta

    model_config = {
        "json_schema_extra": {
            "example": {
                "risk": {"overall": 5, "by_rule": {"prox_veh_person": 3, "ppe_missing_helmet": 2}},
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.93,
                        "box": {"x1": 0.12, "y1": 0.15, "x2": 0.42, "y2": 0.60},
                    },
                    {
                        "label": "helmet",
                        "confidence": 0.88,
                        "box": {"x1": 0.18, "y1": 0.10, "x2": 0.30, "y2": 0.22},
                    },
                ],
                "breaches": [
                    {
                        "rule_id": "prox_veh_person",
                        "severity": "medium",
                        "message": "Person within 1.5m of moving vehicle",
                        "subjects": [0],
                    }
                ],
                "meta": {"model": "stub@phase1", "inference_ms": 1},
            }
        }
    }
