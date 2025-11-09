from __future__ import annotations

from pydantic import BaseModel

from .detection import Detection
from .enums import RiskLevel
from .rule_violation import RuleViolation


class AnalysisResult(BaseModel):
    """
    - risk_rating: integer 0–10 summary of overall risk.
    - overall_risk_score: underlying float score used to derive risk_rating.
    - risk_level: coarse-grained level (low/medium/high) derived from score.
    """

    risk_rating: int
    overall_risk_score: float
    risk_level: RiskLevel

    model_name: str
    model_version: str | None

    detections: list[Detection]
    violations: list[RuleViolation]
