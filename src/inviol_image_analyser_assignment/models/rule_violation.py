from __future__ import annotations

from pydantic import BaseModel

from .enums import RiskLevel


class RuleViolation(BaseModel):
    """
    A single safety rule breach detected in the image.
    """

    rule_id: str
    rule_name: str
    severity: RiskLevel
    risk_score: float
    description: str
    involved_detection_ids: list[str]
