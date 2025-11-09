from __future__ import annotations

import math
from typing import Iterable

from inviol_image_analyser_assignment.models.detection import Detection
from inviol_image_analyser_assignment.models.enums import RiskLevel
from inviol_image_analyser_assignment.models.rule_violation import RuleViolation
from inviol_image_analyser_assignment.services.rules.base import RuleContext, SafetyRule
from inviol_image_analyser_assignment.services.rules.config import (
    PersonMachineProximityConfig,
)


class PersonMachineProximityRule(SafetyRule):
    """
    Flags situations where people are too close to heavy machinery.

    Configuration (from rules.json via PersonMachineProximityConfig):
    - medium/high_risk_distance_ratio: thresholds as fraction of image diagonal.
    - medium/high_risk_score: contribution to overall risk for each violation.
    """

    id = "person_machine_proximity"
    name = "Person near machinery"
    description = (
        "Flags people standing too close to heavy machinery "
        "(cars, trucks, buses, motorcycles)."
    )

    HEAVY_MACHINERY_LABELS = {"car", "truck", "bus", "motorcycle", "forklift"}

    def __init__(self, config: PersonMachineProximityConfig | None = None) -> None:
        cfg = config or PersonMachineProximityConfig()
        self.medium_risk_distance_ratio = cfg.medium_risk_distance_ratio
        self.high_risk_distance_ratio = cfg.high_risk_distance_ratio
        self.medium_risk_score = cfg.medium_risk_score
        self.high_risk_score = cfg.high_risk_score

    def evaluate(
        self,
        detections: list[Detection],
        context: RuleContext,
    ) -> Iterable[RuleViolation]:
        people: list[Detection] = [
            d for d in detections if d.label.lower() == "person"
        ]
        machinery: list[Detection] = [
            d for d in detections
            if d.label.lower() in self.HEAVY_MACHINERY_LABELS
        ]

        if not people or not machinery:
            return []

        diag = math.hypot(context.image_width, context.image_height) or 1.0
        violations: list[RuleViolation] = []

        for person in people:
            px, py = self._centre(person)
            for machine in machinery:
                mx, my = self._centre(machine)
                distance = math.hypot(px - mx, py - my)
                ratio = distance / diag

                if ratio < self.high_risk_distance_ratio:
                    severity = RiskLevel.HIGH
                    score = self.high_risk_score
                elif ratio < self.medium_risk_distance_ratio:
                    severity = RiskLevel.MEDIUM
                    score = self.medium_risk_score
                else:
                    continue

                description = (
                    f"Person '{person.id}' is very close to machinery "
                    f"'{machine.id}' (normalised distance={ratio:.3f})."
                )
                violations.append(
                    RuleViolation(
                        rule_id=self.id,
                        rule_name=self.name,
                        severity=severity,
                        risk_score=score,
                        description=description,
                        involved_detection_ids=[person.id, machine.id],
                    ),
                )

        return violations

    @staticmethod
    def _centre(det: Detection) -> tuple[float, float]:
        bbox = det.bbox
        return (bbox.x_min + bbox.x_max) / 2.0, (bbox.y_min + bbox.y_max) / 2.0
