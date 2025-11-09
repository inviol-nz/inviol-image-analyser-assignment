from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from inviol_image_analyser_assignment.models.detection import Detection
from inviol_image_analyser_assignment.models.enums import RiskLevel
from inviol_image_analyser_assignment.models.rule_violation import RuleViolation
from inviol_image_analyser_assignment.services.rules.base import RuleContext, SafetyRule
from inviol_image_analyser_assignment.services.rules.config import load_rules_config
from inviol_image_analyser_assignment.services.rules.crowding import CrowdingRule
from inviol_image_analyser_assignment.services.rules.proximity_person_machine import (
    PersonMachineProximityRule,
)


@dataclass
class RuleEngine:
    """
    Coordinates evaluation of safety rules and aggregates overall risk.

    Design choice:
    - Rule thresholds and risk weights are loaded from JSON at evaluation time
      via `load_rules_config()`. This means they can be adjusted at runtime
      without restarting the service.
    """

    @classmethod
    def default(cls) -> "RuleEngine":
        # factory method for future extensibility
        return cls()

    def evaluate(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, RiskLevel, int, list[RuleViolation]]:
        """
        Evaluate all configured rules and compute an overall risk assessment.

        Returns:
            - overall_risk_score (float, capped at 10.0)
            - risk_level (LOW / MEDIUM / HIGH)
            - risk_rating (int 0–10)
            - list of individual rule violations
        """
        # Load rule configuration at runtime (thresholds + weights).
        cfg = load_rules_config()
        context = RuleContext(image_width=image_width, image_height=image_height)

        # Instantiate rules with their current configuration, format: (rule_instance, weight).
        rules: list[tuple[SafetyRule, float]] = [
            (PersonMachineProximityRule(cfg.person_machine_proximity), cfg.person_machine_proximity.weight),
            (CrowdingRule(cfg.crowding), cfg.crowding.weight),
        ]

        violations: list[RuleViolation] = []
        total_score = 0.0

        for rule, weight in rules:
            rule_violations = list(rule.evaluate(detections=detections, context=context))
            violations.extend(rule_violations)
            for v in rule_violations:
                # Apply rule-specific weight when aggregating risk.
                total_score += v.risk_score * weight

        # Cap the score so we stay on a 0–10 scale even with many violations.
        overall_score = min(total_score, 10.0)

        if overall_score < 3.0:
            level = RiskLevel.LOW
        elif overall_score < 7.0:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH

        risk_rating = int(round(overall_score))

        return overall_score, level, risk_rating, violations
