from __future__ import annotations

from typing import Iterable

from inviol_image_analyser_assignment.models.detection import Detection
from inviol_image_analyser_assignment.models.enums import RiskLevel
from inviol_image_analyser_assignment.models.rule_violation import RuleViolation
from inviol_image_analyser_assignment.services.rules.base import RuleContext, SafetyRule
from inviol_image_analyser_assignment.services.rules.config import CrowdingRuleConfig


class CrowdingRule(SafetyRule):
    """
    Flags scenes with an unusually high number of people.

    Configuration:
    - max_safe_people: number of people considered safe in the scene.
    - high_risk_extra_people: if extra people >= this value, severity is HIGH.
    - medium/high_risk_score_per_person: contribution to overall risk per extra person.

    All of these thresholds and scores come from `CrowdingRuleConfig`, which is
    loaded from config/rules.json.
    """

    id = "crowding"
    name = "Crowding"
    description = "Flags scenes with too many people present."

    def __init__(self, config: CrowdingRuleConfig | None = None) -> None:
        cfg = config or CrowdingRuleConfig()
        self.max_safe_people = cfg.max_safe_people
        self.high_risk_extra_people = cfg.high_risk_extra_people
        self.medium_risk_score_per_person = cfg.medium_risk_score_per_person
        self.high_risk_score_per_person = cfg.high_risk_score_per_person

    def evaluate(
        self,
        detections: list[Detection],
        context: RuleContext,  # noqa: ARG002 - reserved for future use
    ) -> Iterable[RuleViolation]:
        people: list[Detection] = [
            d for d in detections if d.label.lower() == "person"
        ]
        count = len(people)

        if count <= self.max_safe_people:
            return []

        extra = count - self.max_safe_people

        if extra >= self.high_risk_extra_people:
            severity = RiskLevel.HIGH
            score_per_person = self.high_risk_score_per_person
        else:
            severity = RiskLevel.MEDIUM
            score_per_person = self.medium_risk_score_per_person

        score = score_per_person * float(extra)

        description = (
            f"{count} people detected; safe maximum is "
            f"{self.max_safe_people}. Extra people: {extra}."
        )

        involved_ids = [p.id for p in people]

        violation = RuleViolation(
            rule_id=self.id,
            rule_name=self.name,
            severity=severity,
            risk_score=score,
            description=description,
            involved_detection_ids=involved_ids,
        )

        return [violation]
