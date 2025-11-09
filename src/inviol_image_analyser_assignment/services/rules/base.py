from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from inviol_image_analyser_assignment.models.detection import Detection
from inviol_image_analyser_assignment.models.rule_violation import RuleViolation


@dataclass
class RuleContext:
    """
    Contextual information for rule evaluation.

    For now this is just the image size, but it could be extended with
    camera metadata, site configuration, etc.
    """

    image_width: int
    image_height: int


class SafetyRule(Protocol):
    """
    Protocol that all safety rules must implement.
    """

    id: str
    name: str
    description: str

    def evaluate(
        self,
        detections: list[Detection],
        context: RuleContext,
    ) -> Iterable[RuleViolation]:
        ...
