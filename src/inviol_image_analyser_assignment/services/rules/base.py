from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from inviol_image_analyser_assignment.models import Detection, RuleBreach

class SafetyRule(ABC):
    rule_id: str
    def __init__(self, rule_id: str):
        self.rule_id = rule_id

    @abstractmethod
    def evaluate(self, detections: List[Detection], image_bgr: Optional[np.ndarray]) -> List[RuleBreach]:
        ...
