from __future__ import annotations
from typing import List
import numpy as np
from inviol_image_analyser_assignment.models import Detection, RuleBreach
from inviol_image_analyser_assignment.core.config import CFG
from inviol_image_analyser_assignment.utils.geometry import bottom_center
from inviol_image_analyser_assignment.utils.colors import mask_red

PERSON_LABEL = "person"

class RestrictedRule:
    rule_id = "restricted_zone"

    def evaluate(self, detections: List[Detection], image_bgr: np.ndarray) -> List[RuleBreach]:
        H, W = image_bgr.shape[:2]
        red = mask_red(image_bgr)  # uint8 mask
        breaches: List[RuleBreach] = []
        for idx, det in enumerate(detections):
            if det.label != PERSON_LABEL:
                continue
            x, y = bottom_center(det.box, W, H, offset=CFG.restricted.feet_offset)
            if red[y, x] > 0:
                breaches.append(RuleBreach(
                    rule_id=self.rule_id,
                    severity="high",
                    message="Person appears within a red restricted area",
                    subjects=[idx],
                ))
        return breaches
