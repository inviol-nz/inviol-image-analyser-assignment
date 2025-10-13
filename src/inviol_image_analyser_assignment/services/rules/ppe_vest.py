from __future__ import annotations
from typing import List
import numpy as np
from inviol_image_analyser_assignment.models import Detection, RuleBreach
from inviol_image_analyser_assignment.core.config import CFG
from inviol_image_analyser_assignment.utils.geometry import torso_crop
from inviol_image_analyser_assignment.utils.colors import mask_hi_vis

PERSON_LABEL = "person"

class VestRule:
    rule_id = "ppe_missing_vest"

    def evaluate(self, detections: List[Detection], image_bgr: np.ndarray) -> List[RuleBreach]:
        H, W = image_bgr.shape[:2]
        breaches: List[RuleBreach] = []
        for idx, det in enumerate(detections):
            if det.label != PERSON_LABEL:
                continue
            x1, y1, x2, y2 = torso_crop(det.box, W, H, CFG.vest.torso_y1, CFG.vest.torso_y2)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask = mask_hi_vis(crop)
            ratio = float(np.count_nonzero(mask)) / float(mask.size)
            if ratio < CFG.vest.min_ratio:
                sev = "medium" if ratio > CFG.vest.min_ratio * 0.5 else "high"
                breaches.append(RuleBreach(
                    rule_id=self.rule_id,
                    severity=sev,
                    message=f"Likely no high-vis vest (hi-vis ratio={ratio:.2f} < {CFG.vest.min_ratio})",
                    subjects=[idx],
                ))
        return breaches
