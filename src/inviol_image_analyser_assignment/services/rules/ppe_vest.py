"""
Rule: checks that each detected person has a corresponding vest detection.

This rule operates purely on *geometric overlap* between detections.
It assumes an upstream model (e.g., RoboflowVestBackend) produces explicit
`"vest"` detections in addition to `"person"` detections.

For each detected person, it computes an approximate *torso region*
and tests whether any vest overlaps that region sufficiently.
If not, it records a high-severity `RuleBreach`.
"""
from __future__ import annotations
from typing import List
from inviol_image_analyser_assignment.models import Detection, RuleBreach, BoundingBox
from inviol_image_analyser_assignment.core.config import CFG
from inviol_image_analyser_assignment.utils.geometry import iou, frac_inside, torso_region_norm

PERSON, VEST = "person", "vest"

class VestRule:
    rule_id = "ppe_missing_vest"

    def evaluate(self, detections: List[Detection], image_bgr) -> List[RuleBreach]:
        people = [(i,d) for i,d in enumerate(detections) if d.label == PERSON]
        vests  = [d for d in detections if d.label == VEST]

        breaches: List[RuleBreach] = []
        for idx, person in people:
            torso: BoundingBox = torso_region_norm(
                person.box,
                CFG.vest.torso_y1,
                CFG.vest.torso_y2,
                CFG.vest.center_x_frac,
            )
            # any vest that overlaps torso sufficiently?
            matched = False
            for v in vests:
                if iou(v.box, torso) >= CFG.vest.vest_torso_iou_thr or \
                   frac_inside(v.box, torso) >= CFG.vest.vest_torso_cover_thr:
                    matched = True
                    break
            if not matched:
                breaches.append(RuleBreach(
                    rule_id=self.rule_id,
                    severity="high",
                    message="No vest detected overlapping torso region",
                    subjects=[idx],
                ))
        return breaches
