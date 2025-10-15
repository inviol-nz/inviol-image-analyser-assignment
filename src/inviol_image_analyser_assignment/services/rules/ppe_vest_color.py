
"""
Rule: detects missing high-visibility vest based on color segmentation.

This rule does **not** rely on object detections for vests.
Instead, it inspects each detected person’s torso region and searches for
orange/yellow HSV color ranges typical of safety vests.

If the fraction of vest-colored pixels is below a configurable threshold,
the person is flagged for missing visible PPE.

"""
import cv2
import numpy as np
from inviol_image_analyser_assignment.models import Detection, RuleBreach
from inviol_image_analyser_assignment.services.rules.base import SafetyRule

class VestColorRule(SafetyRule):
    """
    Detects presence of orange/yellow high-vis vest
    based on HSV color segmentation inside torso region
    of detected persons.
    """

    def __init__(self, coverage_threshold: float = 0.1):
        super().__init__(rule_id="ppe_vest_color")
        self.coverage_threshold = coverage_threshold

        # HSV bounds for vest colors (tunable)
        self.orange_lower = np.array((2, 150, 150))
        self.orange_upper = np.array((25, 255, 255))
        self.yellow_lower = np.array((25, 100, 150))
        self.yellow_upper = np.array((45, 255, 255))

    def evaluate(self, detections: list[Detection], image_bgr: np.ndarray) -> list[RuleBreach]:
        breaches: list[RuleBreach] = []

        for det in detections:
            if det.label != "person":
                continue

            H, W, _ = image_bgr.shape
            x1 = int(det.box.x1 * W)
            y1 = int(det.box.y1 * H)
            x2 = int(det.box.x2 * W)
            y2 = int(det.box.y2 * H)

            # torso = middle third
            torso_top = y1 + int((y2 - y1) / 3)
            torso_bottom = y1 + int((y2 - y1) * 2 / 3)
            torso = image_bgr[torso_top:torso_bottom, x1:x2]
            if torso.size == 0:
                continue

            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

            mask_orange = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
            mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            mask = cv2.bitwise_or(mask_orange, mask_yellow)

            vest_ratio = np.count_nonzero(mask) / mask.size

            if vest_ratio < self.coverage_threshold:
                breaches.append(RuleBreach(
                    rule_id=self.rule_id,
                    message="Person not wearing visible high-vis vest (no orange/yellow detected)",
                    severity="high",
                    subjects=[0]  # or whatever subject/person index fits your data
                ))

        return breaches
