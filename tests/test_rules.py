from inviol_image_analyser_assignment.models.detection import BoundingBox, Detection
from inviol_image_analyser_assignment.services.rules.base import RuleContext
from inviol_image_analyser_assignment.services.rules.config import load_rules_config
from inviol_image_analyser_assignment.services.rules.crowding import CrowdingRule
from inviol_image_analyser_assignment.services.rules.proximity_person_machine import (
    PersonMachineProximityRule,
)


def make_person(id_: str, x: float, y: float) -> Detection:
    return Detection(
        id=id_,
        label="person",
        confidence=0.9,
        bbox=BoundingBox(
            x_min=x - 5,
            y_min=y - 5,
            x_max=x + 5,
            y_max=y + 5,
        ),
    )


def make_car(id_: str, x: float, y: float) -> Detection:
    return Detection(
        id=id_,
        label="car",
        confidence=0.9,
        bbox=BoundingBox(
            x_min=x - 10,
            y_min=y - 10,
            x_max=x + 10,
            y_max=y + 10,
        ),
    )


def test_crowding_uses_config_thresholds():
    cfg = load_rules_config()
    rule = CrowdingRule(cfg.crowding)

    assert rule.max_safe_people == cfg.crowding.max_safe_people
    assert rule.high_risk_extra_people == cfg.crowding.high_risk_extra_people


def test_crowding_no_violation_when_at_or_below_safe_limit():
    cfg = load_rules_config()
    rule = CrowdingRule(cfg.crowding)
    ctx = RuleContext(image_width=640, image_height=480)

    # Exactly at safe limit
    count = cfg.crowding.max_safe_people
    detections = [make_person(f"p{i}", 10 * i, 10 * i) for i in range(count)]
    violations = list(rule.evaluate(detections, ctx))

    assert violations == []


def test_crowding_violation_when_above_safe_limit():
    cfg = load_rules_config()
    rule = CrowdingRule(cfg.crowding)
    ctx = RuleContext(image_width=640, image_height=480)

    count = cfg.crowding.max_safe_people + 2
    detections = [make_person(f"p{i}", 10 * i, 10 * i) for i in range(count)]
    violations = list(rule.evaluate(detections, ctx))

    assert len(violations) == 1
    v = violations[0]
    assert v.rule_id == "crowding"
    assert str(count) in v.description


def test_person_machine_proximity_uses_config_thresholds():
    cfg = load_rules_config()
    rule = PersonMachineProximityRule(cfg.person_machine_proximity)

    assert rule.medium_risk_distance_ratio == cfg.person_machine_proximity.medium_risk_distance_ratio
    assert rule.high_risk_distance_ratio == cfg.person_machine_proximity.high_risk_distance_ratio


def test_person_machine_proximity_flags_close_person():
    cfg = load_rules_config()
    rule = PersonMachineProximityRule(cfg.person_machine_proximity)
    ctx = RuleContext(image_width=1000, image_height=1000)

    # Put person and car at exactly the same centre -> distance 0 -> must be HIGH
    person = make_person("p1", 500, 500)
    car = make_car("c1", 500, 500)

    violations = list(rule.evaluate([person, car], ctx))

    assert len(violations) >= 1
    v = violations[0]
    assert "p1" in v.involved_detection_ids
    assert "c1" in v.involved_detection_ids
