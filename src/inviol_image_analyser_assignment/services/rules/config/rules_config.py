from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Directory of this file: .../services/rules/config
BASE_DIR = Path(__file__).resolve().parent
RULES_FILE = BASE_DIR / "rules.json"


@dataclass
class CrowdingRuleConfig:
    """
    All fields have sensible defaults, so the service still works even if the
    JSON file is missing or partially defined.
    """

    max_safe_people: int = 5
    high_risk_extra_people: int = 5
    medium_risk_score_per_person: float = 0.8
    high_risk_score_per_person: float = 1.2
    weight: float = 1.0


@dataclass
class PersonMachineProximityConfig:
    """
    Configuration for the person/machinery proximity rule.
    """

    medium_risk_distance_ratio: float = 0.25
    high_risk_distance_ratio: float = 0.15
    medium_risk_score: float = 2.0
    high_risk_score: float = 3.0
    weight: float = 1.5


@dataclass
class RulesConfig:
    """
    Top-level rule configuration object.
    """

    crowding: CrowdingRuleConfig
    person_machine_proximity: PersonMachineProximityConfig


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_rules_config() -> RulesConfig:
    """
    Load rule thresholds and weights from rules.json.

    This is stateless and is called at evaluation time, that means changes to rules.json are picked up at runtime without restarting the service.
    """
    if not RULES_FILE.exists():
        # Fall back to defaults if file is missing
        return RulesConfig(
            crowding=CrowdingRuleConfig(),
            person_machine_proximity=PersonMachineProximityConfig(),
        )

    try:
        with RULES_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        # On any parse error, use defaults rather than crashing the service
        return RulesConfig(
            crowding=CrowdingRuleConfig(),
            person_machine_proximity=PersonMachineProximityConfig(),
        )

    crowding_raw = raw.get("crowding", {})
    prox_raw = raw.get("person_machine_proximity", {})

    crowding = CrowdingRuleConfig(
        max_safe_people=_as_int(crowding_raw.get("max_safe_people"), 5),
        high_risk_extra_people=_as_int(
            crowding_raw.get("high_risk_extra_people"),
            5,
        ),
        medium_risk_score_per_person=_as_float(
            crowding_raw.get("medium_risk_score_per_person"),
            0.8,
        ),
        high_risk_score_per_person=_as_float(
            crowding_raw.get("high_risk_score_per_person"),
            1.2,
        ),
        weight=_as_float(crowding_raw.get("weight"), 1.0),
    )

    person_machine_proximity = PersonMachineProximityConfig(
        medium_risk_distance_ratio=_as_float(
            prox_raw.get("medium_risk_distance_ratio"),
            0.1,
        ),
        high_risk_distance_ratio=_as_float(
            prox_raw.get("high_risk_distance_ratio"),
            0.05,
        ),
        medium_risk_score=_as_float(prox_raw.get("medium_risk_score"), 2.0),
        high_risk_score=_as_float(prox_raw.get("high_risk_score"), 3.0),
        weight=_as_float(prox_raw.get("weight"), 1.5),
    )

    return RulesConfig(
        crowding=crowding,
        person_machine_proximity=person_machine_proximity,
    )
