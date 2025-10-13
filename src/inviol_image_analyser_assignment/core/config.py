from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class VestCfg:
    """Configuration for color-based vest detection"""
    min_ratio: float = float(os.getenv("CFG_VEST_MIN_RATIO", 0.12))
    torso_y1: float = float(os.getenv("CFG_VEST_TORSO_Y1", 0.35))
    torso_y2: float = float(os.getenv("CFG_VEST_TORSO_Y2", 0.75))

    # association thresholds (if used later)
    vest_torso_iou_thr: float = float(os.getenv("CFG_VEST_TORSO_IOU_THR", 0.05))
    vest_torso_cover_thr: float = float(os.getenv("CFG_VEST_TORSO_COVER_THR", 0.30))
    center_x_frac: float = float(os.getenv("CFG_VEST_CENTER_X", 0.60))

    # HSV color thresholds (can be overridden with env vars)
    orange_lower: tuple[int, int, int] = (2, 150, 150)
    orange_upper: tuple[int, int, int] = (25, 255, 255)
    yellow_lower: tuple[int, int, int] = (25, 100, 150)
    yellow_upper: tuple[int, int, int] = (45, 255, 255)
    coverage_threshold: float = 0.1


@dataclass(frozen=True)
class RestrictedCfg:
    feet_offset: float = float(os.getenv("CFG_RESTRICTED_FEET_OFFSET", 0.02))


@dataclass(frozen=True)
class RiskCfg:
    low: int = 1
    med: int = 2
    high: int = 3


@dataclass(frozen=True)
class AppCfg:
    vest: VestCfg = VestCfg()
    restricted: RestrictedCfg = RestrictedCfg()
    risk: RiskCfg = RiskCfg()


# Global singleton
CFG = AppCfg()
