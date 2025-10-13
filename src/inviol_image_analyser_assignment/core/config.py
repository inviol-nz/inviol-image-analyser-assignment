from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class VestCfg:
    # fraction of torso pixels that must be hi-vis to pass
    min_ratio: float = float(os.getenv("CFG_VEST_MIN_RATIO", 0.12))
    # torso crop: middle vertical band of the person box
    torso_y1: float = float(os.getenv("CFG_VEST_TORSO_Y1", 0.30))
    torso_y2: float = float(os.getenv("CFG_VEST_TORSO_Y2", 0.70))
    # association thresholds (vest ↔ torso)
    vest_torso_iou_thr: float = float(os.getenv("CFG_VEST_TORSO_IOU_THR", 0.05))
    vest_torso_cover_thr: float = float(os.getenv("CFG_VEST_TORSO_COVER_THR", 0.30))
    center_x_frac: float = float(os.getenv("CFG_VEST_CENTER_X", 0.60))
    torso_y1: float = float(os.getenv("CFG_VEST_TORSO_Y1", 0.35))
    torso_y2: float = float(os.getenv("CFG_VEST_TORSO_Y2", 0.75))


@dataclass(frozen=True)
class RestrictedCfg:
    # consider feet point a bit above box bottom to avoid floor shadow
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

CFG = AppCfg()
