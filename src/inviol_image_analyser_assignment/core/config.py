"""
Centralized, typed configuration for the service.
Keeps all tunables (rule thresholds, color ranges, risk weights) in one place.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Tuple

# ---------- Small parsing helpers ----------

def _f(name: str, default: float) -> float:
    """Parse a float from env var `name`, falling back to `default`."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _frac(name: str, default: float) -> float:
    """Parse and clamp a fractional float in [0,1]."""
    val = _f(name, default)
    return 0.0 if val < 0.0 else 1.0 if val > 1.0 else val


def _hsv_triplet(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Parse "H,S,V" (integers) from env var `name`, clamped to OpenCV HSV ranges.
    If not set or invalid, return `default`.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        parts = [int(p.strip()) for p in raw.split(",")]
        if len(parts) != 3:
            return default
        h = max(0, min(179, parts[0]))
        s = max(0, min(255, parts[1]))
        v = max(0, min(255, parts[2]))
        return (h, s, v)
    except Exception:
        return default


# ---------- Config dataclasses ----------

@dataclass(frozen=True)
class VestCfg:
    """
    Tunables for color-based high-vis vest detection.

    The rule inspects a torso crop (vertical band of the person bounding box) and
    measures the ratio of pixels within orange/yellow HSV ranges. If the ratio is
    below `min_ratio`, it flags a breach.
    """
    # Torso crop as fractions of the person bbox height (top/bottom)
    torso_y1: float = field(default_factory=lambda: _frac("CFG_VEST_TORSO_Y1", 0.35))
    torso_y2: float = field(default_factory=lambda: _frac("CFG_VEST_TORSO_Y2", 0.75))

    # Minimum fraction of hi-vis pixels inside torso crop to consider "vest present"
    # (kept alias CFG_VEST_COVERAGE_THRESHOLD for convenience)
    min_ratio: float = field(
        default_factory=lambda: _frac("CFG_VEST_MIN_RATIO", _frac("CFG_VEST_COVERAGE_THRESHOLD", 0.10))
    )

    # Reserved / optional association thresholds if you later combine color + bbox models
    vest_torso_iou_thr: float = field(default_factory=lambda: _frac("CFG_VEST_TORSO_IOU_THR", 0.05))
    vest_torso_cover_thr: float = field(default_factory=lambda: _frac("CFG_VEST_TORSO_COVER_THR", 0.30))
    center_x_frac: float = field(default_factory=lambda: _frac("CFG_VEST_CENTER_X", 0.60))

    # HSV ranges (OpenCV ranges: H=0..179, S=0..255, V=0..255)
    orange_lower: Tuple[int, int, int] = field(
        default_factory=lambda: _hsv_triplet("CFG_VEST_ORANGE_LOWER", (2, 150, 150))
    )
    orange_upper: Tuple[int, int, int] = field(
        default_factory=lambda: _hsv_triplet("CFG_VEST_ORANGE_UPPER", (25, 255, 255))
    )
    yellow_lower: Tuple[int, int, int] = field(
        default_factory=lambda: _hsv_triplet("CFG_VEST_YELLOW_LOWER", (25, 100, 150))
    )
    yellow_upper: Tuple[int, int, int] = field(
        default_factory=lambda: _hsv_triplet("CFG_VEST_YELLOW_UPPER", (45, 255, 255))
    )

    def __post_init__(self) -> None:
        # Ensure torso_y1 <= torso_y2 and both in [0,1]
        y1 = min(self.torso_y1, self.torso_y2)
        y2 = max(self.torso_y1, self.torso_y2)
        object.__setattr__(self, "torso_y1", y1)
        object.__setattr__(self, "torso_y2", y2)


@dataclass(frozen=True)
class RestrictedCfg:
    """
    Tunables for restricted-area checks.

    We use a point slightly above the bottom of the person bbox as an approximate
    "feet" location to test inclusion in restricted polygons.
    """
    feet_offset: float = field(default_factory=lambda: _frac("CFG_RESTRICTED_FEET_OFFSET", 0.02))


@dataclass(frozen=True)
class RiskCfg:
    """
    Risk weights used to aggregate per-breach severities into an overall score.
    """
    low: int = 1
    med: int = 2
    high: int = 3


@dataclass(frozen=True)
class AppCfg:
    """Top-level configuration container."""
    vest: VestCfg = field(default_factory=VestCfg)
    restricted: RestrictedCfg = field(default_factory=RestrictedCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)


# Global, importable singleton
CFG = AppCfg()
