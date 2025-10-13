from __future__ import annotations
import numpy as np
import cv2

def mask_hi_vis(bgr: np.ndarray) -> np.ndarray:
    """Return mask for neon yellow/green + orange (typical vests)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # neon yellow/green band
    lo_y = np.array([20, 120, 120], dtype=np.uint8)
    hi_y = np.array([45, 255, 255], dtype=np.uint8)

    # orange band
    lo_o = np.array([5, 120, 120], dtype=np.uint8)
    hi_o = np.array([18, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lo_y, hi_y)
    m2 = cv2.inRange(hsv, lo_o, hi_o)
    mask = cv2.bitwise_or(m1, m2)
    # light morph to fill small holes
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

def mask_red(bgr: np.ndarray) -> np.ndarray:
    """Robust red (wraps hue around 0)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1, hi1 = np.array([0, 110, 110]),  np.array([10, 255, 255])
    lo2, hi2 = np.array([170,110,110]),  np.array([180,255,255])
    m1 = cv2.inRange(hsv, lo1, hi1)
    m2 = cv2.inRange(hsv, lo2, hi2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask
