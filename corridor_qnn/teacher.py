"""A transparent geometric heuristic demonstrator (NOT a trained network).

Uses the *same structured simulator-state features* the QNN sees. This is a
curriculum / demonstration generator, not claimed to be an expert Doom bot.
"""
from __future__ import annotations
import math
import numpy as np


def teacher_action(features: np.ndarray, step: int = 0) -> int:
    f = np.asarray(features)
    enemy_exists = bool(f[14] > 0.5)
    visible = bool(f[13] > 0.5)
    enemy_distance = f[12] * 800
    eb = math.atan2(float(f[11]), float(f[10]))
    gb = math.atan2(float(f[8]), float(f[7]))
    # Only chase visible combat targets; hidden object positions are still
    # present for learning features but do not stall navigation.
    if enemy_exists and visible and enemy_distance < 550:
        if eb > 0.17: return 2  # turn left + shoot
        if eb < -0.17: return 3 # turn right + shoot
        if enemy_distance < 125:
            return 4 if (step//10)%2 == 0 else 5  # dodge while shooting
        return 1 if enemy_distance > 280 else 6  # advance & fire, or aim & fire
    if gb > 0.20: return 8  # turn toward the vest while advancing
    if gb < -0.20: return 9
    # Shoot opportunistically while advancing when enemies exist nearby.
    return 1 if enemy_exists and enemy_distance < 420 else 0
