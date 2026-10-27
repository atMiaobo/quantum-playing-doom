"""Multi-button decisions: DOOM allows moving and attacking simultaneously."""
from __future__ import annotations

# Each macro-action is held for a small number of engine tics.
ACTION_SETS: tuple[tuple[str, ...], ...] = (
    ("MOVE_FORWARD",),                  # 0
    ("MOVE_FORWARD", "ATTACK"),        # 1
    ("TURN_LEFT", "ATTACK"),          # 2
    ("TURN_RIGHT", "ATTACK"),         # 3
    ("MOVE_LEFT", "ATTACK"),          # 4
    ("MOVE_RIGHT", "ATTACK"),         # 5
    ("ATTACK",),                       # 6
    ("MOVE_BACKWARD", "ATTACK"),      # 7
    ("TURN_LEFT", "MOVE_FORWARD"),    # 8
    ("TURN_RIGHT", "MOVE_FORWARD"),   # 9
)
ACTION_NAMES = (
    "advance", "advance+fire", "turn left+fire", "turn right+fire", "strafe left+fire",
    "strafe right+fire", "fire", "retreat+fire", "advance left", "advance right",
)
NUM_ACTIONS = len(ACTION_SETS)


def action_vectors(available_buttons: list[str]) -> list[list[int]]:
    seen = set(available_buttons)
    needed = set().union(*ACTION_SETS)
    missing = needed - seen
    if missing:
        raise ValueError(f"Deadly Corridor button mismatch; missing {sorted(missing)} from {available_buttons}")
    return [[int(b in macros) for b in available_buttons] for macros in ACTION_SETS]
