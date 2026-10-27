"""Explicit *privileged simulator-state* observations for sample-efficient learning.

RGB is included alongside health/position/object information.  This is NOT
pixels-only learning.  We report the privilege transparently in the docs.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from PIL import Image

PIXEL_H, PIXEL_W = 64, 96
FEATURE_NAMES = (
    "health", "ammo", "kills", "relative_x", "relative_y", "yaw_cos", "yaw_sin",
    "goal_bearing_cos", "goal_bearing_sin", "goal_dist", "enemy_bearing_cos",
    "enemy_bearing_sin", "enemy_dist", "enemy_visible", "enemy_present",
    "enemy_count", "recent_damage", "progress_since_reset", "time_fraction",
    "prev_reward",
)
FEATURE_DIM = len(FEATURE_NAMES)


def image_to_gray(buffer: np.ndarray) -> np.ndarray:
    """HWC or CHW uint8 RGB to fixed-size uint8 grayscale."""
    b = np.asarray(buffer)
    if b.ndim != 3:
        raise ValueError(f"Expected RGB image, got shape {b.shape}")
    if b.shape[0] == 3 and b.shape[-1] != 3:
        b = np.moveaxis(b, 0, -1)
    if b.shape[-1] != 3:
        raise ValueError(f"Expected three channels, got {b.shape}")
    return np.asarray(Image.fromarray(b.astype(np.uint8), "RGB")
                      .convert("L").resize((PIXEL_W, PIXEL_H), Image.Resampling.BILINEAR),
                      dtype=np.uint8).copy()


def wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def bearing_to(px: float, py: float, angle_degrees: float,
               tx: float, ty: float) -> tuple[float, float]:
    delta = wrap_pi(math.atan2(ty - py, tx - px) - math.radians(angle_degrees))
    return delta, math.hypot(tx - px, ty - py)


def cls_category(obj) -> str:
    return str(getattr(obj, "category", getattr(obj, "object_category", "")) or "").lower()


def cls_name(obj) -> str:
    return str(getattr(obj, "name", getattr(obj, "object_name", "")) or "").lower()


def is_monster(obj) -> bool:
    cat, name = cls_category(obj), cls_name(obj)
    if cat and cat not in ("other", "unknown", "none"):
        return "monster" in cat
    return any(n in name for n in (
        "zombie", "imp", "demon", "caco", "revent", "baron", "knight", "sergeant",
        "former", "shotgun", "chaingun", "spectre", "hell", "trooper",
    ))


def is_goal(obj) -> bool:
    cat, name = cls_category(obj), cls_name(obj)
    if cat and cat not in ("other", "unknown", "none"):
        return "armor" in cat
    return any(n in name for n in ("armor", "vest", "greenarmor"))


def position(obj) -> tuple[float, float]:
    return (float(getattr(obj, "position_x", getattr(obj, "object_position_x", 0.0))),
            float(getattr(obj, "position_y", getattr(obj, "object_position_y", 0.0))))


@dataclass
class Observation:
    pixels: np.ndarray             # uint8 [64,96]
    features: np.ndarray           # float32 [20]
    frame: np.ndarray              # uint8 [H,W,3]
    context: dict                  # non-learning diagnostic details


def compose_observation(*, frame: np.ndarray, variables: dict,
                        objects: list, labels: list, start_x: float,
                        start_y: float, last_health: float,
                        previous_reward: float, steps: int, max_steps: int) -> Observation:
    """Pure construction: can be unit-tested without ViZDoom installed."""
    px = variables["POSITION_X"]
    py = variables["POSITION_Y"]
    angle = variables["ANGLE"]
    health = variables["HEALTH"]
    enemies = [o for o in (objects or []) if is_monster(o)]
    goals = [o for o in (objects or []) if is_goal(o)]
    visible_ids = {int(l.object_id) for l in (labels or []) if is_monster(l)}
    visible_enemy_positions = [position(o) for o in enemies if int(getattr(o, "id", -1)) in visible_ids]
    # Selection: prefer a visible target, then the closest one (world-state access).
    enemy_candidates = [o for o in enemies if int(getattr(o, "id", -1)) in visible_ids]
    if not enemy_candidates:
        enemy_candidates = enemies
    enemy = min(enemy_candidates, key=lambda o: math.dist(position(o), (px,py))) if enemy_candidates else None
    enemy_bearing, enemy_distance = bearing_to(px, py, angle, *position(enemy)) if enemy else (0.0, 1000.0)
    enemy_visible = bool(enemy is not None and int(getattr(enemy, "id", -1)) in visible_ids)
    if goals:
        goal = min(goals, key=lambda o: math.dist(position(o), (px,py)))
        goal_x, goal_y = position(goal)
        goal_bearing, goal_distance = bearing_to(px, py, angle, goal_x, goal_y)
        found_goal = True
    else:
        # In the stock corridor the goal is at the far positive-X end.
        # Coordinate fallback, explicitly disclosed in the docs.
        goal_x, goal_y = start_x + 1500, start_y
        goal_bearing, goal_distance = bearing_to(px, py, angle, goal_x, goal_y)
        found_goal = False
    v = np.asarray([
        np.clip(health/100, 0, 2), np.clip(variables["SELECTED_WEAPON_AMMO"]/50, 0, 3),
        np.clip(variables["KILLCOUNT"]/6, 0, 2),
        np.clip((px-start_x)/1500, -2,2), np.clip((py-start_y)/400,-2,2),
        math.cos(math.radians(angle)), math.sin(math.radians(angle)),
        math.cos(goal_bearing), math.sin(goal_bearing), np.clip(goal_distance/1500, 0, 3),
        math.cos(enemy_bearing) if enemy is not None else 0.0,
        math.sin(enemy_bearing) if enemy is not None else 0.0,
        np.clip(enemy_distance/800, 0, 2), float(enemy_visible), float(enemy is not None),
        np.clip(len(enemies)/6, 0, 2), np.clip((last_health-health)/50, -1, 2),
        np.clip((px-start_x)/1500, -2,2),
        np.clip(steps/max(1,max_steps),0,2), np.clip(previous_reward/50,-3,3),
    ], dtype=np.float32)
    return Observation(image_to_gray(frame), v, frame, {
        "enemy_count": len(enemies), "enemy_visible": enemy_visible,
        "enemy_bearing_deg": round(math.degrees(enemy_bearing),1),
        "enemy_distance": round(enemy_distance,1),
        "goal_bearing_deg": round(math.degrees(goal_bearing),1),
        "goal_distance": round(goal_distance,1), "goal_visible_in_objects": found_goal,
        "visible_enemy_count": len(visible_enemy_positions),
        "position": (round(px,1), round(py,1)), "yaw": round(angle,1),
        "health": health, "kills": int(variables["KILLCOUNT"]),
    })
