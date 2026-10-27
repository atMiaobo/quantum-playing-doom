"""Direct ViZDoom Deadly Corridor API, robust to .cfg action ordering."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
from .features import Observation, compose_observation
from .actions import ACTION_SETS, action_vectors


class DeadlyCorridor:
    def __init__(self, skill: int = 1, frame_skip: int = 4, seed: int = 42,
                 max_steps: int = 450, visible: bool = False):
        if not 1 <= skill <= 5:
            raise ValueError("--skill must be 1 through 5")
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip and max_steps must be positive")
        try:
            import vizdoom as vzd
        except ImportError as e:
            raise RuntimeError("ViZDoom not installed. Use ./run-nixos.sh build first.") from e
        self.vzd, self.skill, self.frame_skip = vzd, skill, frame_skip
        self.max_steps = max_steps
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
        self.game.set_doom_skill(skill)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_render_hud(False)
        self.game.set_render_weapon(True)
        self.game.set_window_visible(visible)
        self.game.set_sound_enabled(False)
        self.game.set_seed(seed)
        self.game.init()
        available = [x.name for x in self.game.get_available_buttons()]
        self.available = available
        self.actions = action_vectors(available)
        self.steps = 0
        self.start_x = self.start_y = self.last_health = 0.0
        self.prev_reward = 0.0
        self.last_stats = {}
        self.total_raw_reward = 0.0
        self.total_train_reward = 0.0
        self.win = False
        self.prev_kills = 0
        self.prev_hits = 0

    def _var(self, name: str) -> float:
        return float(self.game.get_game_variable(getattr(self.vzd.GameVariable, name)))

    def _vars(self) -> dict:
        return {k: self._var(k) for k in (
            "POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "KILLCOUNT",
            "HITCOUNT", "SELECTED_WEAPON_AMMO",
        )}

    def _observe(self, state, variables) -> Observation:
        frame = np.asarray(state.screen_buffer).copy()
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1,2,0)
        return compose_observation(frame=frame, variables=variables,
            objects=list(state.objects or []), labels=list(state.labels or []),
            start_x=self.start_x, start_y=self.start_y, last_health=self.last_health,
            previous_reward=self.prev_reward, steps=self.steps, max_steps=self.max_steps)

    def reset(self) -> Observation:
        self.game.new_episode()
        self.steps = 0
        self.prev_reward = 0.0
        self.total_raw_reward = self.total_train_reward = 0.0
        self.win = False
        v = self._vars()
        self.start_x, self.start_y = v["POSITION_X"], v["POSITION_Y"]
        self.last_health = v["HEALTH"]
        self.prev_kills = int(v["KILLCOUNT"])
        self.prev_hits = int(v["HITCOUNT"])
        state = self.game.get_state()
        if state is None:
            raise RuntimeError("ViZDoom produced no initial state")
        obs = self._observe(state, v)
        self.last_stats = obs.context
        return obs

    def step(self, action: int) -> tuple[Observation | None, float, bool, dict]:
        if not 0 <= int(action) < len(ACTION_SETS):
            raise ValueError(f"Invalid action {action}")
        before_health = self._var("HEALTH")
        raw = float(self.game.make_action(self.actions[int(action)], self.frame_skip))
        self.total_raw_reward += raw
        self.steps += 1
        terminated = bool(self.game.is_episode_finished())
        v = self._vars()
        kills = int(v["KILLCOUNT"])
        hits = int(v["HITCOUNT"])
        kill_delta = max(0, kills-self.prev_kills)
        hit_delta = max(0, hits-self.prev_hits)
        health_delta = max(0, before_health-v["HEALTH"])
        # Native reward already encourages progression; shaping makes combat more learnable.
        shaped = float(np.clip(raw/50, -15, 25) + kill_delta*2 + hit_delta*0.2
                       - health_delta*0.015 - 0.01)
        self.total_train_reward += shaped
        self.prev_reward = raw
        self.prev_kills, self.prev_hits = kills, hits
        self.win |= raw >= 500
        self.last_health = before_health
        timeout = self.steps >= self.max_steps
        done = terminated or timeout
        obs = None
        if not terminated:
            s = self.game.get_state()
            if s is not None:
                obs = self._observe(s, v)
                self.last_stats = obs.context
        info = dict(self.last_stats)
        info.update(raw_reward=raw, shaped_reward=shaped, terminal=terminated,
            timeout=timeout and not terminated, won=bool(self.win),
            kills=kills, health=v["HEALTH"], episode_raw_reward=self.total_raw_reward,
            episode_shaped_reward=self.total_train_reward, steps=self.steps,
            skill=self.skill)
        return obs, shaped, done, info

    def debug(self) -> dict:
        state = self.game.get_state()
        return {"buttons":self.available, "object_classes":
            [(getattr(o,'category',''),getattr(o,'name','')) for o in (state.objects or [])][:35],
            "visible_labels": [(getattr(o,'object_category',''),getattr(o,'object_name',''))
            for o in (state.labels or [])][:20], **self.last_stats}

    def close(self) -> None:
        self.game.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
