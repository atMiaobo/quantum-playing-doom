"""Headless full IWAD levels, separate from the legacy corridor environment."""
from pathlib import Path
from collections import deque
import hashlib
import math
import numpy as np
from .features import Observation, image_to_gray, bearing_to, is_monster, position
from .full_actions import ACTIONS, BUTTONS, action_vectors
from .full_navigation import Navigator, LegacyNavigator, map_name

BASE_FEATURES = (
    'health','ammo','armor','weapon','kills','items','yaw_cos','yaw_sin',
    'waypoint_cos','waypoint_sin','waypoint_distance','interaction_distance',
    'enemy_cos','enemy_sin','enemy_distance','enemy_visible',
    'depth_left','depth_midleft','depth_front','depth_midright','depth_right',
    'speed','stuck','coverage','time','damage','dx','dy',
    'previous_waypoint_sin','previous_enemy_sin','previous_enemy_visible',
    'has_shotgun','has_chaingun',
)
FEATURE_NAMES = BASE_FEATURES + tuple('last_'+str(i) for i in range(len(ACTIONS)))
FEATURE_DIM = len(FEATURE_NAMES)
VARIABLES = ('POSITION_X','POSITION_Y','ANGLE','HEALTH','ARMOR','KILLCOUNT',
             'ITEMCOUNT','SECRETCOUNT','SELECTED_WEAPON','SELECTED_WEAPON_AMMO',
             'WEAPON2','WEAPON3','WEAPON4','DAMAGECOUNT','HITCOUNT') + tuple(f'AMMO{i}' for i in range(10))


def terminal_status(finished, dead, engine_timeout, step_limit, exit_event):
    """Exit event comes exclusively from native map_exit_reward (all others zero)."""
    if dead: return 'death'
    if finished and exit_event: return 'completed'
    if engine_timeout or step_limit: return 'timeout'
    return 'unknown_terminal' if finished else 'running'


def shape_reward(before, after, new_cell, outcome):
    return (0.15*new_cell + 3*max(0, after['KILLCOUNT']-before['KILLCOUNT'])
            + .4*max(0, after['ITEMCOUNT']-before['ITEMCOUNT'])
            + .03*max(0, after['HEALTH']-before['HEALTH'])
            + .01*max(0, after['ARMOR']-before['ARMOR'])
            + .05*max(0, after['HITCOUNT']-before['HITCOUNT'])
            - .02*max(0, before['HEALTH']-after['HEALTH']) - .005
            + (50 if outcome=='completed' else -5 if outcome=='death' else 0))


class FullDoom:
    def __init__(self, map='MAP01', skill=1, frame_skip=4, max_steps=1000,
                 seed=42, iwad=None, secret_maps=False, visible=False, navigation='improved'):
        import vizdoom as vzd
        self._settings=dict(map=map,skill=skill,frame_skip=frame_skip,max_steps=max_steps,
                            seed=seed,iwad=iwad,secret_maps=secret_maps,visible=visible,navigation=navigation)
        self._recording=False
        self.map = map_name(map, secret_maps)
        if not 1 <= skill <= 5 or min(frame_skip,max_steps)<1:
            raise ValueError('skill must be 1..5 and frame_skip/max_steps positive')
        self.vzd=vzd; self.skill=skill; self.frame_skip=frame_skip
        self.max_steps=max_steps; self.seed=seed
        self.iwad = Path(iwad) if iwad else Path(vzd.install_path)/'freedoom2.wad'
        if not self.iwad.is_file():
            raise FileNotFoundError(f'IWAD absent: {self.iwad}; supply --iwad assets/freedoom2.wad')
        self.navigation=navigation
        if navigation not in ('legacy','improved'): raise ValueError('Unknown navigation version')
        self.navigator=(LegacyNavigator if navigation=='legacy' else Navigator)(self.iwad,self.map)
        self.iwad_hash=hashlib.sha256(self.iwad.read_bytes()).hexdigest()
        self.game=vzd.DoomGame()
        try:
            self.game.load_config(str(Path(vzd.scenarios_path)/'freedoom2.cfg'))
            self.game.set_doom_game_path(str(self.iwad.resolve()))
            self.game.set_doom_map(self.map)
            self.game.set_doom_skill(skill)
            self.game.set_available_buttons([getattr(vzd.Button,b) for b in BUTTONS])
            self.game.set_available_game_variables([getattr(vzd.GameVariable,v) for v in VARIABLES])
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
            self.game.set_labels_buffer_enabled(True)
            self.game.set_objects_info_enabled(True)
            self.game.set_sectors_info_enabled(navigation=='improved')
            self.game.set_depth_buffer_enabled(True)
            self.game.set_automap_buffer_enabled(True)
            self.game.set_audio_buffer_enabled(False)
            self.game.set_sound_enabled(False)
            self.game.set_window_visible(visible)
            self.game.set_render_hud(False)
            self.game.set_episode_timeout(max_steps*frame_skip+1)
            # Isolated native exit event; training shaping is calculated separately.
            for setting in ('living_reward','death_penalty','kill_reward','item_reward',
                            'secret_reward','hit_reward','hit_taken_penalty',
                            'damage_made_reward','damage_taken_penalty','health_reward','armor_reward'):
                getattr(self.game,'set_'+setting)(0)
            self.game.set_map_exit_reward(1)
            self.game.set_seed(seed)
            self.game.init()
        except Exception:
            self.game.close(); raise
        self.actions=action_vectors([b.name for b in self.game.get_available_buttons()])

    def variables(self):
        return {n:float(self.game.get_game_variable(getattr(self.vzd.GameVariable,n))) for n in VARIABLES}

    def reset(self, seed=None, demo=None):
        # ViZDoom 1.3.2 can leave subsequent episodes frozen when a native
        # recording is interrupted at our decision cap. Isolate that session.
        if self._recording:
            settings=self._settings.copy()
            self.close()
            self.__init__(**settings)
        self._recording=bool(demo)
        self.game.set_seed(self.seed if seed is None else int(seed))
        if demo:
            Path(demo).parent.mkdir(parents=True,exist_ok=True)
        self.game.new_episode(str(demo) if demo else '')
        self.steps=0; self.last_action=-1; self.reward=0.; self.native_reward=0.
        self.visited=set(); self.positions=deque(maxlen=9)
        self.previous=np.zeros(FEATURE_DIM,dtype=np.float32)
        self.navigator.reset()
        self.v=self.variables(); self.last_v=dict(self.v)
        self.start=np.array([self.v['POSITION_X'],self.v['POSITION_Y']])
        self.peak_distance=0.; self.pickups=0; self.done=False
        return self.observe()

    def observe(self):
        state=self.game.get_state()
        if state is None: raise RuntimeError('Missing nonterminal full-game state')
        v=self.v
        xy=np.array([v['POSITION_X'],v['POSITION_Y']])
        cell=tuple(np.floor(xy/64).astype(int))
        self.visited.add(cell)
        self.positions.append(xy.copy())
        if self.navigation=='legacy': target,interaction=self.navigator.update(xy)
        else: target,interaction=self.navigator.update(xy,v['ANGLE'],self.last_action,bool(self.previous[15]))
        wb,wd=bearing_to(*xy,v['ANGLE'],*target)
        labels=[l for l in state.labels if is_monster(l) and l.width*l.height>=6]
        # Visible engine labels only; objects behind walls never become combat targets.
        enemy=min(labels,key=lambda l: math.dist(xy,position(l))) if labels else None
        eb,ed=bearing_to(*xy,v['ANGLE'],*position(enemy)) if enemy else (0.,1000.)
        depth=np.asarray(state.depth_buffer)
        rays=[float(np.median(depth[90:150,max(0,int(x)-5):int(x)+5]))/255 for x in (32,96,160,224,288)]
        delta=xy-self.positions[-2] if len(self.positions)>1 else np.zeros(2)
        stuck=len(self.positions)==9 and np.linalg.norm(xy-self.positions[0])<12
        values=[v['HEALTH']/100,v['SELECTED_WEAPON_AMMO']/50,v['ARMOR']/100,v['SELECTED_WEAPON']/9,
                v['KILLCOUNT']/20,v['ITEMCOUNT']/20,math.cos(math.radians(v['ANGLE'])),math.sin(math.radians(v['ANGLE'])),
                math.cos(wb),math.sin(wb),min(wd/256,4),min(interaction/128,4),
                math.cos(eb) if enemy else 0,math.sin(eb) if enemy else 0,min(ed/800,3),float(enemy is not None),
                *rays,min(np.linalg.norm(delta)/20,3),float(stuck),len(self.visited)/100,
                self.steps/self.max_steps,(self.last_v['HEALTH']-v['HEALTH'])/50,*np.clip(delta/32,-3,3),
                self.previous[9],self.previous[13],self.previous[15],float(v['WEAPON3']>0),float(v['WEAPON4']>0)]
        f=np.asarray(values+[float(i==self.last_action) for i in range(len(ACTIONS))],dtype=np.float32)
        if len(f)!=FEATURE_DIM or not np.isfinite(f).all(): raise RuntimeError('Invalid full observation')
        self.previous=f.copy()
        frame=np.asarray(state.screen_buffer).copy()
        context=dict(position=xy.tolist(),yaw=v['ANGLE'],kills=v['KILLCOUNT'],
                     waypoint=target.tolist(),enemy_visible=bool(enemy),depth=rays,
                     automap=np.asarray(state.automap_buffer).copy())
        return Observation(image_to_gray(frame),f,frame,context)

    def step(self, action):
        if self.done: raise RuntimeError('Call reset after episode end')
        if not isinstance(action,(int,np.integer)) or not 0<=action<len(ACTIONS):
            raise ValueError('Invalid full-game action')
        before=dict(self.v)
        before_tics=self.game.get_episode_time()
        raw=float(self.game.make_action(self.actions[action],self.frame_skip))
        self.steps+=1; self.last_action=action
        finished=self.game.is_episode_finished()
        current_tics=self.game.get_episode_time()
        if not finished and current_tics<=before_tics:
            raise RuntimeError('ViZDoom failed to advance game time; episode state is invalid')
        outcome=terminal_status(finished,self.game.is_player_dead(),self.game.is_episode_timeout_reached(),
                                self.steps>=self.max_steps,raw>=1.)
        self.done=outcome!='running'
        self.last_v=before; self.v=self.variables()
        # The engine destroys/resets map counters on exit. Preserve the last
        # valid sample; final-action counters/time are bounded by frame_skip.
        if outcome=='completed': self.v=before.copy()
        xy=np.array([self.v['POSITION_X'],self.v['POSITION_Y']])
        self.peak_distance=max(self.peak_distance,float(np.linalg.norm(xy-self.start)))
        old_count=len(self.visited)
        self.visited.add(tuple(np.floor(xy/64).astype(int)))
        # Additional pickup events include ammo/health/armor that ITEMCOUNT excludes.
        self.pickups+=int(any(self.v[n]>before[n] for n in ('HEALTH','ARMOR','WEAPON2','WEAPON3','WEAPON4')+tuple(f'AMMO{i}' for i in range(10))))
        r=shape_reward(before,self.v,len(self.visited)-old_count,outcome)
        self.reward+=r; self.native_reward+=raw
        obs=None if finished else self.observe()
        info=dict(map=self.map,skill=self.skill,steps=self.steps,outcome=outcome,
                  completed=outcome=='completed',death=outcome=='death',timeout=outcome=='timeout',
                  kills=self.v['KILLCOUNT'],items=self.v['ITEMCOUNT'],pickup_events=self.pickups,
                  health=max(0,self.v['HEALTH']),secrets=self.v['SECRETCOUNT'],
                  survival_seconds=(before_tics if outcome=='completed' else current_tics)/35,
                  coverage_cells=len(self.visited),coverage_cell_size=64,
                  max_displacement=self.peak_distance,raw_reward=self.native_reward,
                  shaped_reward=self.reward)
        return obs,r,self.done,info

    def close(self): self.game.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
