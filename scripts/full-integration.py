"""Small real-engine checks, outside training/evaluation. Run in the Podman image.

The exit test teleports to the exit linedef read from MAP01's IWAD, then uses
normal actions. This diagnostic shortcut is never used in policy evaluation.
"""
from pathlib import Path
import json
import math
import numpy as np
from corridor_qnn.full_environment import FullDoom, FEATURE_DIM

results={}
for name in ('MAP01','MAP02','MAP30','MAP31'):
    with FullDoom(map=name,secret_maps=True,max_steps=24,seed=7) as env:
        obs=env.reset()
        assert obs.features.shape==(FEATURE_DIM,)
        assert np.isfinite(obs.features).all()
        positions=[]
        for step in range(24):
            obs,_,done,info=env.step(0)
            positions.append([env.v['POSITION_X'],env.v['POSITION_Y']])
            if done: break
        assert done and info['timeout'] and not info['completed'],info
        assert max(np.linalg.norm(np.array(p)-positions[0]) for p in positions)>1
        results[name]=info
with FullDoom(max_steps=30) as env:
    env.reset(); env.game.send_game_command('kill')
    _,_,done,info=env.step(0)
    assert done and info['death'] and not info['completed'],info
    results['death']=info
# Map01 has a standard use exit. Use both sides and approach normally after warp.
with FullDoom(max_steps=120) as env:
    found=False
    for endpoint in env.navigator.exits:
        for angle in (0,90,180,270):
            env.reset()
            xy=endpoint-40*np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
            env.game.send_game_command(f'warp {xy[0]} {xy[1]}')
            obs,_,done,info=env.step(12)
            if done: continue
            for step in range(100):
                yaw=env.v['ANGLE']; target=math.degrees(math.atan2(endpoint[1]-env.v['POSITION_Y'],endpoint[0]-env.v['POSITION_X']))
                error=(target-yaw+180)%360-180
                action=2 if error>10 else 3 if error< -10 else (13 if step%2==0 else 0)
                obs,_,done,info=env.step(action)
                if done:
                    if info['completed']: found=True
                    break
            if found: break
        if found: break
    assert found, f'Exit event diagnostic failed: {info}'
    results['actual_exit_after_diagnostic_warp']=info
# Native recording must not freeze the following seeded episode.
with FullDoom(max_steps=12) as env:
    env.reset(seed=800,demo='outputs/full_game/integration_recording.lmp')
    for step in range(12): env.step(0)
    env.reset(seed=801)
    _,_,_,info=env.step(0)
    assert info['survival_seconds']>4/35
    results['recording_then_reset']=info
out=Path('outputs/full_game/integration.json'); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
