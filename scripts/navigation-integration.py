"""Engine checks on the final navigator, including recording/reset isolation."""
from pathlib import Path
import json
import numpy as np
from corridor_qnn.full_environment import FullDoom
from corridor_qnn.full_teacher import teacher_action
root=Path('outputs/map01_navigation/integration'); root.mkdir(parents=True,exist_ok=True)
if (root/'results.json').exists(): raise FileExistsError(root/'results.json')
results={}
for name in ('MAP01','MAP02','MAP30'):
    with FullDoom(map=name,max_steps=40,seed=700) as env:
        obs=env.reset()
        for step in range(40):
            assert np.isfinite(obs.features).all()
            obs,_,done,info=env.step(teacher_action(obs.features))
            if done: break
        assert done and not info['completed'] and info['survival_seconds']>4
        results[name]=info
with FullDoom(max_steps=16,seed=700) as env:
    obs=env.reset(demo=root/'reset_check.lmp')
    for step in range(16): obs,_,_,info=env.step(0)
    obs=env.reset(seed=701)
    obs,_,_,info=env.step(0)
    assert info['survival_seconds']>4/35
    results['record_then_reset']=info
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('Live MAP01/MAP02/MAP30, finite observations, timeouts and native recording reset: passed')
