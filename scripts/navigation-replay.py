"""Verify successful native demos by playback, independently of policy inference."""
from pathlib import Path
import json
from corridor_qnn.full_environment import FullDoom
root=Path('outputs/map01_navigation')
out=root/'recording_replay.json'
if out.exists(): raise FileExistsError(out)
paths=sorted((root/'recording').glob('*.lmp'))+sorted((root/'recommended_recording').glob('*.lmp'))
if not paths: raise FileNotFoundError('No successful navigation recordings')
results=[]
for path in paths:
    with FullDoom(max_steps=1000,navigation='improved') as env:
        env.reset(); env.game.replay_episode(str(path))
        reward=0.; count=0
        while not env.game.is_episode_finished() and count<1001:
            env.game.advance_action(4)
            reward+=env.game.get_last_reward(); count+=1
        result=dict(path=str(path),playback_decisions=count,native_reward=reward,
                    finished=env.game.is_episode_finished(),dead=env.game.is_player_dead())
        assert result['finished'] and not result['dead'] and reward>=1,result
        results.append(result)
out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
