"""Command-line tool: all modes headless and safe for NixOS Podman."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .actions import ACTION_NAMES
from .environment import DeadlyCorridor
from .teacher import teacher_action
from .training import train_imitation, train_ppo, load_model, model_path, infer


def common(p):
    p.add_argument('--skill',type=int,choices=range(1,6),default=1,
                   help='Difficulty 1-5; official hard corridor is skill 5. Default is 1 for learning.')
    p.add_argument('--frame-skip',type=int,default=4,help='Engine tics per macro-action')
    p.add_argument('--max-steps',type=int,default=450,help='Cap on macro-actions per episode')
    p.add_argument('--seed',type=int,default=42)


def model_flags(p):
    p.add_argument('--model',choices=['quantum','classical'],default='quantum')
    p.add_argument('--qubits',type=int,default=6,choices=range(2,9))
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--checkpoint',type=Path,help='Input checkpoint; defaults to outputs/models/<model>.pt')
    p.add_argument('--output-checkpoint',type=Path,help='Training destination; defaults to outputs/models/<model>.pt')


def parser():
    p=argparse.ArgumentParser(prog='doom-corridor',
        description='Deadly Corridor: hybrid CNN + variational quantum actor-critic. See README.')
    sp=p.add_subparsers(dest='cmd',required=True)
    doc=sp.add_parser('doctor',help='Initialize REAL ViZDoom and save screenshot + control diagnostics');common(doc)
    test=sp.add_parser('test',help='Use pytest in the container')
    demo=sp.add_parser('eval',help='Evaluate teacher/quantum/classical and optional gameplay GIF');common(demo)
    demo.add_argument('--policy',choices=['teacher','quantum','classical','random'],default='quantum')
    demo.add_argument('--checkpoint',type=Path,help='Explicit quantum/classical checkpoint')
    demo.add_argument('--episodes',type=int,default=5)
    demo.add_argument('--record',default='',help='GIF path (e.g. outputs/corridor.gif)')
    demo.add_argument('--stochastic',action='store_true',help='Sample network rather than argmax')
    bc=sp.add_parser('imitate',help='Collect heuristic corridor demonstrations then supervised train');common(bc);model_flags(bc)
    bc.add_argument('--episodes',type=int,default=35)
    bc.add_argument('--epochs',type=int,default=8)
    bc.add_argument('--cap',type=int,default=12000)
    bc.add_argument('--batch-size',type=int,default=32)
    bc.add_argument('--lr',type=float,default=8e-4)
    bc.add_argument('--teacher-prob',type=float,default=1.0,help='For DAgger after first train: 0.5')
    bc.add_argument('--resume',action='store_true')
    rl=sp.add_parser('ppo',help='Online PPO fine-tuning on real corridor rewards');common(rl);model_flags(rl)
    rl.add_argument('--steps',type=int,default=25000)
    rl.add_argument('--rollout',type=int,default=256)
    rl.add_argument('--epochs',type=int,default=2)
    rl.add_argument('--batch-size',type=int,default=32)
    rl.add_argument('--lr',type=float,default=1e-4)
    rl.add_argument('--gamma',type=float,default=.99)
    rl.add_argument('--gae-lambda',type=float,default=.95)
    rl.add_argument('--clip',type=float,default=.2)
    rl.add_argument('--entropy-coef',type=float,default=.01)
    rl.add_argument('--value-coef',type=float,default=.5)
    cmp=sp.add_parser('compare',help='Evaluate quantum, classical and teacher for same difficulty');common(cmp)
    cmp.add_argument('--quantum-checkpoint',type=Path)
    cmp.add_argument('--classical-checkpoint',type=Path)
    cmp.add_argument('--episodes',type=int,default=10)
    return p


def doctor(args):
    import importlib.metadata
    print('PyTorch',torch.__version__,'| ViZDoom',importlib.metadata.version('vizdoom'))
    print('CPU torch threads',torch.get_num_threads())
    with DeadlyCorridor(skill=args.skill,frame_skip=args.frame_skip,
                        max_steps=args.max_steps,seed=args.seed) as game:
        obs=game.reset()
        path=Path('outputs/debug_corridor.png');path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(obs.frame).save(path)
        print(json.dumps(game.debug(),indent=2,default=str))
        print('Frame',obs.frame.shape,'| vector feature count',len(obs.features))
        print('Saved',path)
        # Verify combinational button actions work and environment advances.
        n,r,d,info=game.step(1)
        print('Step forward+attack: reward=',r,'done=',d,'info=',info)
        if n is None and not d:
            raise RuntimeError('Nonterminal game state unexpectedly absent')
    print('Doctor OK; full gameplay win rate still requires eval.')


def eval_one(policy,episodes,skill,frame_skip,max_steps,seed,
             record='',stochastic=False,checkpoint=None):
    model=load_model(checkpoint or model_path(policy),kind=policy) if policy in ('quantum','classical') else None
    if model is not None:model.eval()
    rng=np.random.default_rng(seed)
    frames=[]
    stats=[]
    with DeadlyCorridor(skill=skill,frame_skip=frame_skip, max_steps=max_steps,seed=seed) as env:
        for ep in range(episodes):
            obs=env.reset()
            for t in range(max_steps):
                if record and ep==0 and t%2==0 and len(frames)<300:
                    frame=Image.fromarray(obs.frame.copy())
                    ImageDraw.Draw(frame).text((7,7),f'{policy} | step {t} | kills {obs.context["kills"]}',
                                               fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
                    frames.append(frame.resize((480,360)))
                if policy=='teacher': action=teacher_action(obs.features,t)
                elif policy=='random': action=int(rng.integers(len(ACTION_NAMES)))
                else: action=infer(model,obs,deterministic=not stochastic)
                next_obs, reward, done, info=env.step(action)
                if done: break
                obs=next_obs
            if record and ep==0 and frames:
                # Briefly linger on final screen if available.
                if next_obs is not None:
                    frames.extend([Image.fromarray(next_obs.frame.copy()).resize((480,360))]*3)
            stats.append(info)
            print(f'  {policy:9s} ep {ep+1:2d}/{episodes}: '
                  f'win={int(info["won"])} kills={info["kills"]} '
                  f'health={info["health"]:.0f} raw={info["episode_raw_reward"]:.1f} '
                  f'steps={info["steps"]}',flush=True)
    if record and frames:
        out=Path(record);out.parent.mkdir(parents=True,exist_ok=True)
        frames[0].save(out,format='GIF',save_all=True,append_images=frames[1:],
            duration=85,loop=0,optimize=True)
        print('Wrote playable GIF',out)
    result={'policy':policy,'skill':skill,'episodes':episodes,
       'win_rate':sum(x['won'] for x in stats)/len(stats),
       'mean_raw_reward':float(np.mean([x['episode_raw_reward'] for x in stats])),
       'mean_kills':float(np.mean([x['kills'] for x in stats])),
       'mean_steps':float(np.mean([x['steps'] for x in stats]))}
    print('SUMMARY',json.dumps(result,sort_keys=True))
    return result


def compare(args):
    policies=['teacher']
    for name in ('classical','quantum'):
        checkpoint = getattr(args, f'{name}_checkpoint', None) or model_path(name)
        if checkpoint.exists():policies.append(name)
        elif getattr(args, f'{name}_checkpoint', None):raise FileNotFoundError(checkpoint)
        else:print(f'Missing {name} model; skipping. Run imitate --model {name}.')
    results=[eval_one(x,args.episodes,args.skill,args.frame_skip,args.max_steps,args.seed,
                     checkpoint=getattr(args, f'{x}_checkpoint', None))
             for x in policies]
    out=Path('outputs/comparison.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2))
    print('Saved',out)


def main():
    args=parser().parse_args()
    if args.cmd=='test':
        import pytest
        raise SystemExit(pytest.main(['-q','tests']))
    if args.cmd=='doctor':doctor(args)
    elif args.cmd=='imitate':train_imitation(args)
    elif args.cmd=='ppo':train_ppo(args)
    elif args.cmd=='eval':eval_one(args.policy,args.episodes,args.skill,args.frame_skip,
                                 args.max_steps,args.seed,args.record,args.stochastic,args.checkpoint)
    elif args.cmd=='compare':compare(args)

if __name__=='__main__':main()
