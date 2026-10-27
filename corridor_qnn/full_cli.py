"""Bounded full-game experiments with explicit episode seeds and JSON artifacts."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from .full_environment import FullDoom, FEATURE_NAMES
from .full_actions import ACTIONS, ACTION_NAMES
from .full_navigation import map_name
from .full_models import FullActorCritic, load_full, save_full, transfer_corridor
from .full_teacher import teacher_action
from .training import seed_everything, infer

ROOT=Path('outputs/full_game')


def add_commands(sp):
    for command in ('full-doctor','full-eval','full-train','full-campaign'):
        p=sp.add_parser(command,help='Full Doom/Freedoom levels (privileged navigation)')
        p.add_argument('--map',default='MAP01')
        p.add_argument('--skill',type=int,choices=range(1,6),default=1)
        p.add_argument('--iwad',type=Path)
        p.add_argument('--secret-maps',action='store_true')
        p.add_argument('--frame-skip',type=int,default=4)
        p.add_argument('--max-steps',type=int,default=1000)
        p.add_argument('--seed',type=int,default=800 if command!='full-train' else 42)
        p.add_argument('--output-dir',type=Path,default=ROOT)
        if command in ('full-eval','full-campaign'):
            p.add_argument('--policy',choices=('random','teacher','quantum','classical'),default='quantum')
            p.add_argument('--checkpoint',type=Path)
            p.add_argument('--episodes',type=int,default=20 if command=='full-eval' else 1)
            p.add_argument('--record',type=Path,help='GIF of first episode, captured across its full duration')
            p.add_argument('--demo',type=Path,help='Native ViZDoom .lmp recording of first episode')
            p.add_argument('--stochastic',action='store_true')
        if command=='full-campaign':
            p.add_argument('--start-map',default='MAP01')
            p.add_argument('--end-map',default='MAP30')
        if command=='full-train':
            p.add_argument('--model',choices=('quantum','classical'),default='quantum')
            p.add_argument('--decisions',type=int,default=6000)
            p.add_argument('--epochs',type=int,default=12)
            p.add_argument('--batch-size',type=int,default=64)
            p.add_argument('--lr',type=float,default=.001)
            p.add_argument('--checkpoint',type=Path,help='Resume full-game weights')
            p.add_argument('--output-checkpoint',type=Path)
            p.add_argument('--transfer-corridor',type=Path)
            p.add_argument('--teacher-prob',type=float,default=1.)
            p.add_argument('--validation-episodes',type=int,default=3)
            p.add_argument('--validation-seed',type=int,default=500)
            p.add_argument('--dataset',type=Path,help='Reuse saved demonstrations; collects no new decisions')


def environment(args, map=None):
    return FullDoom(map=map or args.map,skill=args.skill,frame_skip=args.frame_skip,
                    max_steps=args.max_steps,seed=args.seed,iwad=args.iwad,secret_maps=args.secret_maps)


def json_write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def evaluate(args, model=None, seeds=None, write=True):
    supplied_model=model is not None
    if args.episodes<1: raise ValueError('episodes must be positive')
    if args.record and args.record.suffix.lower()!='.gif':
        raise ValueError('--record supports GIF; use a .gif path')
    seed_everything(args.seed); torch.set_num_threads(1)
    seeds=list(seeds) if seeds is not None else list(range(args.seed,args.seed+args.episodes))
    if model is None and args.policy in ('quantum','classical'):
        model=load_full(args.checkpoint or args.output_dir/'models'/f'{args.policy}.pt',args.policy)
    if model is not None: model.eval()
    records=[]; frames=[]; started=time.perf_counter(); inference_seconds=0.
    with environment(args) as env:
        for ep,seed in enumerate(seeds):
            rng=np.random.default_rng(seed)
            obs=env.reset(seed=seed,demo=args.demo if ep==0 else None)
            start=time.perf_counter()
            for step in range(args.max_steps):
                if args.record and ep==0 and step%max(1,args.max_steps//400)==0:
                    frame=Image.fromarray(obs.frame.copy())
                    ImageDraw.Draw(frame).text((4,4),f'{args.policy} {args.map} seed {seed} step {step}',fill='yellow',stroke_width=1,stroke_fill='black')
                    frames.append(frame)
                t=time.perf_counter()
                if args.policy=='random': action=int(rng.integers(len(ACTIONS)))
                elif args.policy=='teacher': action=teacher_action(obs.features)
                else: action=infer(model,obs,deterministic=not args.stochastic)
                inference_seconds+=time.perf_counter()-t
                obs,_,done,info=env.step(action)
                if done: break
            info.update(seed=seed,wall_seconds=time.perf_counter()-start)
            records.append(info)
        identity=dict(iwad=str(env.iwad),iwad_sha256=env.iwad_hash)
    if args.record and frames:
        args.record.parent.mkdir(parents=True,exist_ok=True)
        interval=max(1,args.max_steps//400)*args.frame_skip*1000/35
        frames[0].save(args.record,format='GIF',save_all=True,append_images=frames[1:],duration=round(interval),loop=0)
    metrics=('kills','survival_seconds','coverage_cells','items','pickup_events','health','raw_reward','shaped_reward','steps','max_displacement','wall_seconds')
    summary=dict(policy=args.policy,map=args.map,skill=args.skill,seeds=seeds,episodes=len(seeds),
                 max_steps=args.max_steps,frame_skip=args.frame_skip,**identity,
                 completion_rate=float(np.mean([r['completed'] for r in records])),
                 death_rate=float(np.mean([r['death'] for r in records])),
                 **{'mean_'+k:float(np.mean([r[k] for r in records])) for k in metrics},
                 inference_seconds=inference_seconds,total_wall_seconds=time.perf_counter()-started)
    import importlib.metadata
    summary['runtime']=dict(torch=str(torch.__version__),vizdoom=importlib.metadata.version('vizdoom'),
                            torch_threads=torch.get_num_threads())
    if args.policy in ('quantum','classical'):
        import hashlib
        checkpoint=args.checkpoint or args.output_dir/'models'/f'{args.policy}.pt'
        summary['checkpoint']='in_memory_candidate' if supplied_model else str(checkpoint)
        summary['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if not supplied_model and Path(checkpoint).is_file() else None
    result=dict(summary=summary,episodes=records)
    if write:
        json_write(args.output_dir/f'{args.map}_{args.policy}.json',result)
        print('SUMMARY',json.dumps(summary),flush=True)
    return result


def doctor(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with environment(args) as env:
        obs=env.reset()
        Image.fromarray(obs.frame).save(args.output_dir/f'{args.map}_screen.png')
        Image.fromarray(obs.context['automap']).save(args.output_dir/f'{args.map}_automap.png')
        initial=env.v.copy(); outcomes=[]
        for i in range(len(ACTIONS)):
            _,_,done,info=env.step(i); outcomes.append(dict(action=ACTION_NAMES[i],**info))
            if done: break
        result=dict(map=args.map,iwad=str(env.iwad),iwad_sha256=env.iwad_hash,
                    feature_names=FEATURE_NAMES,buttons=[b.name for b in env.game.get_available_buttons()],
                    initial_variables=initial,exit_linedefs=[p.tolist() for p in env.navigator.exits],
                    action_smoke=outcomes,continuous_campaign=False)
    json_write(args.output_dir/f'{args.map}_doctor.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='action_smoke'},indent=2))
    print(f'Full-game smoke OK: {len(outcomes)} macro actions executed')


def train(args):
    from types import SimpleNamespace
    import copy
    if args.decisions<1 or args.decisions>10000 or args.epochs<1 or args.batch_size<1:
        raise ValueError('decisions must be 1..10000, epochs and batch size positive')
    if not 0<=args.teacher_prob<=1: raise ValueError('teacher-prob must be 0..1')
    if args.validation_episodes<1: raise ValueError('validation-episodes must be positive')
    seed_everything(args.seed); torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output_checkpoint or args.output_dir/'models'/f'{args.model}.pt'
    if args.checkpoint and Path(checkpoint).resolve()==args.checkpoint.resolve():
        raise ValueError('Use a separate output checkpoint to preserve the input candidate')
    model=load_full(args.checkpoint,args.model) if args.checkpoint else FullActorCritic(args.model)
    transferred=transfer_corridor(model,args.transfer_corridor) if args.transfer_corridor else []
    started=time.perf_counter(); rng=np.random.default_rng(args.seed)
    dataset_path=args.output_dir/f'demonstrations_{args.seed}.npz'
    collection=[]
    if args.dataset:
        data=np.load(args.dataset,allow_pickle=False)
        images,features,labels,episode_ids=(data[k] for k in ('images','features','labels','episode_ids'))
        if features.shape[1]!=len(FEATURE_NAMES) or images.shape[1:]!=(64,96) or not (len(images)==len(features)==len(labels)==len(episode_ids)):
            raise ValueError('Incompatible full-game demonstration dimensions')
        if np.any(labels<0) or np.any(labels>=len(ACTIONS)):
            raise ValueError('Invalid demonstration action labels')
        collected=0
    else:
        images=[]; features=[]; labels=[]; episode_ids=[]
        with environment(args) as env:
            ep=0; obs=env.reset(seed=args.seed)
            for i in range(args.decisions):
                label=teacher_action(obs.features)
                images.append(obs.pixels); features.append(obs.features); labels.append(label); episode_ids.append(ep)
                action=label if rng.random()<args.teacher_prob else infer(model,obs)
                obs,_,done,info=env.step(action)
                if done or i+1==args.decisions:
                    collection.append(dict(seed=args.seed+ep,**info))
                    print(f'COLLECT decisions={i+1} episode={ep} kills={info["kills"]} cells={info["coverage_cells"]} outcome={info["outcome"]}',flush=True)
                    ep+=1
                    if i+1<args.decisions: obs=env.reset(seed=args.seed+ep)
        images=np.stack(images); features=np.stack(features); labels=np.asarray(labels); episode_ids=np.asarray(episode_ids)
        np.savez_compressed(dataset_path,images=images,features=features,labels=labels,episode_ids=episode_ids)
        collected=args.decisions
    if len(set(episode_ids.tolist()))<2: raise ValueError('Need at least two episodes for episode-held-out validation')
    valid=episode_ids==max(episode_ids)
    train_idx=torch.from_numpy(np.flatnonzero(~valid)); val_idx=torch.from_numpy(np.flatnonzero(valid))
    x=torch.from_numpy(images); f=torch.from_numpy(features); y=torch.from_numpy(labels).long()
    counts=np.bincount(labels[~valid],minlength=len(ACTIONS))
    weights=torch.tensor(np.where(counts>0,np.sqrt(counts.max()/np.maximum(counts,1)),0),dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    initial_quantum=model.core.weights.detach().clone() if args.model=='quantum' else None
    metadata=dict(training='full-game behavior cloning',collected_decisions=collected,samples=len(labels),
                  map=args.map,skill=args.skill,seed=args.seed,validation_seed=args.validation_seed,
                  transferred_keys=transferred,collection=collection,history=[],
                  dataset=str(args.dataset or dataset_path), validation_episodes=args.validation_episodes)
    validation_args=SimpleNamespace(**vars(args))
    validation_args.policy=args.model; validation_args.episodes=args.validation_episodes
    validation_args.record=None; validation_args.demo=None; validation_args.stochastic=False
    best_score=None; best_state=None
    # Select by real held-out gameplay at two bounded checkpoints, not loss alone.
    validation_epochs={max(1,args.epochs//2),args.epochs}
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.
        for idx in train_idx[torch.randperm(len(train_idx))].split(args.batch_size):
            logits,_=model(x[idx],f[idx])
            loss=torch.nn.functional.cross_entropy(logits,y[idx],weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2); optimizer.step()
            total+=loss.item()*len(idx)
        model.eval(); correct=0
        with torch.no_grad():
            for idx in val_idx.split(args.batch_size):
                logits,_=model(x[idx],f[idx]); correct+=(logits.argmax(-1)==y[idx]).sum().item()
        row=dict(epoch=epoch,loss=total/len(train_idx),episode_heldout_accuracy=correct/len(val_idx))
        if epoch in validation_epochs:
            result=evaluate(validation_args,model=model,seeds=range(args.validation_seed,args.validation_seed+args.validation_episodes),write=False)
            s=result['summary']; row['gameplay']=s
            score=(s['completion_rate'],s['mean_coverage_cells']+2*s['mean_kills'])
            if best_score is None or score>best_score:
                best_score=score; best_state=copy.deepcopy(model.state_dict()); metadata['selected_epoch']=epoch
        metadata['history'].append(row)
        print('EPOCH',json.dumps(row),flush=True)
        torch.set_num_threads(2)
    metadata['training_wall_seconds']=time.perf_counter()-started
    model.load_state_dict(best_state)
    if initial_quantum is not None:
        metadata['quantum_parameter_l2_change']=float(torch.linalg.vector_norm(model.core.weights.detach()-initial_quantum))
    save_full(checkpoint,model,metadata)
    json_write(args.output_dir/f'training_{args.model}_{args.seed}.json',metadata)
    print(f'SAVED {checkpoint} collected={collected} samples={len(labels)}',flush=True)


def campaign(args):
    start=int(map_name(args.start_map,args.secret_maps)[3:]); end=int(map_name(args.end_map,args.secret_maps)[3:])
    if start>end: raise ValueError('start-map must precede end-map')
    results=[]
    for i in range(start,end+1):
        args.map=f'MAP{i:02d}'
        result=evaluate(args); results.append(result)
        # Progress only after a genuine exit; fresh inventory on each initialized map.
        if result['summary']['completion_rate']<1: break
        args.record=None; args.demo=None
    json_write(args.output_dir/'campaign.json',dict(mode='independent_maps_fresh_inventory',maps=results))


def run(args):
    {'full-doctor':doctor,'full-eval':evaluate,'full-train':train,'full-campaign':campaign}[args.cmd](args)
