"""Demonstration collection, behavior cloning, and PPO fine-tuning."""
from __future__ import annotations
import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .environment import DeadlyCorridor
from .teacher import teacher_action
from .models import ActorCritic, pack


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def save_model(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kind": model.kind,
                "qubits":model.qubits, "layers":model.layers,
                "metadata": metadata or {}}, path)


def load_model(path, kind=None, qubits=6, layers=2):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Pass --checkpoint pretrained/quantum_skill3_candidate.pt, initialize with python scripts/init-checkpoints.py, or run imitate")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if kind is not None and kind != data['kind']:
        raise ValueError(f"Checkpoint contains {data['kind']} model, not {kind}")
    model = ActorCritic(data['kind'], data['qubits'], data['layers'])
    model.load_state_dict(data['state_dict'])
    return model


def model_path(kind):
    return Path('outputs') / 'models' / f'{kind}.pt'


def infer(model, obs, deterministic=True):
    x, f = pack(obs)
    with torch.no_grad():
        action, _, _ = model.act(x, f, deterministic)
    return int(action.item())


def collect_demonstrations(episodes, skill, frame_skip, max_steps, seed,
                           model=None, teacher_probability=1.0, cap=12000):
    """DAgger-like labels: action label is teacher at every visited state.

    Model may control portions of the rollout, gathering correction labels for
    off-expert trajectories. No fabricated successful demonstrations.
    """
    rng = random.Random(seed)
    images, features, labels = [], [], []
    stats = []
    total_seen = 0
    with DeadlyCorridor(skill=skill, frame_skip=frame_skip, max_steps=max_steps, seed=seed) as env:
        for episode in range(episodes):
            obs = env.reset()
            info = {}
            for t in range(max_steps):
                label = teacher_action(obs.features, t)
                total_seen += 1
                if len(labels) < cap:
                    images.append(obs.pixels)
                    features.append(obs.features)
                    labels.append(label)
                else:
                    # Reservoir sampling avoids bias toward initial episodes.
                    index = rng.randrange(total_seen)
                    if index < cap:
                        images[index], features[index], labels[index] = obs.pixels, obs.features, label
                action = label if model is None or rng.random()<teacher_probability else infer(model,obs)
                next_obs, _, done, info = env.step(action)
                if done:
                    break
                obs = next_obs
            stats.append({k: info[k] for k in ('won','kills','episode_raw_reward','steps')})
            print(f"  demo {episode+1:3d}/{episodes} | win={info.get('won')} kills={info.get('kills')} "
                  f"reward={info.get('episode_raw_reward',0):.1f} steps={info.get('steps')}",flush=True)
    if not labels:
        raise RuntimeError("No demonstration frames collected")
    return (np.stack(images), np.stack(features), np.asarray(labels,dtype=np.int64), stats)


def fit_imitation(model, dataset, epochs=8, batch_size=32, lr=8e-4,
                  seed=42, checkpoint='outputs/models/quantum.pt'):
    imgs, features, labels, _ = dataset
    device = 'cpu'
    img_t = torch.from_numpy(imgs)
    feature_t = torch.from_numpy(features)
    labels_t = torch.from_numpy(labels)
    num = len(labels)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num,generator=gen)
    split = max(1, int(num*.9))
    train_idx, valid_idx = indices[:split], indices[split:]
    if not len(valid_idx): valid_idx = train_idx
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_loss = -1.0, float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0, 0
        order = train_idx[torch.randperm(len(train_idx), generator=gen)]
        for chunk in order.split(batch_size):
            logits, _ = model(img_t[chunk],feature_t[chunk])
            loss = nn.functional.cross_entropy(logits,labels_t[chunk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total_loss += loss.item()*len(chunk)
            seen += len(chunk)
        model.eval()
        matches = 0
        val_loss = 0.0
        with torch.no_grad():
            for chunk in valid_idx.split(batch_size):
                logits,_ = model(img_t[chunk],feature_t[chunk])
                matches += (logits.argmax(-1)==labels_t[chunk]).sum().item()
                val_loss += nn.functional.cross_entropy(logits,labels_t[chunk]).item()*len(chunk)
        acc = matches/len(valid_idx)
        val_loss /= len(valid_idx)
        print(f"  epoch {epoch+1:2d}/{epochs}: train_ce={total_loss/max(1,seen):.3f} "
              f"val_ce={val_loss:.3f} val_acc={acc:.1%}",flush=True)
        if acc > best_acc or (acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = acc,val_loss
            save_model(checkpoint,model,{"training":"behavior_cloning", "samples":num,
                "val_accuracy":round(acc,4),"note":"demonstrations from heuristic, not expert wins"})
    model.load_state_dict(load_model(checkpoint).state_dict())
    return best_acc


def train_imitation(args):
    seed_everything(args.seed)
    ckpt = getattr(args, "output_checkpoint", None) or model_path(args.model)
    source = getattr(args, "checkpoint", None) or model_path(args.model)
    if args.resume or getattr(args, "checkpoint", None):
        model = load_model(source,kind=args.model)
        print(f"Continuing from {source}")
    else:
        model = ActorCritic(kind=args.model, qubits=args.qubits, layers=args.layers)
    print(f"{args.model} parameters={model.params():,} | {args.skill=} | {args.episodes=} "
          f"| {args.epochs=} | {args.teacher_prob=}")
    source_model = model if args.teacher_prob < 1 else None
    data = collect_demonstrations(args.episodes, args.skill, args.frame_skip,
        args.max_steps, args.seed, model=source_model,
        teacher_probability=args.teacher_prob, cap=args.cap)
    print(f"Collected {len(data[2])} samples. Demonstration wins="
          f"{sum(x['won'] for x in data[3])}/{len(data[3])}")
    acc = fit_imitation(model, data, args.epochs,args.batch_size,args.lr,args.seed,ckpt)
    print(f"Best held-out imitation accuracy: {acc:.1%}. Saved {ckpt}")
    print("IMPORTANT: classification accuracy is not Doom success. Run 'eval' to measure wins.")


def gae_advantages(rewards, values, dones, next_value,
                   gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n,dtype=np.float32)
    gae = 0.
    for t in reversed(range(n)):
        nonterminal = 1.0-float(dones[t])
        future_value = (next_value if t==n-1 else values[t+1])
        delta = rewards[t] + gamma*future_value*nonterminal - values[t]
        gae = delta + gamma*lam*nonterminal*gae
        advantages[t] = gae
    return advantages, advantages+np.asarray(values,dtype=np.float32)


def train_ppo(args):
    seed_everything(args.seed)
    ckpt = getattr(args, "output_checkpoint", None) or model_path(args.model)
    source = getattr(args, "checkpoint", None) or model_path(args.model)
    if getattr(args, "checkpoint", None) or source.exists():
        model = load_model(source,kind=args.model)
    else:
        model = ActorCritic(args.model,args.qubits,args.layers)
    if not source.exists():
        print("WARNING: PPO starting from random parameters. Run 'imitate' first for a faster warm start.")
    opt = torch.optim.Adam(model.parameters(),lr=args.lr)
    results = Path('outputs/ppo_history.csv'); results.parent.mkdir(exist_ok=True)
    header = not results.exists()
    f = results.open('a',newline=''); writer=csv.DictWriter(f, fieldnames=['step','skill','avg_raw_reward','last_wins','policy_loss','value_loss'])
    if header: writer.writeheader()
    episodes, global_steps = [], 0
    with DeadlyCorridor(skill=args.skill,frame_skip=args.frame_skip,
                        max_steps=args.max_steps,seed=args.seed) as env:
        obs = env.reset()
        while global_steps < args.steps:
            pixels, feats, actions, old_probs, vals, rewards, dones = [],[],[],[],[],[],[]
            for _ in range(min(args.rollout,args.steps-global_steps)):
                x, s = pack(obs)
                with torch.no_grad():
                    action, logp, v = model.act(x,s,False)
                act = int(action.item())
                pixels.append(obs.pixels)
                feats.append(obs.features)
                actions.append(act)
                old_probs.append(float(logp.item()))
                vals.append(float(v.item()))
                new_obs, r, done, info = env.step(act)
                rewards.append(r)
                dones.append(done)
                global_steps += 1
                if done:
                    episodes.append(info)
                    obs=env.reset()
                else:
                    obs=new_obs
            with torch.no_grad():
                x,s = pack(obs)
                _, nv = model(x,s)
                boot=float(nv.item())
            adv, returns = gae_advantages(rewards, vals,dones,boot,
                                           gamma=args.gamma,lam=args.gae_lambda)
            xall = torch.from_numpy(np.stack(pixels))
            sall = torch.from_numpy(np.stack(feats))
            aall = torch.tensor(actions)
            old = torch.tensor(old_probs)
            adv = torch.tensor(adv)
            ret = torch.tensor(returns)
            adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
            loss_pol, loss_val=0.,0.
            for ep in range(args.epochs):
                for idx in torch.randperm(len(actions)).split(args.batch_size):
                    logits, values = model(xall[idx], sall[idx])
                    dist = Categorical(logits=logits)
                    lp = dist.log_prob(aall[idx])
                    ratio = (lp-old[idx]).exp()
                    loss_p = -torch.min(ratio*adv[idx], torch.clamp(ratio,1-args.clip,1+args.clip)*adv[idx]).mean()
                    loss_v = 0.5*(values-ret[idx]).square().mean()
                    loss = loss_p + args.value_coef*loss_v - args.entropy_coef*dist.entropy().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), .5)
                    opt.step()
                    loss_pol,loss_val=loss_p.item(),loss_v.item()
            recent=episodes[-10:]
            mean_raw = float(np.mean([i['episode_raw_reward'] for i in recent])) if recent else float('nan')
            wins = sum(i['won'] for i in recent)
            writer.writerow(dict(step=global_steps,skill=args.skill,avg_raw_reward=mean_raw,
                                 last_wins=wins,policy_loss=loss_pol,value_loss=loss_val));f.flush()
            save_model(ckpt,model,{"training":"PPO", "steps":global_steps,
                "skill":args.skill,"episode_success_last_10":wins})
            print(f"PPO {global_steps:6d}/{args.steps} | raw={mean_raw:.1f} "
                  f"wins(last {len(recent)})={wins} | policy={loss_pol:.3f} value={loss_val:.3f}",flush=True)
    f.close()
    print(f"Saved {ckpt} and {results}; evaluate on held-out episodes next.")
