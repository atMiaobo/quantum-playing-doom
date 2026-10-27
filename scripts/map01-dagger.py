"""One conservative DAgger iteration. Run with ./run-nixos.sh python scripts/map01-dagger.py.

Collect exactly 4,000 on-policy QNN decisions, query the improved teacher, then
replay all 6,000 prior demonstrations. Train only navigation action-head rows;
freeze visual, shared, quantum and combat weights. Combat-state KL regularization
also protects competition between navigation and combat logits. No evaluation
seed is used for collection, gradient fitting or model selection.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import torch
from corridor_qnn.full_environment import FullDoom
from corridor_qnn.full_models import load_full,save_full
from corridor_qnn.full_teacher import teacher_action
from corridor_qnn.full_cli import json_write
from corridor_qnn.training import infer,seed_everything

root=Path('outputs/map01_navigation/dagger'); root.mkdir(parents=True,exist_ok=True)
source=Path('outputs/full_game/models/quantum.pt')
previous=Path('outputs/full_game/demonstrations_42.npz')
destination=Path('outputs/full_game/models/quantum_map01_dagger.pt')
if destination.exists() or (root/'corrections_100.npz').exists():
    raise FileExistsError('One-stage experiment already exists; preserve its artifacts')
# Require a useful measured teacher before collecting any additional data.
benchmark=json.loads(Path('outputs/map01_navigation/improved_navigation/teacher/MAP01_teacher.json').read_text())
original=json.loads(Path('outputs/full_game/MAP01_teacher.json').read_text())
if benchmark['summary']['completion_rate']<=original['summary']['completion_rate']:
    raise RuntimeError('Teacher did not improve; no training authorized by the signal gate')
seed_everything(100); torch.set_num_threads(1)
reference=load_full(source,'quantum'); model=copy.deepcopy(reference)
images=[]; features=[]; labels=[]; episode_ids=[]; collection=[]
started=time.perf_counter()
with FullDoom(seed=100,navigation='improved') as env:
    ep=0; obs=env.reset(seed=100)
    for i in range(4000):
        images.append(obs.pixels); features.append(obs.features)
        labels.append(teacher_action(obs.features)); episode_ids.append(ep)
        # Pure student occupancy, not a mixture with teacher executions.
        action=infer(reference,obs,deterministic=True)
        obs,_,done,info=env.step(action)
        if done or i==3999:
            collection.append(dict(seed=100+ep,student_actions=True,**info))
            ep+=1
            if i<3999: obs=env.reset(seed=100+ep)
corrections=dict(images=np.stack(images),features=np.stack(features),labels=np.array(labels),episode_ids=np.array(episode_ids))
np.savez_compressed(root/'corrections_100.npz',**corrections)
old=np.load(previous,allow_pickle=False)
data={key:np.concatenate((old[key],corrections[key]+(int(old[key].max())+1 if key=='episode_ids' else 0))) for key in corrections}
np.savez_compressed(root/'aggregate_100.npz',**data)
collection_seconds=time.perf_counter()-started
x=torch.from_numpy(data['images']); f=torch.from_numpy(data['features']); y=torch.from_numpy(data['labels']).long()
# Hold out the final correction episode, not individual neighboring frames.
valid=data['episode_ids']==data['episode_ids'].max()
train_idx=torch.from_numpy(np.flatnonzero(~valid)); val_idx=torch.from_numpy(np.flatnonzero(valid))
for parameter in model.parameters(): parameter.requires_grad_(False)
nav_rows=(0,1,2,3,4,5,12,13,14,15)
for head in (model.actor,model.classical_actor):
    for parameter in head.parameters():
        parameter.requires_grad_(True)
        mask=torch.zeros_like(parameter); mask[list(nav_rows)]=1
        parameter.register_hook(lambda grad,mask=mask:grad*mask)
optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=2e-4)
# Cache reference logits once: shared encoder and the circuit stay frozen.
reference_logits=[]
with torch.no_grad():
    for idx in torch.arange(len(y)).split(128): reference_logits.append(reference(x[idx],f[idx])[0])
reference_logits=torch.cat(reference_logits)
combat=f[:,15]>.5
history=[]; training_start=time.perf_counter(); torch.set_num_threads(2)
for epoch in range(1,5):
    model.train(); total=0.; kl_total=0.
    for idx in train_idx[torch.randperm(len(train_idx))].split(64):
        logits,_=model(x[idx],f[idx])
        ce=torch.nn.functional.cross_entropy(logits,y[idx])
        active=combat[idx]
        kl=torch.nn.functional.kl_div(torch.log_softmax(logits[active],-1),torch.softmax(reference_logits[idx][active],-1),reduction='batchmean') if active.any() else logits.sum()*0
        loss=ce+3*kl
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
        optimizer.step(); total+=ce.item()*len(idx); kl_total+=kl.item()*len(idx)
    model.eval()
    with torch.no_grad():
        logits=torch.cat([model(x[idx],f[idx])[0] for idx in val_idx.split(128)])
        accuracy=float((logits.argmax(-1)==y[val_idx]).float().mean())
    row=dict(epoch=epoch,cross_entropy=total/len(train_idx),combat_kl=kl_total/len(train_idx),correction_episode_heldout_accuracy=accuracy)
    history.append(row); print(json.dumps(row),flush=True)
# Final epoch is prespecified: no held-out gameplay tuning or second stage.
unchanged=[]
for name,value in model.state_dict().items():
    if name.startswith(('actor.','classical_actor.')):
        combat_rows=[i for i in range(21) if i not in nav_rows]
        assert torch.equal(value[combat_rows],reference.state_dict()[name][combat_rows])
    else:
        assert torch.equal(value,reference.state_dict()[name]); unchanged.append(name)
meta=dict(training='single DAgger iteration; frozen encoder/circuit/combat head rows',
    collected_decisions=4000,previous_demonstrations=len(old['labels']),samples=len(y),
    seed=100,collection=collection,epochs=4,lr=2e-4,combat_kl_coefficient=3,
    trainable_navigation_rows=list(nav_rows),unchanged_tensors=unchanged,history=history,
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    previous_demonstrations_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
    correction_dataset=str(root/'corrections_100.npz'),aggregate_dataset=str(root/'aggregate_100.npz'),
    collection_seconds=collection_seconds,training_seconds=time.perf_counter()-training_start,
    total_seconds=time.perf_counter()-started,quantum_parameter_l2_change=0.,selected_epoch=4,
    selection='prespecified final epoch; no gameplay model selection')
save_full(destination,model,meta); json_write(root/'training.json',meta)
print('SAVED',destination,'additional_decisions=4000',flush=True)
