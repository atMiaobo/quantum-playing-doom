"""Offline check that the saved policy depends on its trainable quantum branch."""
from pathlib import Path
import json
import numpy as np
import torch
from corridor_qnn.full_models import load_full

torch.set_num_threads(1)
root=Path('outputs/full_game')
model=load_full(root/'models/quantum.pt')
data=np.load(root/'demonstrations_42.npz')
idx=np.flatnonzero(data['episode_ids']==data['episode_ids'].max())
changed=0; logit_delta=0.
for chunk in np.array_split(idx,16):
    if not len(chunk): continue
    x=torch.from_numpy(data['images'][chunk]); f=torch.from_numpy(data['features'][chunk])
    with torch.no_grad():
        full,_=model(x,f)
        gain=model.quantum_gain.detach().clone()
        model.quantum_gain.zero_(); ablated,_=model(x,f); model.quantum_gain.copy_(gain)
        changed+=(full.argmax(-1)!=ablated.argmax(-1)).sum().item()
        logit_delta+=(full-ablated).abs().sum().item()
meta=torch.load(root/'models/quantum.pt',weights_only=True)['metadata']
result=dict(samples=len(idx),argmax_actions_changed=changed,action_change_fraction=changed/len(idx),
            mean_absolute_logit_change=logit_delta/(len(idx)*21),quantum_gain=float(model.quantum_gain.detach()),
            quantum_parameter_l2_change=meta['quantum_parameter_l2_change'],
            note='Offline branch ablation, not an independent gameplay benchmark or quantum advantage test.')
(root/'quantum_branch_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
