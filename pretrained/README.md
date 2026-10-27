# Pretrained quantum policies

Both files are unmodified PyTorch checkpoints for `ActorCritic(kind="quantum",
qubits=6, layers=2)`, with 67,568 trainable parameters. Each is 278,235 bytes
(about 272 KiB); Git LFS is unnecessary at this size. The classical baseline
implementation is included, but no classical trained checkpoint was present.

| File | Role | Demonstration samples | Held-out action accuracy |
|---|---|---:|---:|
| `quantum_skill3_candidate.pt` | Primary example, identified as difficulty-3 imitation fine-tuning | 5,220 | 94.64% |
| `quantum_skill1_baseline.pt` | Earlier skill-1 imitation baseline | 2,858 | 91.96% |

Sample counts and action accuracies come directly from checkpoint metadata;
these are not gameplay win rates. Both store `training: behavior_cloning` and
note that demonstrations came from a heuristic rather than expert wins.
Metadata does not store difficulty, training seed, frame skip, total curriculum,
or optimizer state. Difficulty labels reflect the accompanying project
provenance, not fields recovered from the checkpoint.

The source `quantum.pt` was identical to the difficulty-3 candidate, and
`quantum_skill1_best.pt` was identical to the skill-1 baseline. Those duplicates
and the older PPO checkpoint are excluded. Full SHA-256 hashes, sizes,
architecture settings, and original metadata are in [manifest.json](manifest.json).

From the repository root:

```bash
python -m corridor_qnn eval --policy quantum \
  --checkpoint pretrained/quantum_skill3_candidate.pt \
  --skill 3 --episodes 5 --seed 800 --record outputs/quantum_skill3.gif
```

For legacy default-checkpoint commands, run `python scripts/init-checkpoints.py`.
This copies the primary model to `outputs/models/quantum.pt` only if that file
is absent. Training should write into `outputs/models/`; an explicit
`--output-checkpoint` keeps separate runs separate. The original checkpoint
format remains unchanged: `state_dict`, `kind`, `qubits`, `layers`, and `metadata`.
Loading uses CPU tensors and reconstructs the dimensions from the file.
