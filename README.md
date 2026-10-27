# Hybrid quantum-classical learning for ViZDoom

The project implements a hybrid quantum-classical policy for ViZDoom's stock
Deadly Corridor scenario. It combines a convolutional neural network (CNN),
structured environment features, a parameterized quantum circuit, and classical
action and value heads. Training supports scripted demonstrations, behavior
cloning, DAgger-inspired correction, and proximal policy optimization (PPO).

The quantum circuit is simulated classically with PyTorch complex autograd.
No quantum hardware or quantum SDK is required. Observations include
**privileged simulator-state features in addition to screenshots**; this is
not a pixels-only benchmark. The results do not establish quantum advantage.

## Full Doom / Freedoom levels

The separate full-game environment runs ViZDoom's bundled Freedoom Phase 2
IWAD, or an externally supplied Doom II IWAD using `--iwad PATH`. It supports
MAP01–MAP30, optional MAP31/32 (`--secret-maps`), difficulty 1–5, headless
execution, seeded episodes, screenshots, GIFs and native `.lmp` recordings.
Doom II assets are not distributed. The existing container/Nix launcher
requires no new dependencies or system configuration changes.

```bash
./run-nixos.sh full-doctor --map MAP01
./run-nixos.sh full-eval --policy random --map MAP01 --episodes 20
./run-nixos.sh full-eval --policy teacher --map MAP01 --episodes 20
mkdir -p outputs/full_game
./run-nixos.sh full-train --model quantum --map MAP01 --decisions 6000 \
  --epochs 12 --transfer-corridor pretrained/quantum_skill1_baseline.pt \
  > outputs/full_game/train_quantum.log 2>&1
./run-nixos.sh full-eval --policy quantum --map MAP01 --episodes 20 \
  --checkpoint outputs/full_game/models/quantum.pt \
  --record outputs/full_game/qnn_MAP01.gif --demo outputs/full_game/qnn_MAP01.lmp
./run-nixos.sh full-campaign --policy quantum --start-map MAP01 --end-map MAP30
```

A shared privileged IWAD geometry planner supplies waypoints to the scripted
teacher and learned policies. The hybrid uses the existing CNN and genuinely
trainable six-qubit/two-layer circuit, 54 navigation/combat/history features,
and 21 movement, shooting, USE and weapon actions. The network selects every
learned-policy action; the teacher does not override it. This is not
pixels-only learning or evidence of quantum advantage.

Full-game checkpoints have a separate schema. Optional partial transfer copies
compatible corridor modules; observation fusion and action heads start new.
The bundled corridor checkpoints are preserved. Training and benchmark artifacts
are generated under `outputs/full_game/` and are not bundled with the repository.

Completion requires the native map-exit event, independently of shaped reward.
The campaign runner progresses only after actual exits and initializes each map
with **fresh inventory**; continuous campaign state is not implemented. Keys,
lifts, teleports and complex map mechanics remain limitations.

See the [full-game guide](docs/FULL_GAME.md) for architecture, flags, metrics,
recording behavior and campaign semantics, and
[measured results](docs/FULL_GAME_RESULTS.md) for the bounded benchmark and exact
reproduction commands. Deadly Corridor commands below retain their original
behavior.

## Architecture

- RGB frames at 320 × 240 are converted to 64 × 96 grayscale and normalized
  inside a three-layer CNN, producing 48 visual features.
- A 20-element vector contains normalized health, ammunition, kills, relative
  position, yaw, goal and enemy geometry, visibility, damage, progress, time,
  and previous reward. Object positions can include enemies outside the view.
  When the goal object is unavailable, its position falls back to 1,500 world
  units along positive X from the starting position.
- Visual and structured features are fused into 48 values and projected to
  rotation angles. The default circuit has six qubits and two layers:
  RY input embedding, trainable RY/RZ rotations, a circular CNOT ladder in
  each layer, and one Pauli-Z expectation per qubit. It uses an ideal,
  noise-free state vector and supports two to eight qubits.
- Quantum measurements feed a ten-action policy head, added to a classical
  residual policy with a learned gain. A value head receives fused features
  and measurements. Actions combine movement, turning, strafing, and firing.
- The classical baseline retains the observations, CNN, fusion, residual,
  and value head, replacing the circuit with an MLP. It is not parameter matched.

## Requirements and installation

Use Linux with Python 3.11 or later; Python 3.11 is used by the container.
Runtime dependencies are PyTorch, NumPy, Pillow, and ViZDoom 1.3.x. Pytest is
needed for tests. Dependencies have bounded version ranges rather than an
exact Python environment lock. The Nix development shell has a retained lockfile.

On Debian/Ubuntu, install the native libraries used by the container:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libgl1 libglib2.0-0 libopenal1 \
  libsdl2-2.0-0 libx11-6 libxext6 libasound2 libsndfile1 libstdc++6 \
  libgcc-s1 libglu1-mesa ca-certificates
```

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
python -m corridor_qnn doctor
```

All usage examples below run from the repository root. Outputs are relative
to the current working directory. The installed `doom-corridor` command is
equivalent to `python -m corridor_qnn`.

### NixOS and Podman

Enable rootless Podman in the host configuration as described in
[docs/NIXOS.md](docs/NIXOS.md), then run:

```bash
nix develop
./run-nixos.sh build
./run-nixos.sh test
./run-nixos.sh doctor
```

For subsequent commands, replace `python -m corridor_qnn` with
`./run-nixos.sh`. The launcher mounts the repository at `/app`, preserving
host outputs and ownership. The Dockerfile uses the fully qualified image
`docker.io/library/python:3.11-slim-bookworm`. Docker is also supported through
`DOOM_CORRIDOR_ENGINE=docker`. Gameplay runs headlessly and does not need a display.

## Pretrained evaluation and recording

The primary example is `pretrained/quantum_skill3_candidate.pt`, the six-qubit,
two-layer model identified as difficulty-3 imitation fine-tuning. It can run
without training or copying checkpoints:

```bash
python -m corridor_qnn eval --policy quantum \
  --checkpoint pretrained/quantum_skill3_candidate.pt \
  --skill 3 --episodes 5 --seed 800 --record outputs/quantum_skill3.gif
```

To use legacy commands that expect `outputs/models/quantum.pt`, initialize it once:

```bash
python scripts/init-checkpoints.py
python -m corridor_qnn eval --policy quantum --skill 3 --episodes 5
```

The initializer copies the difficulty-3 candidate and preserves an existing
checkpoint. Native training continues to save to `outputs/models/<model>.pt`
unless `--output-checkpoint` is supplied. Model dimensions are read from the
checkpoint when loading. [pretrained/README.md](pretrained/README.md) records
checkpoint metadata and checksums. No classical pretrained model is included.

Evaluate the scripted teacher or random policy without a checkpoint:

```bash
python -m corridor_qnn eval --policy teacher --skill 1 --episodes 1 \
  --record outputs/teacher.gif
python -m corridor_qnn eval --policy random --skill 1 --episodes 5
```

`eval` uses argmax actions by default; `--stochastic` samples the network.
It prints per-episode statistics and a JSON `SUMMARY` containing win rate,
mean kills, raw reward, and steps. Recordings contain the first episode only,
sampled every second action, capped at 300 captured frames, and saved as GIFs.
Defaults are frame skip 4 and at most 450 macro-actions per episode. Success
uses the original implementation's raw action reward threshold of at least 500.

## Training

The geometric scripted teacher aims and fires at nearby visible enemies,
strafe-dodges at close range, and advances toward the goal. It uses the same
structured features as the learned policy and is not a guaranteed expert.

Start behavior cloning from random initialization:

```bash
python -m corridor_qnn imitate --model quantum --qubits 6 --layers 2 \
  --skill 1 --episodes 35 --epochs 8
```

Collection uses reservoir sampling with a default cap of 12,000 observations.
Cross-entropy training selects the checkpoint by held-out action accuracy
(90/10 frame split, not an episode split). Action accuracy is not win rate.
Fresh imitation overwrites the default training checkpoint; bundled models
remain available in `pretrained/`.

Continue from a trained student with mixed teacher/student rollouts:

```bash
python -m corridor_qnn imitate --model quantum --resume --teacher-prob 0.5 \
  --skill 1 --episodes 30 --epochs 8
```

Every visited state receives a teacher label, including student-controlled
states. This is DAgger-inspired correction: each invocation collects a new
dataset and does not persist or aggregate datasets across invocations.

Fine-tune imitation at difficulty 3 with explicit input and output paths:

```bash
python -m corridor_qnn imitate --model quantum \
  --checkpoint pretrained/quantum_skill1_baseline.pt \
  --output-checkpoint outputs/models/quantum_skill3_run.pt \
  --teacher-prob 0.5 --skill 3 --episodes 30 --epochs 8
```

This is an example training command, not a complete reconstruction of the
historical training schedule, which was not recorded in checkpoint metadata.
`--checkpoint` loads input weights even without `--resume`; `--resume` alone
loads the legacy default and fails if it is missing.

PPO fine-tuning uses generalized advantage estimation, a clipped policy loss,
a value loss, entropy regularization, and gradient clipping:

```bash
python -m corridor_qnn ppo --model quantum \
  --checkpoint pretrained/quantum_skill3_candidate.pt \
  --output-checkpoint outputs/models/quantum_ppo.pt --skill 3 --steps 25000
```

Without an explicit input, PPO loads `outputs/models/<model>.pt` if present,
otherwise starts randomly. Optimizer state is not saved; subsequent runs
restart the optimizer. PPO saves after each rollout and appends metrics to
`outputs/ppo_history.csv`. Its training reward is clipped native reward / 50,
plus kill and hit bonuses, minus damage and time penalties. Evaluation reports
unmodified native rewards.

Train and evaluate the classical baseline using the same CLI:

```bash
python -m corridor_qnn imitate --model classical --skill 1 --episodes 35 --epochs 8
python -m corridor_qnn ppo --model classical --skill 1 --steps 25000
python -m corridor_qnn compare --skill 1 --episodes 10 \
  --quantum-checkpoint pretrained/quantum_skill3_candidate.pt
```

`compare` evaluates the teacher and available models with the same environment
seed and settings, saving `outputs/comparison.json`; absent default models are
skipped. `--classical-checkpoint` selects a separate classical model. These
commands do not constitute a controlled benchmark by themselves.

Use `python -m corridor_qnn --help` or `<command> --help` for all options.

## Preliminary results

The following measured results were supplied with the project for the six-qubit
hybrid model after difficulty-3 imitation fine-tuning: 30 episodes per difficulty,
environment seed 800. Training primarily used scripted demonstrations followed
by imitation fine-tuning. PPO is supported separately; these results are not
attributed to PPO.

| Difficulty | Win rate | Average kills |
|---|---:|---:|
| 1 | 93.3% | 5.87 |
| 3 | 76.7% | 5.43 |
| 5 | 23.3% | 3.77 |

These are preliminary single-seed evaluation results, not evidence of quantum
advantage or superiority over classical architectures. Per-episode logs and a
complete historical training schedule were not available in the source project.
The checkpoint records behavior-cloning sample count and validation accuracy,
but does not record training difficulty. The difficulty-3 attribution accompanies
the supplied checkpoint and results.

The evaluation commands for the reported settings are:

```bash
for skill in 1 3 5; do
  python -m corridor_qnn eval --policy quantum \
    --checkpoint pretrained/quantum_skill3_candidate.pt \
    --skill "$skill" --episodes 30 --seed 800
done
```

## Project structure

```text
corridor_qnn/          Python package and module CLI
  actions.py          Ten combined movement/combat actions
  environment.py      Stock ViZDoom scenario and reward adapter
  features.py         Image preprocessing and privileged features
  teacher.py          Scripted geometric demonstration policy
  quantum.py          Differentiable state-vector circuit
  models.py           Hybrid and classical actor-critic policies
  training.py         Imitation collection/training, PPO, checkpoints
pretrained/           Two compact quantum checkpoints and provenance
scripts/              Non-destructive default-checkpoint initializer
tests/                Model, feature, training, and packaging tests
docs/                 NixOS setup and validation details
nixos/                Optional host Podman module
outputs/              Generated checkpoints, metrics, and recordings (ignored)
Dockerfile            Linux runtime image
run-nixos.sh          Podman/Docker launcher
flake.nix, flake.lock  Nix development tooling
pyproject.toml        Package metadata, dependencies, and console command
requirements.txt      Container dependency list (PyTorch installed separately)
```

See [docs/VALIDATION.md](docs/VALIDATION.md) for checks performed on this package
and their execution limitations. Pretrained assets and repository utilities
are distributed with the checkout; a Python wheel alone does not include them.
