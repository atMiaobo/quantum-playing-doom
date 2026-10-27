# Full Doom / Freedoom levels

A separate `FullDoom` environment runs the actual maps in ViZDoom's bundled
Freedoom Phase 2 IWAD. An externally supplied, legally obtained Doom II IWAD
is supported through `--iwad`. No Doom II assets are distributed. The adapter
starts from the official `freedoom2.cfg`, enables labels, depth and automap,
and supports MAP01–MAP30, difficulty 1–5, headless execution, reproducible
per-episode seeds, and configurable decision limits. MAP31/32 require
`--secret-maps`. IWADs and generated datasets are ignored by Git.

```bash
./run-nixos.sh build
./run-nixos.sh full-doctor --map MAP01
./run-nixos.sh full-eval --policy random --map MAP01 --episodes 20
./run-nixos.sh full-eval --policy teacher --map MAP01 --episodes 20
./run-nixos.sh full-eval --policy random --map MAP02 --max-steps 1000
./run-nixos.sh full-doctor --iwad /absolute/path/to/doom2.wad --map MAP01
```

The launcher mounts an absolute `--iwad PATH` read-only into the container.
A relative IWAD path must be inside the checkout, for example
`--iwad assets/doom2.wad`. Native Python uses the same flags. The existing
Nix development shell and native libraries suffice; no NixOS configuration
changes or new Python dependencies are required.

### Controller and model

Full-game actions have their own explicit 21-entry mapping in
`full_actions.py`: movement in four directions, turning, shooting, movement
and turning with fire, USE, forward+USE, forward+turn, weapon slots 2–4,
and next weapon. Buttons are matched by name rather than configuration order.

A shared **privileged classical navigator** parses Doom-format IWAD vertices,
linedefs, sidedefs and sector heights. The default uses a 16-unit player-clearance
grid, front-side exit approaches and bounded local collision recovery; the
original 32-unit grid is available with `--navigation legacy`. It finds routes
to exit linedefs when reachable, and falls back
to less-visited reachable cells. Exit positions come from map geometry; there
are no map-specific coordinates or recorded action sequences. Closed doors
are treated as potentially traversable. This approximation does not solve
keys, lifts, teleports, dynamic geometry or arbitrary scripted specials.
Hexen/UDMF map formats are rejected explicitly.

See the [MAP01 navigation experiment](MAP01_NAVIGATION.md) for the measured
follow-up and conservative DAgger update. Historical results in
`FULL_GAME_RESULTS.md` use `--navigation legacy --teacher-version legacy`.
Evaluation and training refuse existing output result/checkpoint files; choose
new destinations to preserve earlier runs.

The scripted teacher steers toward these waypoints, aims at visible engine
monster labels, shoots, chooses a shotgun when acquired, and attempts USE,
backward movement and strafing when stuck. It is a heuristic demonstrator,
not an expert or a trained network.

The learned hybrid retains the original 64×96 grayscale CNN, 48-value fusion,
six-qubit/two-layer differentiable RY/RZ/CNOT state-vector circuit and residual
classical action head. Its 54 structured features include waypoint bearing,
visible enemy bearing, depth samples, health, armor, selected weapon/ammo,
shotgun/chaingun ownership, motion, damage, visited-cell count, stuck status,
previous bearings/visibility and a previous-action one-hot vector. Temporal
information is explicit recent state rather than a recurrent network. The
critic is retained but is not fitted during behavior cloning. The quantum
circuit is trained by autograd and evaluated for every learned-policy action;
there is no teacher action override during learned evaluation. Navigation is
provided by the shared planner, so this is not autonomous pixels-only learning.

Full-game checkpoints use a versioned schema with exact feature/action names.
Old corridor checkpoints cannot be loaded directly. `--transfer-corridor`
explicitly copies compatible CNN, angle projection, circuit, second fusion
layer and critic weights, leaving the input fusion and both action heads new.
A classical ablation can transfer the same compatible weights except the
quantum circuit; its core is an MLP. This comparison is not parameter matched
and does not establish quantum advantage. Bundled corridor checkpoints remain
unchanged.

### Bounded training and recording

```bash
mkdir -p outputs/full_game
./run-nixos.sh full-train --model quantum --map MAP01 --skill 1 \
  --decisions 6000 --epochs 12 --seed 42 --max-steps 1000 \
  --transfer-corridor pretrained/quantum_skill1_baseline.pt \
  > outputs/full_game/train_quantum.log 2>&1
./run-nixos.sh full-eval --policy quantum --map MAP01 --skill 1 \
  --episodes 20 --seed 800 --max-steps 1000 \
  --checkpoint outputs/full_game/models/quantum.pt \
  --record outputs/full_game/qnn_MAP01.gif \
  --demo outputs/full_game/qnn_MAP01.lmp
```

The generated checkpoint above is local under `outputs/`, not bundled in
`pretrained/`. Screenshots and automap images come from `full-doctor`.
`--record` supports GIF and captures the first episode across its duration
(up to approximately 400 frames); MP4 export is not implemented. `--demo`
saves a native ViZDoom replay of the same first episode. ViZDoom 1.3.2 can
freeze episodes following an interrupted native recording: the adapter
recreates the engine after a recording session and rejects steps if game
time does not advance.

Training saves demonstrations, epoch history, collection statistics and the
selected checkpoint under `outputs/full_game/`. `--output-dir`,
`--output-checkpoint`, `--decisions`, `--epochs`, `--batch-size`, and `--lr`
control paths and limits. Collection stops at the decision budget, including
partial final episodes. `--checkpoint` loads a full-game model and requires a
different output path to preserve the source. Optional teacher/student mixed
collection uses `--teacher-prob`; no PPO command is added for full levels.

The bounded run used 6,000 collected decisions from seeds 42–47, one imitation
stage per architecture, 12 epochs, and no improvement or RL iterations. A
classical baseline reused the same dataset with **zero additional collection**:

```bash
./run-nixos.sh full-train --model classical --map MAP01 --skill 1 \
  --dataset outputs/full_game/demonstrations_42.npz --epochs 12 --seed 42 \
  --max-steps 1000 --transfer-corridor pretrained/quantum_skill1_baseline.pt \
  > outputs/full_game/train_classical.log 2>&1
./run-nixos.sh full-eval --policy classical --map MAP01 --episodes 20 \
  --seed 800 --max-steps 1000
```

Validation holds out an entire demonstration episode and evaluates gameplay
at epochs 6 and 12 on seeds 500–502 (six episodes per model). Checkpoint
selection first uses actual completion rate, then visited cells + 2×kills.
The final evaluation seeds 800–819 are excluded from training and selection.
The 10,000-decision per-invocation collection guard is not a cumulative budget
manager; account for collection across separate invocations.

### Evaluation and campaign semantics

`full-eval` writes individual episodes and a summary to
`outputs/full_game/MAP01_<policy>.json`. All policies use the same IWAD,
difficulty, seed list, action space, frame skip (4) and 1,000-decision cap.
Metrics include genuine completion/death/timeout, kills, survival seconds,
unique visited 64×64 world-unit XY cells, counted items (`ITEMCOUNT`), pickup
increase events, remaining health, maximum displacement, native exit reward,
shaped reward and runtime. Cells measure trajectory coverage, not a percentage
of accessible floor area; vertical floors share XY cells. Pickup events may
combine simultaneous pickups and do not count individual inventory objects.

Only the dedicated native `map_exit_reward=1` identifies completion. All other
native reward terms are zero; a generic terminal state alone is not a win.
Engine timeouts, the decision cap, death and unrecognized terminal states are
separate outcomes. Native exit reward and shaped training reward are reported
separately. Shaping adds exploration, kills, hits, items, health/armor gains,
and exit bonuses, with damage, death and per-decision penalties. USE attempts
receive no bonus without an observed effect. ViZDoom resets some counters and
time at exits, so completed episodes retain their last valid pre-exit counters
and time, with at most one action (four tics) of omitted terminal events.

```bash
./run-nixos.sh full-campaign --policy quantum --start-map MAP01 --end-map MAP30
./run-nixos.sh full-campaign --policy teacher --start-map MAP31 --end-map MAP32 \
  --secret-maps
```

This runner evaluates **independently initialized maps with fresh inventory**,
advancing in numeric order only after every requested episode completes.
It stops on failure and does not follow secret-exit branches. ViZDoom's
single-player episode ends at a map exit; this implementation uses
`new_episode`/new map initialization, which does not carry health, weapons,
ammo or keys across maps. Continuous inventory-preserving campaign play is
not implemented or claimed. MAP30 boss/engine-specific completion mechanics
and Doom II gameplay have not been benchmarked.

See [FULL_GAME_RESULTS.md](FULL_GAME_RESULTS.md) for measured
benchmarks, reproduction commands, tested maps and limitations. The official
[full-level configuration](https://vizdoom.farama.org/environments/original_doom_levels/)
and [terminal-state API](https://vizdoom.farama.org/api/python/doom_game/)
describe the ViZDoom behavior used here.

