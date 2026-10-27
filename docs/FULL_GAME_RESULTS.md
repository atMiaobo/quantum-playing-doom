# Full-game benchmark, 2026-10-28

The trained hybrid policy plays the actual bundled Freedoom Phase 2 MAP01 and
outperforms random actions on kills, trajectory coverage, collected items and
maximum displacement. It reached the genuine level exit in **1/20 episodes**.
This is a bounded privileged-state proof of concept, not a reliable map solver
or evidence of quantum advantage. The classical ablation completed 2/20.

## Protocol and budget

- ViZDoom 1.3.2, PyTorch 2.14.1+cpu, Python 3.11.17, x86-64, rootless Podman.
- Bundled `freedoom2.wad` SHA-256:
  `a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b`.
- MAP01, skill 1, 4 engine tics/action, 1,000 decisions maximum per episode.
- Final comparison: **20 episodes per policy**, identical seeds **800–819**.
  Learned policies use argmax actions. Random samples uniformly from all 21
  macro actions. Every policy receives the same geometry/observation pipeline.
- Exactly **6,000 collected training decisions**, teacher seeds 42–47.
  The classical model reuses those demonstrations and collects zero new ones.
  One 12-epoch imitation stage per architecture; no QNN improvement iterations,
  PPO, larger quantum circuits or further training were used.
- Training defaults: Adam, learning rate 0.001, batch size 64, square-root inverse
  action-frequency loss weighting; compatible partial corridor transfer for
  both architectures. The quantum circuit additionally transfers its weights.
- The final demonstration episode is held out for action accuracy (5,000 gradient-training samples and 1,000 validation samples). Gameplay
  validation uses seeds 500–502 at epochs 6 and 12: six validation episodes per
  model, separate from collection and final testing. Selection prioritizes
  completion, then visited cells + 2×kills; both selected epoch 12.
- QNN training, including collection and validation: 126.88 seconds. Classical
  training with dataset reuse and validation: 59.40 seconds. Timings are local
  shared-host measurements, not isolated hardware benchmarks.

## Measured results

Values are episode means, except completion counts. Coverage is the number of
unique visited 64×64 world-unit XY cells, not a fraction of the whole map.
Items use the engine's `ITEMCOUNT`, which excludes many ammo/weapon pickups.

| Policy | Completed | Kills | Coverage cells | Items | Health | Survival seconds | Shaped reward | Wall seconds/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0/20 | 0.90 | 44.50 | 1.60 | 29.90 | 100.64 | 2.01 | 4.81 |
| Teacher | 1/20 | 7.60 | 113.45 | 5.30 | 82.00 | 110.50 | 40.01 | 5.60 |
| Hybrid QNN | 1/20 | 6.40 | 98.90 | 5.70 | 48.60 | 111.10 | 32.77 | 10.06 |
| Classical neural | 2/20 | 6.05 | 96.25 | 5.60 | 52.60 | 109.63 | 34.08 | 7.60 |

Mean native evaluation reward, which is exclusively the exit event, was 0.00,
0.05, 0.05 and 0.10 respectively. Death counts were 9, 1, 4 and 3. Mean maximum
displacement was 1,085.00, 2,170.57, 2,094.01 and 2,091.93 world units. Mean pickup
increase events were 2.10, 8.70, 8.75 and 8.25; these are observed increases in
ammo/health/armor/weapon state, not counts of every inventory object.

Paired QNN-minus-random mean differences were **+5.50 kills** and **+54.40 visited
cells**. A 10,000-resample paired episode bootstrap gives intervals [4.70, 6.20]
and [40.00, 67.35], respectively. These intervals describe variation over the
specified seeds on one map, not generalization to new maps. One exit versus
zero in 20 episodes does not establish a reliable completion advantage.

Measured action-selection cost was approximately 0.02 ms/action for random,
0.04 ms for the teacher, 4.17 ms for the QNN and 1.73 ms for the classical
network. Environment/navigation costs are additional. Some jobs ran
concurrently, so runtime comparisons are approximate.

The quantum parameter L2 change from transferred initialization was 0.46583;
its learned residual gain was 1.64930. Removing the quantum contribution changed
142/1,000 held-out demonstration argmax actions (14.2%). This offline check and
gradient tests establish an active trained circuit, not a beneficial quantum
ablation result. No quantum-versus-classical advantage is claimed.

## Verified behavior and artifacts

- Real launches and movement on MAP01, MAP02, MAP30 and MAP31. Only MAP01 has
  policy benchmarks; Doom II and MAP32 were not executed.
- All 21 macro actions ran in the MAP01 doctor smoke check.
- Real engine timeout and death checks passed. A separate diagnostic teleports
  near the exit linedef read from MAP01 geometry and activates its real exit;
  this is solely a terminal-detection test, excluded from all benchmark results.
- The QNN's unassisted completion was **seed 807**, 657 decisions, 8 kills,
  102 visited cells, 6 counted items, 61 health and native exit reward 1.
  The selected successful recording is an illustrative episode, not the
  basis for the mean results. The first benchmark episode is also recorded. The successful native demo was
  replayed independently and again ended with native exit reward 1 after 657
  playback decisions.
- **48 tests passed**. Coverage includes actions, maps, rewards, terminal states,
  waypoint geometry, model dimensions, actual quantum gradients, checkpoint
  schema and partial transfer, seeded evaluation/recording, campaign progression,
  external IWAD mounts, and all original corridor tests.
- Updated Dockerfile built successfully. Original corridor doctor passed;
  `quantum_skill1_baseline.pt` completed a real corridor episode with 6 kills
  in 75 decisions. Both bundled checkpoint hashes remain unchanged.

Local generated paths (ignored by Git):

| Artifact | Path |
|---|---|
| Trained hybrid | `outputs/full_game/models/quantum.pt` |
| Classical ablation | `outputs/full_game/models/classical.pt` |
| Demonstrations | `outputs/full_game/demonstrations_42.npz` |
| First evaluated QNN episode | `outputs/full_game/qnn_MAP01.gif` and `.lmp` |
| Successful seed 807 recording | `outputs/full_game/qnn_MAP01_exit.gif` and `.lmp` |
| Per-policy final episodes | `outputs/full_game/MAP01_{random,teacher,quantum,classical}.json` |
| Consolidated benchmark / intervals | `outputs/full_game/benchmark.json` |
| Training logs / metadata | `outputs/full_game/train_{quantum,classical}.log`, `training_{quantum,classical}_42.json` |
| Tests and engine checks | `outputs/full_game/tests.log`, `integration.json`, `integration.log` |
| Quantum branch ablation | `outputs/full_game/quantum_branch_check.json` |

The QNN checkpoint SHA-256 is
`461d98644bbfd378588212cf4bcfb4fa724946853ea9037d1e342e01eb4a5f0a`.
Generated models are local artifacts, not bundled pretrained releases.

## Reproduction

From the repository root, after `./run-nixos.sh build`:

```bash
mkdir -p outputs/full_game
./run-nixos.sh test
./run-nixos.sh full-doctor --map MAP01
./run-nixos.sh full-train --model quantum --map MAP01 --skill 1 \
  --decisions 6000 --epochs 12 --seed 42 --max-steps 1000 \
  --transfer-corridor pretrained/quantum_skill1_baseline.pt \
  > outputs/full_game/train_quantum.log 2>&1
./run-nixos.sh full-train --model classical --map MAP01 --skill 1 \
  --dataset outputs/full_game/demonstrations_42.npz --epochs 12 --seed 42 \
  --max-steps 1000 --transfer-corridor pretrained/quantum_skill1_baseline.pt \
  > outputs/full_game/train_classical.log 2>&1
for policy in random teacher quantum classical; do
  ./run-nixos.sh full-eval --policy "$policy" --map MAP01 --skill 1 \
    --episodes 20 --seed 800 --max-steps 1000 \
    > "outputs/full_game/${policy}_eval.log" 2>&1
done
./run-nixos.sh full-eval --policy quantum --map MAP01 --skill 1 \
  --episodes 1 --seed 807 --max-steps 1000 \
  --checkpoint outputs/full_game/models/quantum.pt \
  --output-dir outputs/full_game/recording_exit \
  --record outputs/full_game/qnn_MAP01_exit.gif \
  --demo outputs/full_game/qnn_MAP01_exit.lmp
```

For the reporting/diagnostic scripts, use the installed Python environment or
run them inside the same container:

```bash
podman run --rm --userns=keep-id -v "$PWD:/app" -w /app -e PYTHONPATH=/app \
  --entrypoint python localhost/doom-corridor-qnn:latest scripts/full-report.py
podman run --rm --userns=keep-id -v "$PWD:/app" -w /app -e PYTHONPATH=/app \
  --entrypoint python localhost/doom-corridor-qnn:latest scripts/full-integration.py
podman run --rm --userns=keep-id -v "$PWD:/app" -w /app -e PYTHONPATH=/app \
  --entrypoint python localhost/doom-corridor-qnn:latest scripts/full-quantum-check.py
```

Dependency ranges are bounded but not fully locked; exact results can vary
across library/platform versions. Runtime versions and IWAD identity are stored
with the benchmark. Training can be skipped when using the generated checkpoint.

## Corrections, limitations and next experiment

Two benchmark corrections were necessary: weapon switches initially inflated
pickup counters, and resetting after a native recording froze subsequent game
time in ViZDoom 1.3.2. Those results were discarded/replaced; no policy weights
were tuned on final seeds. The invalid recording-reset output is explicitly
named `INVALID_recording_reset_quantum.json`. The adapter now isolates native
recording sessions and rejects nonadvancing steps. Native exits reset engine
counters/time, so evaluation retains the last valid pre-exit values, potentially
omitting at most four tics of terminal events.

The planner uses full static IWAD geometry and visible monster labels. It can
misjudge doors, narrow passages, heights and dynamic obstacles. It does not
track keys or plan switch dependencies, lifts or teleports. Behavior cloning
still accumulates steering errors and loops; completion remains rare. The
campaign runner is a numeric sequence of independently initialized maps with
fresh inventory, not continuous campaign play. No campaign completion is claimed.

The smallest useful next experiment is one capped DAgger correction collection
on student-induced stuck/door states, retaining this checkpoint and using new
training seeds and unchanged validation seeds. Improve observable door/key
state before attempting longer maps or PPO. This experiment was not run.
