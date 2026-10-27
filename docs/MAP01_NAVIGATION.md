# MAP01 navigation experiment, 2026-10-28

This bounded follow-up keeps the original models and historical benchmark intact.
It changes the shared privileged geometry planner and teacher, then runs one
conservative DAgger update. No coordinates or action sequences specific to MAP01
are embedded in either controller. The six-qubit, two-layer architecture and
54-feature/21-action checkpoint schema are unchanged.

## Diagnosis from actual gameplay

Diagnostic episodes use seeds 600–602, separate from collection and final tests.
Per-decision positions, yaw, waypoints, actions, health, kills and features, plus
sampled gameplay contact sheets, are under
`outputs/map01_navigation/diagnostic_legacy_{teacher,quantum}/traces/`.

* The original QNN's seed 600 spent 588 decisions marked stuck, with 591 samples
  in the same approximately 32-unit location. It repeatedly tried to cross a
  ledge near (1200, -40), while its waypoint remained across the height barrier.
  The contact sheet shows the same wall for most of the episode.
* The old planner targets (232, -744), over 110 units from the actual exit
  linedef midpoint (216, -632). The exit region is disconnected in its static
  collision grid. The teacher nevertheless sometimes drifts into that region,
  points at the wrong side of the switch, then abandons the goal when its visit
  counter exceeds eight. Seed 600 leaves the exit region and returns to distant
  corridors by the decision cap.
* WAD and live-engine inspection identify a thin depression between two banks
  of equal floor height. The engine's player footprint spans the gap, while
  rasterizing both banks as independent impassable steps invents a barrier.
  Live sector heights remained at their WAD values during the diagnostic;
  this was not an observed moving-floor dependency.
* Death was secondary in these diagnostic samples: two of the three original
  teacher episodes timed out; one died after exploration with poor positioning.
  Closed-door interactions and repeated visits were directly visible in traces.

The first targeted iteration added clearance, route recovery and front-side
exit goals but still treated the narrow depression as disconnected: teacher
0/3 exits, 86.67 cells, 4.67 kills on diagnostic seeds. The second and final
iteration adds footprint spanning for narrow depressions with compatible bank
heights: teacher 3/3 exits, 94.67 cells, 8.00 kills. No further navigation
implementation iteration was performed after seeing the final benchmark.

## Implemented behavior

`LegacyNavigator` and `legacy_teacher_action` retain the original behavior.
`--navigation legacy --teacher-version legacy` reproduces the original setup.
The new default planner uses a 16-unit grid and 18-unit clearance from solid
linedefs, with sampled line-of-sight lookahead rather than skipping arbitrary
nearby grid corners. Narrow depressions below the player's 32-unit diameter
are spanned only when adjacent bank floors agree within step height.

Use-exit approach points come from linedef orientation: Doom switches activate
from their front/right side. The planner retains a reachable exit goal instead
of abandoning it after eight visits. Unreachable exits fall back to reachable,
less-visited areas with travel-distance pressure. Repeated failed forward
movement temporarily rejects a local cell and replans; stationary turning and
combat are excluded from this collision detector.

Interactions are limited to recognized use specials ahead of the current route,
with distance measured to their segment. Scrolling textures and walk triggers
are excluded. The teacher aligns with the route and releases USE between pulses.
Its original visible-enemy aiming, firing and weapon-selection rules are retained.

Evaluation supports JSON traces and contact sheets (`--trace-dir`) and a
`--quantum-disabled` gameplay ablation. Existing result files and output
checkpoints are refused instead of overwritten. The launcher can run project
Python scripts through the existing image and explicitly sets `PYTHONPATH=/app`;
otherwise script execution can silently import the image's old installed package.
No new container build, WAD download or NixOS change was required.

## Budget and training

The improved teacher's 20-episode benchmark was completed before starting
fine-tuning. DAgger collected exactly **4,000 additional training decisions**,
executing only the original QNN's deterministic actions with improved navigation
and querying the improved teacher at each visited state. Collection seeds are
100–105; the last episode is partial. These were aggregated with the existing
6,000 demonstrations, for 10,000 stored samples. The last correction episode
is held out from gradient fitting.

There is one four-epoch fine-tuning stage: Adam, learning rate 0.0002, batch 64,
clip norm 0.5. Only navigation rows 0–5 and 12–15 of the two action heads can
change. The CNN, fusion, angle projection, quantum circuit, residual gain,
critic and combat action-head rows remain byte-for-byte equal to the baseline;
the training script asserts this before saving. A KL penalty of 3 against the
original policy on visible-enemy replay states also protects the competition
between navigation and combat actions. Frozen combat rows alone do not guarantee
identical action choice because navigation logits still compete with them.

The final epoch is prespecified. Final seeds 800–819 are never used for gradients,
collection or epoch selection. Diagnostics and evaluation trajectories are not
added to training. There was no second training stage, PPO, circuit expansion,
or additional collection after inspecting the candidate benchmark.

## Measured comparison

All rows use the same bundled IWAD, MAP01, skill 1, seeds **800–819**, four tics
per action and a **1,000-decision cap**, with deterministic learned policies.
Completion is exclusively the engine's native map-exit reward, not displacement,
shaped reward or proximity. No benchmark episode uses teleportation or a scripted
solution. Original-policy results were rerun and checked against the saved
historical per-episode gameplay metrics.

| Policy | Exits | Kills | Cells | Items | Pickup events | Health | Deaths | Wall s (20 eps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random, original planner | 0/20 | 0.90 | 44.50 | 1.60 | 2.10 | 29.90 | 9 | 95.19 |
| Original teacher | 1/20 | 7.60 | 113.45 | 5.30 | 8.70 | 82.00 | 1 | 115.31 |
| Improved teacher | 19/20 | 7.45 | 92.05 | 5.10 | 6.95 | 85.35 | 0 | 125.19 |
| Original QNN, original planner | 1/20 | 6.40 | 98.90 | 5.70 | 8.75 | 48.60 | 4 | 229.31 |
| Original QNN, improved planner | 19/20 | 7.05 | 94.85 | 5.60 | 8.25 | 58.80 | 0 | 213.52 |
| DAgger QNN, improved planner | 17/20 | 7.00 | 91.50 | 4.90 | 7.05 | 57.10 | 0 | 215.23 |
| DAgger QNN, quantum disabled | 13/20 | 5.95 | 109.35 | 5.90 | 9.35 | 30.75 | 4 | 232.19 |
| Classical, original planner | 2/20 | 6.05 | 96.25 | 5.60 | 8.25 | 52.60 | 3 | 171.81 |
| Classical, improved planner | 20/20 | 6.80 | 98.00 | 5.25 | 8.20 | 65.45 | 0 | 157.82 |

Cells are unique visited 64-unit XY cells, so a shorter successful route can have
lower coverage than a long looping timeout. Items are `ITEMCOUNT`; pickup events
also include observed ammo, health, armor and weapon increases. Final health is
retained from the last valid pre-exit sample; at most four terminal tics can be
omitted. Wall time is measured locally under concurrent bounded jobs and should
not be interpreted as an isolated performance comparison.

**Recommended QNN configuration: preserve the original checkpoint and use the
improved planner.** It reaches 19/20 exits versus 1/20 originally, raises kills
from 6.40 to 7.05 and removes the four deaths observed in the original benchmark.
The separately retained DAgger candidate reaches 17/20 exits and 7.00 kills,
with zero deaths. Thus navigation substantially improves completion, but this
fine-tuning stage does not improve on the planner-only change and is not promoted.
The classical policy improves from 2/20 to 20/20 with the same planner, further
supporting navigation as the main bottleneck.

Disabling the candidate's quantum residual changes completion from 17/20 to
13/20, average kills from 7.00 to 5.95 and deaths from zero to four. The branch
therefore changes gameplay in this matched sample; the classical result and
small single-map sample preclude a quantum-advantage claim.

All four original policies reproduce their historical per-episode completion,
kills, cells, items, pickups, health and decision counts exactly. **53 automated
tests pass**. Live checks on MAP01, MAP02 and MAP30 verify finite observations,
advancing game time and timeouts; the recording/reset isolation check also passes.
The candidate's seed 800 recording exits after 696 decisions with 8 kills and
36 health. The recommended original-weights/new-planner seed 800 recording exits
after 823 decisions with 8 kills and 42 health. Independent native replay checks
are stored in `recording_replay.json`.

Quantum disabling sets `quantum_gain=0`, removing the quantum residual from action
logits on the same candidate checkpoint; it still computes the circuit. This
measures gameplay dependence on that branch, not execution speed. It does not
establish quantum advantage, and this small single-map experiment does not
establish generalization.

## Artifacts and reproduction

The original checkpoint is still `outputs/full_game/models/quantum.pt`, SHA-256
`461d98644bbfd378588212cf4bcfb4fa724946853ea9037d1e342e01eb4a5f0a`.
The candidate is `outputs/full_game/models/quantum_map01_dagger.pt`.
Predictable experiment paths under `outputs/map01_navigation/` include:

| Artifact | Relative path |
|---|---|
| Reproduced original benchmarks | `baseline/{random,teacher,quantum,classical}/MAP01_*.json` |
| Final planner with original weights / improved teacher | `improved_navigation/{teacher,quantum,classical}/MAP01_*.json` |
| Fine-tuned QNN and branch ablation | `candidate/MAP01_quantum.json`, `quantum_disabled/MAP01_quantum.json` |
| Corrective data and aggregate | `dagger/corrections_100.npz`, `dagger/aggregate_100.npz` |
| Training metadata and collection outcomes | `dagger/training.json` |
| Failure screenshots and per-decision traces | `diagnostic_legacy_*/traces/`, `final_failure_teacher/traces/` |
| Candidate regression trace | `final_failure_candidate/traces/quantum_809.{json,png}` |
| Successful QNN recording | `recording/qnn_exit_*.gif`, `recording/qnn_exit_*.lmp` |
| Recommended original weights with new planner | `recommended_recording/qnn_original_weights_new_navigation_800.{gif,lmp}` |
| Independent native replay verification | `recording_replay.json` |
| Checked report and identities | `benchmark.json`, `benchmark_table.md`, `manifest.json` |
| Engine checks | `integration/results.json`, `integration.log` |

Run from this experiment worktree, using the already installed rootless image.
The full reproduction sequence requires a fresh experiment output location:
the scripts deliberately reject existing result/checkpoint files. Preserve the
existing artifacts rather than rerunning these commands over them. Supply the
original two full-game checkpoints and `demonstrations_42.npz` at their existing
paths. No baseline retraining is needed.

```bash
./run-nixos.sh test
./run-nixos.sh python scripts/navigation-integration.py
./run-nixos.sh python scripts/map01-navigation.py baseline
./run-nixos.sh python scripts/map01-navigation.py teacher
./run-nixos.sh python scripts/map01-dagger.py
./run-nixos.sh python scripts/map01-navigation.py candidate
./run-nixos.sh python scripts/navigation-replay.py
python scripts/map01-report.py
```

The report uses only the host Python standard library and Git; gameplay and
training run inside the existing container.

The candidate phase benchmarks the updated QNN and disabled branch separately
and records the first successful candidate evaluation seed, asserting that the
recorded episode again produces a genuine exit. It also records the recommended
original checkpoint with the new planner. The replay script verifies both native
demos independently. Single-run alternatives with
new result directories:

```bash
./run-nixos.sh full-eval --policy quantum --navigation legacy \
  --checkpoint outputs/full_game/models/quantum.pt \
  --episodes 20 --seed 800 --skill 1 --frame-skip 4 --max-steps 1000 \
  --output-dir outputs/reproduction_original_qnn
./run-nixos.sh full-eval --policy quantum --navigation improved \
  --checkpoint outputs/full_game/models/quantum_map01_dagger.pt \
  --episodes 20 --seed 800 --skill 1 --frame-skip 4 --max-steps 1000 \
  --output-dir outputs/reproduction_candidate_qnn
./run-nixos.sh full-eval --policy quantum --navigation improved --quantum-disabled \
  --checkpoint outputs/full_game/models/quantum_map01_dagger.pt \
  --episodes 20 --seed 800 --skill 1 --frame-skip 4 --max-steps 1000 \
  --output-dir outputs/reproduction_quantum_disabled
./run-nixos.sh full-eval --policy teacher --navigation legacy --teacher-version legacy \
  --episodes 3 --seed 600 --output-dir outputs/reproduction_diagnostic_teacher \
  --trace-dir outputs/reproduction_diagnostic_teacher/traces
./run-nixos.sh python scripts/navigation-diagnostics.py
```

The runtime remains ViZDoom 1.3.2, PyTorch 2.14.1+cpu and Python 3.11.17.
Check `manifest.json` for source, checkpoint, dataset and recording hashes.
Generated weights, traces, datasets and videos remain local ignored artifacts;
a compact measured report is in `docs/results/map01_navigation.json`.

## Limitations and next experiment

The teacher still times out on seed 813 at the second east-side door: 796 stuck
observations, with repeated use pulses while facing the door edge. Its saved
contact sheet and trace show the failure. Fine grid clearance and a use/release
pulse do not provide door-open state, a wait-until-open phase or portal centering.
This remaining issue is documented rather than changed after the two-iteration
budget was consumed.

The candidate's seed 809 timeout was also recorded: 636 stuck observations and
506 attack actions with zero selected-weapon ammo. Screenshots show prolonged
fist attacks near a wall while an enemy is visible. It later recovers movement,
but too late to complete. Thus retaining combat weights does not eliminate
ammo-starvation and poor-positioning failure states. The candidate loses exits
on seeds 809, 810 and 813 relative to the unchanged weights/new planner, while
recovering seed 801. No additional fitting was attempted to repair these cases.

Narrow-depression handling uses axis-aligned sector bounds and is an approximation,
not a full Doom collision engine. Static planning still does not solve arbitrary
moving sectors, keys, switch dependencies, lifts, teleports or solid map objects.
MAP02 and MAP30 received live smoke checks, not completion benchmarks. No broader
campaign completion or pixels-only navigation is claimed.

The next experiment should add a map-independent door interaction state machine
with portal centering, a bounded open wait and a use cooldown. Validate it on new
diagnostic seeds and MAP02 before any additional DAgger collection, and reserve
fresh evaluation seeds: 800–819 have now been inspected for this study.
