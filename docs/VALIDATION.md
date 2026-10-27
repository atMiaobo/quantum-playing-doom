# Packaging validation

Validation was performed on 2026-10-27. The original project was read only;
file contents, permission modes, and modification times were compared against
a snapshot after packaging and were unchanged.

## Executed checks

- All Python source files passed syntax compilation; package modules imported.
- `python -m pytest -q`: **22 passed** (14 original tests and 8 packaging cases).
- Existing tests exercise preprocessing, privileged features, macro-actions,
  teacher decisions, quantum gradients, both policy architectures, checkpoint
  round trips, GAE terminal handling, and a small synthetic imitation fit.
- Packaging cases verify both bundled checkpoint hashes and forward passes,
  non-overwriting initialization, explicit imitation input/output paths, missing
  input errors, evaluation GIF and comparison JSON output, and a four-step PPO
  update with independent input/output checkpoints. The evaluation and PPO
  packaging tests use a **synthetic environment**, not live ViZDoom.
- All five source checkpoints loaded with `weights_only=True`, matched the
  six-qubit/two-layer architecture, and produced finite ten-action logits.
  The two distributed files remain byte-for-byte identical to their sources.
- A wheel was built and installed offline with
  `python -m pip install --no-deps --no-build-isolation --target <temporary-directory> .`.
  Imports, checkpoint loading, and the installed `doom-corridor --help` worked
  from outside the checkout. All six subcommand help pages parsed successfully.
- `bash -n run-nixos.sh` and `./run-nixos.sh --help` passed. A mocked Podman
  executable verified argument forwarding, repository mounts, `--userns=keep-id`,
  and execution from another directory with spaces in the checkout path.
- `nix-instantiate --parse flake.nix` and the corresponding parse check for
  `nixos/podman.nix` passed. The retained flake lockfile was not regenerated.
- The Dockerfile references the fully qualified Python image; required build
  context files, launcher permissions, local documentation links, and Git
  ignore rules were checked. Generated build, test, engine, and run artifacts
  were removed before delivery. Git staging was audited using a separate
  temporary Git directory; the deliverable contains no Git history.

## Execution environment and limits

The validation environment had Python 3.11.14. Cached container dependencies
provided PyTorch 2.14.1+cpu, NumPy 2.4.6, Pillow 12.3.0, pytest 9.1.1, and ViZDoom
1.3.2. These dependencies were used for checks only; the repository contains
no references to their local cache locations and requires no source checkout.

Podman execution was unavailable because the sandbox could not create its
runtime configuration directory under `/run/user`. A native
`python -m corridor_qnn doctor --max-steps 4` attempt ended in a signal-11
ViZDoom crash; its cause was not established. No successful live gameplay,
ViZDoom training, or gameplay recording is claimed. The GIF test exercised
recording with synthetic frames. Container build/run and `nix develop` were
not executed in this sandbox. Native system-package installation and fresh
network dependency downloads were not performed.

The README training and evaluation commands were checked against the parser,
path handling, and checkpoint format. Full training budgets and the 30-episode
historical evaluations were not rerun. The README results are the supplied
measurements, not new measurements from packaging. A complete historical
training schedule and per-episode evaluation logs were unavailable.

On a host with working Podman, the remaining live verification is:

```bash
nix develop
./run-nixos.sh build
./run-nixos.sh test
./run-nixos.sh doctor
./run-nixos.sh eval --policy quantum \
  --checkpoint pretrained/quantum_skill3_candidate.pt \
  --skill 3 --episodes 1 --record outputs/quantum_skill3.gif
```

## Scope of the public copy

The active CNN, structured features, variational circuit, teacher, imitation
loss and collection, PPO updates, action set, and environment rewards retain
their original implementation. Changes add package metadata and a console
command, explicit input/output checkpoint arguments, a standard-library
initializer, documentation, and packaging tests. Explicit missing checkpoint
inputs now fail rather than silently starting a new model. The unused
four-feature/three-action `DoomPolicy` from an earlier scenario was omitted.

The difficulty-3 candidate and skill-1 baseline are included. Duplicate
checkpoints, the older PPO checkpoint, historical PPO CSV, existing GIFs,
local engine configuration, caches, and private development directories are
excluded. No classical checkpoint was available. The checkpoint metadata and
SHA-256 manifest are in `pretrained/`.
