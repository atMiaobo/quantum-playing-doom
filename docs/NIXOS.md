# NixOS execution

The development flake supplies Podman and command-line tools on x86_64 Linux.
PyTorch and ViZDoom run inside a conventional Linux container. The development
shell does not install the Python training dependencies on the host.

Enable rootless Podman in the host's own NixOS configuration:

```nix
virtualisation.podman.enable = true;
```

The optional `nixos/podman.nix` module supplies this setting. Rebuild the host
using its normal configuration and log in again if required for rootless setup.
The project flake does not replace the host's system flake.

From the repository root:

```bash
nix develop
./run-nixos.sh build
./run-nixos.sh test
./run-nixos.sh doctor
./run-nixos.sh eval --policy quantum \
  --checkpoint pretrained/quantum_skill3_candidate.pt \
  --skill 3 --episodes 1 --record outputs/quantum_skill3.gif
```

The launcher resolves its own location, builds the image if missing, mounts the
checkout at `/app`, and uses Podman's `--userns=keep-id`. Output files stay in
`outputs/` with the host user's ownership. All runtime commands accept the same
arguments as `python -m corridor_qnn`. `DOOM_CORRIDOR_IMAGE` changes the image tag;
`DOOM_CORRIDOR_ENGINE=docker` selects Docker and uses the host UID/GID instead.
An image build and dependency installation require network access.

If Podman is unavailable, inspect `podman info` and the host rootless setup.
The image uses `docker.io/library/python:3.11-slim-bookworm` to avoid unqualified
registry prompts. Headless SDL drivers are configured in the image. On ordinary
Linux, native installation is also supported as described in the README.

The initializer uses only Python's standard library and can run on the host:

```bash
python3 scripts/init-checkpoints.py
./run-nixos.sh eval --policy quantum --skill 3 --episodes 5
```
