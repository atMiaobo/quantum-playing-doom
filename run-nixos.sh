#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${DOOM_CORRIDOR_IMAGE:-localhost/doom-corridor-qnn:latest}"
if [[ $# -eq 0 || $1 == "help" || $1 == "--help" ]]; then
  cat <<'HELP'
DOOM Deadly Corridor QNN (NixOS, rootless Podman)
  ./run-nixos.sh build
  ./run-nixos.sh doctor
  ./run-nixos.sh eval --policy teacher --episodes 1 --skill 1 --record outputs/teacher.gif
  ./run-nixos.sh imitate --model quantum --episodes 35 --epochs 8 --skill 1
  ./run-nixos.sh eval --policy quantum --episodes 5 --record outputs/quantum.gif
  ./run-nixos.sh ppo --model quantum --steps 25000 --skill 1
  ./run-nixos.sh compare --episodes 10 --skill 1
  ./run-nixos.sh test
Use --skill 5 to evaluate at official corridor difficulty.
HELP
  exit 0
fi
if [[ -n "${DOOM_CORRIDOR_ENGINE:-}" ]]; then ENGINE="$DOOM_CORRIDOR_ENGINE"
elif command -v podman >/dev/null; then ENGINE=podman
elif command -v docker >/dev/null; then ENGINE=docker
else echo 'Missing podman/docker. Enable virtualisation.podman.enable = true; in NixOS config' >&2; exit 1
fi
if ! "$ENGINE" info >/dev/null 2>&1; then
  echo "Cannot access $ENGINE; run '$ENGINE info' and check rootless setup." >&2; exit 1
fi
build() { "$ENGINE" build -t "$IMAGE" -f "$DIR/Dockerfile" "$DIR"; }
if [[ "$1" == "build" ]]; then build; exit 0; fi
if ! "$ENGINE" image inspect "$IMAGE" >/dev/null 2>&1; then build; fi
mkdir -p "$DIR/outputs/models"
args=(run --rm -v "$DIR:/app" -w /app -e HOME=/tmp)
if [[ "$ENGINE" == podman ]]; then args+=(--userns=keep-id)
else args+=(--user "$(id -u):$(id -g)"); fi
exec "$ENGINE" "${args[@]}" "$IMAGE" "$@"
