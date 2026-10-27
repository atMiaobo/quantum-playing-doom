# Optional small NixOS module. In an existing flake-based configuration,
# either import this module or just add its single setting to configuration.nix.
{ ... }: {
  virtualisation.podman.enable = true;
}
