{
  description = "Deadly Corridor QNN: NixOS Podman development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      # The bundled Linux container and dependency wheels are targeted at x86_64.
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShellNoCC {
        name = "doom-corridor-shell";
        packages = with pkgs; [
          podman
          bash
          coreutils
          git
          python311 # Convenient for inspecting data; training runs in the container.
          unzip
        ];

        shellHook = ''
          echo ""
          echo "DOOM Corridor QNN shell ready (NixOS / Podman)."
          echo "  ./run-nixos.sh build"
          echo "  ./run-nixos.sh doctor"
          echo "  ./run-nixos.sh imitate --model quantum --episodes 35 --epochs 8"
          echo ""
          echo "The first two commands require rootless Podman to be enabled in NixOS."
          echo "See docs/NIXOS.md for one-time setup."
          echo ""
        '';
      };

      # Optional reusable module for the user's *separate* NixOS system flake.
      # Users can equivalently add virtualisation.podman.enable = true;
      nixosModules.podman = import ./nixos/podman.nix;
    };
}
