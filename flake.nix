{
  inputs = {
    # use nixos-unstable-2024-10-09
    nixpkgs.url = "github:NixOS/nixpkgs/c31898adf5a8ed202ce5bea9f347b1c6871f32d1";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }:
  let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [
        pkgs.python312
        pkgs.python312Packages.matplotlib
        pkgs.python312Packages.numpy
        pkgs.python312Packages.cvxpy
        pkgs.python312Packages.torch
        pkgs.python312Packages.seaborn
        pkgs.python312Packages.torchsummary
        pkgs.python312Packages.torchmetrics
        pkgs.python312Packages.python-lsp-server
        pkgs.python312Packages.scikit-learn
        pkgs.cairo
      ];
      shellHook = ''
      '';
    };
  };
}
