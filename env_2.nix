# shell.nix

# If you are not using nixos, ignore this file
# TODO: Make this into a flake to make it reproducible, offender: pkgs = import <nixpkgs> {};

let
  pkgs = import <nixpkgs> { 
      config.allowUnfree = true; 
      config.cudaSupport = true;
  };
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.fenics
      python-pkgs.torch
      python-pkgs.torch-geometric
    ]))
  ];
  shellHook = ''
    export PS1="\n\[\033[1;32m\][python3_masterProject:\w]\$\[\033[0m\]"
    echo
    echo "Enviorment ready"
    echo "Activated virtual environment: $(which python)"
    echo
  '';
}
