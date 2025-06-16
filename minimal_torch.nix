# minimal_torch_test.nix
let
  pkgs = import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true; # Might still influence non-torch-bin cuda packages
  };
in pkgs.mkShell {
  packages = [
    pkgs.python312
    pkgs.python312Packages.torch-bin # The package in question
    # Minimal CUDA dependencies for runtime
    pkgs.cudatoolkit
    pkgs.linuxPackages.nvidia_x11
    pkgs.cudaPackages.cudnn
    pkgs.libGL # Often needed
  ];
  shellHook = ''
    echo "Setting LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cudatoolkit}/lib64:${pkgs.cudaPackages.cudnn}/lib:${pkgs.libGL}/lib"
    echo "LD_LIBRARY_PATH set."
    echo "Running torch import test directly..."
  '';
  # Command to run directly: 'python -c "import torch; print(f\'Direct import torch version: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\')"'
}
