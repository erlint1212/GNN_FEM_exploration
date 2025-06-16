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
    pkgs.cudatoolkit
    pkgs.linuxPackages.nvidia_x11
    pkgs.cudaPackages.cudnn
    pkgs.libGLU 
    pkgs.libGL
    pkgs.xorg.libXi 
    pkgs.xorg.libXmu 
    pkgs.freeglut
    pkgs.xorg.libXext 
    pkgs.xorg.libX11 
    pkgs.xorg.libXv 
    pkgs.xorg.libXrandr 
    pkgs.zlib 
    pkgs.ncurses5 
    pkgs.stdenv.cc 
    pkgs.binutils
    pkgs.ffmpeg
    pkgs.conda
    pkgs.sphinx # Documentation
    #pkgs.paraview
    (pkgs.python312.withPackages (python-pkgs: [
      python-pkgs.numpy
      python-pkgs.sympy
      python-pkgs.matplotlib
      python-pkgs.pandas
      python-pkgs.fenics
      python-pkgs.python-dotenv
      python-pkgs.scipy
      python-pkgs.pip
      python-pkgs.torch-bin
      python-pkgs.torchvision-bin
      python-pkgs.torch-geometric #Broken package
      python-pkgs.scikit-learn
      #python-pkgs.torch
      python-pkgs.virtualenv
      #python-pkgs.tensorflowWithCuda
      #python-pkgs.keras
      #python-pkgs.tensorflow
      #python-pkgs.jax
      #python-pkgs.jaxlibWithCuda
      # For documentation
      python-pkgs.sphinx
      python-pkgs.sphinx-autodoc-typehints
      python-pkgs.sphinx-rtd-theme
      #python-pkgs.myst-parser
      python-pkgs.pynvml
      python-pkgs.networkx
    ]))
    pkgs.paraview
  ];
  shellHook = ''
    export PS1="\n\[\033[1;32m\][python3_masterProject:\w]\$\[\033[0m\]"
    export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cudatoolkit}/lib64:${pkgs.cudaPackages.cudnn}/lib:${pkgs.libGL}/lib:${pkgs.freeglut}/lib"
    # export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
    # export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
    # Create .venv if it doesn't exist
    if [ ! -d ".venv" ]; then
      echo "Creating Python virtual environment (.venv)..."
      # Use the python from Nix pkgs to create the venv
      ${pkgs.python312}/bin/python -m venv .venv
    fi

    # Activate venv
    source .venv/bin/activate

    # Install PyG dependencies via pip 
    # Ensure torch-bin version from Nix matches torch-2.4.0 from PyG wheel URL
    # Ensure CUDA setup matches cu121 from PyG wheel URL
    echo
    echo "--- ENVIORMENT CHECK ---"
    echo "Installing PyTorch"
    #pip install 
    echo "Ensuring PyG dependencies are installed via pip..."
    #pip install torch_geometric
    #pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    # pip freeze | %{$_.split('==')[0]} | %{pip install --upgrade $_} #upgrade all packages in venv to the latest version available in the Python Package Index (PyPI)
    echo "Attempting to check PyTorch CUDA status..."
    # Run python check in subshell to avoid polluting history/variables
    (python -c "import torch; import os; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CuDNN version: {torch.backends.cudnn.version()}'); print(f'LD_LIBRARY_PATH (sample): {os.environ.get(\"LD_LIBRARY_PATH\", \"Not Set\")[:100]}...')") || echo "*** Python/Torch check failed! ***"
    echo "--- END CHECK ---"
    echo
    echo "Enviorment ready"
    echo "Activated virtual environment: $(which python)"
    echo
  '';
}
