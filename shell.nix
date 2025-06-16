# shell.nix

# If you are not using nixos, ignore this file
# TODO: Make this into a flake to make it reproducible, offender: pkgs = import <nixpkgs> {};
# Using LD_LIBRARY_PATH may lead to weird errors if the libc version of the shell doesn’t match the one of the system. For a dev shell that uses <nixpkgs> it shouldn’t be an issue, but otherwise I’d recommend using nix-ld. https://ayats.org/blog/nix-workflow#poetry

let
  pkgs = import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
  };

  python = pkgs.python312;

  pythonEnv = python.withPackages (ps: [
    ps.gmsh
    ps.numpy
    ps.sympy
    ps.matplotlib
    ps.pandas
    ps.fenics
    ps.h5py # Add the h5py Python package, Nix should build it against ps.hdf5
    ps.meshio
    ps.python-dotenv
    ps.scipy
    ps.pip
    ps.scikit-learn
    ps.conda
    ps.virtualenv
    ps.sphinx
    ps.sphinx-autodoc-typehints
    ps.sphinx-rtd-theme
    ps.pynvml
    ps.networkx
  ]);
in
pkgs.mkShell {
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
    pythonEnv
    pkgs.paraview
    pkgs.hdf5
    pkgs.pkg-config
  ];

  shellHook = ''
    unset PYTHONPATH
    export PS1="\n\[\033[1;32m\][python3_masterProject:\w]\$\[\033[0m\]"
    export CUDA_HOME=${pkgs.cudatoolkit}
    export PATH="$CUDA_HOME/bin:$PATH"

    export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
    export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv) with --system-site-packages..."
    ${pythonEnv}/bin/python -m venv .venv --system-site-packages
    if [ $? -ne 0 ]; then echo "*** ERROR: Create .venv FAILED. ***"; exit 1; fi
    echo "Venv created."
    if [ ! -f ".venv/bin/activate" ]; then echo "*** ERROR: .venv/bin/activate NOT created. ***"; exit 1; fi
  else
    echo ".venv directory already exists."
  fi

  echo "--- Python Path for base pythonEnv (${pythonEnv}/bin/python) ---"
  ${pythonEnv}/bin/python -c "import sys; import pprint; print('Base pythonEnv sys.path:'); pprint.pprint(sys.path)"

  echo "Attempting to activate .venv ..."
  source .venv/bin/activate
  ACTIVATE_EXIT_CODE=$? 
  
  if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment activated (VIRTUAL_ENV is set to: $VIRTUAL_ENV)."
    if [ $ACTIVATE_EXIT_CODE -ne 0 ]; then
      echo "*** WARNING: 'source .venv/bin/activate' finished with exit code ($ACTIVATE_EXIT_CODE), likely due to 'hash -r'. Proceeding. ***"
    fi

    # --- MODIFIED SOLUTION FOR PYTHONPATH ---
    # Use the simpler and more robust way to get pythonEnv's site-packages.
    # Remove the problematic line that used: python -c "import site; print(...)"
    
    # This line uses shell globbing and echo, which is safer from Nix parsing errors.
    PYTHONENV_SITE_PACKAGES=$(echo "${pythonEnv}/lib/python"*/site-packages)

    if [ -d "$PYTHONENV_SITE_PACKAGES" ]; then # Check if the determined path is a directory
        echo "Adding pythonEnv site-packages to PYTHONPATH: $PYTHONENV_SITE_PACKAGES"
        if [ -n "$PYTHONPATH" ]; then
          export PYTHONPATH="$PYTHONENV_SITE_PACKAGES:$PYTHONPATH"
        else
          export PYTHONPATH="$PYTHONENV_SITE_PACKAGES"
        fi
        echo "PYTHONPATH is now: $PYTHONPATH"
    else
        echo "*** WARNING: Could not determine pythonEnv site-packages directory: $PYTHONENV_SITE_PACKAGES was not a directory. FEniCS might not be found. ***"
        echo "pythonEnv path was: ${pythonEnv}"
    fi
    # --- END OF MODIFIED SOLUTION ---

  else
    echo "*** ERROR: Failed to activate .venv (VIRTUAL_ENV not set, exit code: $ACTIVATE_EXIT_CODE). Exiting. ***"
    exit 1 
  fi

  echo
  echo "--- Python Path for activated .venv ($(which python)) ---"
  python -c "import sys; import pprint; print('Activated .venv sys.path (after PYTHONPATH mod):'); pprint.pprint(sys.path)"
  echo
  # echo "Installing h5py (pip)..."
  # Set HDF5_DIR to help pip find the HDF5 library from Nix
  # export HDF5_DIR=${pkgs.hdf5}
  # Tell pip where to find HDF5 headers and libs
  # export C_INCLUDE_PATH="${pkgs.hdf5}/include''${C_INCLUDE_PATH:+:''$C_INCLUDE_PATH}"
  # export LIBRARY_PATH="${pkgs.hdf5}/lib''${LIBRARY_PATH:+:''$LIBRARY_PATH}
  # Install h5py, forcing it to build from source
  # pip install --no-binary=h5py h5py
  # echo

  # ... (Pip installs and other checks remain the same as your last full working version) ...
  echo "Installing PyTorch (pip)..."
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install torch-geometric torch-scatter torch-sparse torch-cluster pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
  pip install torchdiffeq
  echo
  echo "--- Environment Check ---"
  echo "Testing basic torch functionality"
  (python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())") || echo "*** Torch check failed! ***"
  echo "Testing torch-geometric"
  if [ -f "./tests/test_pyg.py" ]; then
    python ./tests/test_pyg.py || echo "*** PyG check failed! ***"
  else
    echo "Skipping PyG test: ./tests/test_pyg.py not found."
  fi
  echo
  echo "--- Testing FEniCS import directly from pythonEnv (${pythonEnv}/bin/python) ---"
  if ${pythonEnv}/bin/python -c "import fenics" &> /dev/null; then
    echo "--- FEniCS available in Nix pythonEnv (Confirmed) ---"
  else
    echo "*** ERROR: FEniCS NOT available in Nix pythonEnv! (Unexpected) ***"
  fi
  
  echo "--- Testing FEniCS import from .venv ---"
  (python -c "import fenics; import dolfin; print('SUCCESS: FEniCS and dolfin imported from .venv Python!')") || echo "*** ERROR: FEniCS/dolfin import from .venv Python FAILED! ***"
  echo
  echo "--- End Check ---"
  echo
  echo "Environment ready"
  echo
'';
}

