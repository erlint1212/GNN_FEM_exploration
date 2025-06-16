# flake.nix
{
  description = "Development environment for Master Project with Python and CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # Or specify a specific commit/tag
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import nixpkgs with necessary configuration
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true; # Note: Effectiveness might depend on specific package versions/overrides
          };
        };

        # Define the Python environment
        python = pkgs.python3;
        pythonEnv = python.withPackages (ps: [
          ps.numpy
          ps.sympy
          ps.matplotlib
          ps.pandas
          ps.fenics # Note: FEniCS might have specific dependencies or setup requirements
          ps.python-dotenv
          ps.scipy
          ps.pip
          ps.scikit-learn
          ps.conda # Including conda alongside pip and venv might be redundant? Consider simplifying.
          ps.virtualenv
          ps.sphinx
          ps.sphinx-autodoc-typehints
          ps.sphinx-rtd-theme
          ps.pynvml
          ps.networkx
        ]);

        # List of native dependencies
        nativeBuildInputs = with pkgs; [
          cudatoolkit           # Provides CUDA runtime/compiler
          # Use cudaPackages for consistency if available and suitable
          (linuxPackagesFor pkgs.linux_latest).nvidia_x11 # Ensure this matches your kernel/driver setup
          cudaPackages.cudnn    # cuDNN library

          # Graphics and Xorg Libraries
          libGLU
          libGL
          xorg.libXi
          xorg.libXmu
          freeglut
          xorg.libXext
          xorg.libX11
          xorg.libXv
          xorg.libXrandr

          # Standard Libraries and Tools
          zlib
          ncurses5 # Older ncurses, ensure it's needed
          stdenv.cc             # C Compiler (gcc)
          binutils              # Binary utilities
          ffmpeg                # For media processing
          conda                 # Conda package manager
          paraview              # Visualization tool
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = nativeBuildInputs ++ [ pythonEnv ]; # Combine native deps and python env

          shellHook = ''
            unset PYTHONPATH

            # Set a custom prompt
            export PS1="\n\[\033[1;32m\][python3_masterProject_flake:\w]\$\[\033[0m\] "

            # Set CUDA related environment variables
            # Ensure pkgs.cudatoolkit points to the correct path in the Nix store
            export CUDA_HOME=${pkgs.cudatoolkit}
            # Add CUDA compiler/tools to PATH
            export PATH="$CUDA_HOME/bin:$PATH"

            # Add necessary library paths
            # NVIDIA driver libs (Adjust if using a different driver package)
            export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"
             # CUDA toolkit libs
            export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH" # Changed from cudatoolkit/lib64 based on typical structure
             # CUDNN libs (often symlinked within CUDA_HOME/lib or needs its own path)
            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"
            # Standard C++ library path
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            # Add other required library paths if needed (e.g., libGL, X11 libs)
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH"
            # Add more paths for other Xorg libs if required by applications


            echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" # Debugging output

            # --- Python Virtual Environment and Pip Installs ---
            # Note: This part reduces reproducibility compared to a pure Nix approach.

            # Create and activate venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating Python virtual environment (.venv)..."
              ${pythonEnv}/bin/python -m venv .venv
            fi

            # Check if already activated, otherwise activate
            if [ -z "$VIRTUAL_ENV" ]; then
                echo "Activating virtual environment..."
                source .venv/bin/activate
            else
                echo "Virtual environment already active."
            fi


            echo
            echo "Installing/Verifying PyTorch & PyG (pip)..."
            # Upgrade pip within the venv
            pip install --upgrade pip

            # Install torch, torchvision, torchaudio with CUDA 12.4 support
            # Check if torch is already installed and matches desired version/device
            # This simple check might not be robust enough for version/cuda compatibility
            if ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
                echo "Installing PyTorch..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            else
                echo "PyTorch with CUDA seems already installed."
            fi

            # Install torch-geometric and dependencies compatible with torch 2.6.0 and CUDA 12.4
            # Similar check for pyg
             if ! python -c "import torch_geometric" &> /dev/null; then
                echo "Installing PyG..."
                pip install torch-geometric torch-scatter torch-sparse torch-cluster pyg-lib -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__.split("+")[0])')+cu124.html
             else
                echo "PyG seems already installed."
             fi


            echo
            echo "--- Environment Check ---"
            echo "Python interpreter: $(which python)"
            echo "Testing basic torch functionality..."
            (python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('cuDNN version:', torch.backends.cudnn.version())") || echo "*** Torch check failed! ***"

            # Check if the test file exists before running it
            if [ -f "./tests/test_pyg.py" ]; then
              echo "Testing torch-geometric..."
              python ./tests/test_pyg.py || echo "*** PyG check failed! ***"
            else
              echo "Skipping PyG test: ./tests/test_pyg.py not found."
            fi
            echo "--- End Check ---"
            echo
            echo "Flake environment ready (with venv and pip installs)"
            echo
          '';
        };
      }
    );
}
