# Ideas

Main working folder is `./src/`

Run `./src/main.py`

## Documentation

Make documentation via make.bat file by writing `make html` in CLI
read the documentation here: [documentation](./docs/_build/html/index.html)

## Installation

I used python=3.12.8, the rest is in the `requirements.txt` file.

* PyTorch version: 2.6.0+cu124
* CuDNN version: 90100
* I used: NVIDIA GeForce RTX 4080 Laptop GPU

I used my `shell.nix` as a working enviorment, so if there is any errors look up the packages mentioned there.

My `nvidia-smi` for reference in case there are any dependency issues:
```
Fri May  9 12:58:28 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.77                 Driver Version: 565.77         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   37C    P3             18W /   80W |      15MiB /  12282MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1848      G   ...zqz19yv04-xorg-server-21.1.16/bin/X          4MiB |
+-----------------------------------------------------------------------------------------+

```

## Environment

-   **Operating System:** NixOS version 24.11 (Vicuna) (Build ID: `24.11.716947.26d499fc9f1d`).
-   **Environment Management:** The core environment was defined using a Nix shell (`shell.nix`) on NixOS 24.11. This setup provided foundational system packages, including the CUDA toolkit and a base Python 3.12 interpreter, derived from the Nixpkgs channel corresponding to this OS build. Within this Nix shell, a Python virtual environment (`.venv`) was created, inheriting system-site-packages from the Nix Python environment. Project-specific Python dependencies, including PyTorch and its ecosystem, were then installed into this virtual environment using `pip`. (The `shell.nix` and a comprehensive list of Python packages from `requirements.txt` are provided in Appendix X / Supplementary Materials).
-   **NVIDIA Driver:** Version 565.77.
-   **CUDA Toolkit (System & Compilation):** Version 12.4 (V12.4.99), provided by the `cudatoolkit` package in the Nix environment (corresponding to NixOS 24.11) and confirmed by `nvcc --version`. This version was available for compiling custom code or for tools linking against a system-level CUDA installation.
-   **Python Environment:**
    -   **Interpreter:** Python 3.12.8 (or "Python 3.12 as provided by NixOS 24.11").
    -   **Key Packages and Versions** (from `requirements.txt` reflecting the state within the `.venv`):
        -   PyTorch: `torch==2.6.0+cu124`
        -   Torchvision: `torchvision==0.21.0+cu124`
        -   Torchaudio: `torchaudio==2.6.0+cu124`
        -   PyTorch Geometric: `torch-geometric==2.6.1` (with components `pyg_lib==0.4.0+pt26cu124`, `torch_cluster==1.6.3+pt26cu124`, `torch_scatter==2.1.2+pt26cu124`, `torch_sparse==0.6.18+pt26cu124`)
        -   NVIDIA CUDA Libraries for PyTorch (via `pip`):
            -   CUDA Runtime: `nvidia-cuda-runtime-cu12==12.4.127`
            -   cuDNN: `nvidia-cudnn-cu12==9.1.0.70` (corresponding to cuDNN 9.1.0)
            -   Other components like `nvidia-cublas-cu12`, `nvidia-cufft-cu12`, etc., were also installed as per PyTorch's distribution for CUDA 12.4.
        -   FEniCS: `fenics-dolfin==2019.1.0`, `fenics-ffc==2019.1.0`, `fenics-ufl==2019.1.0`, `fenics-fiat==2019.1.0`, `fenics-dijitso==2019.1.0` (available from the Nix environment via system-site-packages).
        -   NumPy: `numpy==1.26.4`
        -   SciPy: `scipy==1.14.1`
        -   Pandas: `pandas==2.2.3`
        -   Matplotlib: `matplotlib==3.9.2`
        -   Scikit-learn: `scikit-learn==1.5.2`
        -   SymPy: `sympy==1.13.3`
        -   NetworkX: `networkx==3.3`
        -   Pynvml: `pynvml==11.5.3`

