# HERMES: GPU-Accelerated Multi-Level Heat Solver for Metal Additive Manufacturing

HERMES is a **GPU-accelerated, three-level multi-resolution transient heat solver** for **Laser Powder Bed Fusion (LPBF)** additive manufacturing. It is designed to efficiently simulate thermal fields during laser scanning with moving nested grids, leveraging **CuPy, Numba-CUDA, and custom CUDA kernels**.

This solver is being developed as part of a research project on **fast, scalable LPBF simulation**, targeting large-scale studies of melt-pool geometry, thermal gradients, and solidification rates. Results are validated against experimental benchmarks and will appear in an upcoming paper.

---

## 🔑 Features
- Multi-level nested domains (Level-1, Level-2, Level-3) with different grid resolutions  
- Domains move with the laser in the global frame. 
- Achieving up to 1e-5 relative errors on thermal gradients and cooling rates for parts in the order of centimeters
- Flexible time stepping: **CFL-based** or **fixed dt** (from config)  
- Configurable laser parameters, grid sizes, and material properties via `sim.ini`  
- Default material = **316L stainless steel**, with optional overrides  
- Runs on **NVIDIA GPUs** using CuPy + Numba-CUDA (tested on TACC Vista / Grace-Hopper GPUs and TACC Lonestar6 / A100 GPUs)  

---

## 📦 Installation

### On TACC Vista (Grace Hopper GPUs)

One-time environment setup:
```bash
module purge
module load cuda/12
module load gcc/12.2

# Miniconda setup
conda create -n test_env python==3.11
conda activate test_env

python -m pip install --upgrade pip
pip install "cython<3"
pip install psutil pyyaml scipy
pip install cupy-cuda12x numba

