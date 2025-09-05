# HERMES: GPU-Accelerated Multi-Level Heat Solver for Metal Additive Manufacturing

HERMES is a **GPU-accelerated, three-level multi-resolution transient heat solver** for **Laser Powder Bed Fusion (LPBF)** additive manufacturing.  
It efficiently simulates thermal fields during laser scanning with moving nested grids, leveraging **CuPy, Numba-CUDA, and custom CUDA kernels**.

This solver is being developed as part of a research project on **fast, scalable LPBF simulation**, targeting large-scale studies of melt-pool geometry, thermal gradients, and solidification rates. Results are validated against experimental benchmarks and will appear in an upcoming paper.

---

## Features
- Multi-level nested domains (Level-1, Level-2, Level-3) with different grid resolutions  
- Domains move with the laser in the global frame  
- Achieves relative errors as low as 1e-5 on thermal gradients and cooling rates for parts on the order of centimeters  
- Flexible time stepping: **CFL-based** or **fixed dt** (from config)  
- Configurable laser parameters, grid sizes, and material properties via `sim.ini`  
- Default material = **316L stainless steel**, with optional overrides  
- Runs on **NVIDIA GPUs** using CuPy + Numba-CUDA  
  (tested on TACC Vista / Grace-Hopper GPUs and TACC Lonestar6 / A100 GPUs)  

---

## Installation
### On TACC Vista (Grace Hopper GPUs)

One-time environment setup:
```bash
module purge
module load cuda/12
module load gcc/12.2

# install miniconda if not already installed, then:
conda create -n test_env python==3.11
conda activate test_env

python -m pip install --upgrade pip
pip install "cython<3"
pip install psutil pyyaml scipy
pip install cupy-cuda12x numba
```
Then set:
```bash
conda activate test_env
ml cuda
ml gcc

# CUDA paths (Vista-specific)
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export CUDA_HOME=$TACC_CUDA_DIR
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH

# Add repo to Python path
export PYTHONPATH="/work/09143/halperen/vista/HERMES/src:$PYTHONPATH"
```
## Configuration
Simulation parameters are controlled by `configs/sim.ini`.
## Running
From the repo root, first move into the `scripts/` directory:
```bash
cd src/hermes/scripts
```
- Default run (uses `configs/sim.ini`):
```bash
python3 multi_level_solver.py
```
-  Custom config:
```bash
python3 multi_level_solver.py --config /path/to/other.ini
```


## Post-Processing
After the simulation, HERMES writes outputs under:
```bash
<dir>/<tag>/snapshots/
```
in either `.npy`; (multiple files per step) or `.npz` (compressed single file per step) format.

A dedicated post-processing script converts these into VTK files for visualization.
It extracts:

- Melt pool dimensions (x, y, z extents)
- Deepest melt-pool plane
- Distribution of G on melt pool surface
- Distribution of R on melt pool surface
- Temperature volume fields
### Usage
```bash
cd src/hermes/post/
python3 surface_export.py --help
```
```text
options:
  -h, --help       show this help message and exit
  --path PATH      Path to output tag dir (contains 'snapshots/'). Example: /abs/.../outputs/demo_run
  --steps STEPS    Which steps: 'all', 'last', 'N', 'N:M', or comma list '10,20,30'. Default: last
  --write-temp     Also write temperature volume as VTK ImageData.
  --skip-G         Do not generate G surface VTK (even if G_flat exists).
  --skip-R         Do not generate R surface VTK (even if R_flat exists).
  --config CONFIG  Optional path to sim.ini. If omitted, tries PATH/sim.ini; else falls back to repo
                   configs/sim.ini.

```
**Important:**
- `--config` must be the same `.ini` file used for the simulation run.
This ensures consistency in material parameters (`Ts`, `Tl`, `ΔT`), length/time scaling, etc.
 - By default G and R VTK files are written.
- The temperature volume (T) is only written if `--write-temp` is provided.

## Notes
- Default material: 316L stainless steel
- Units: m, mm, um, nm are supported in config
- Timestep: choose either CFL or dt (not both)
- Nested grids: Level 3 (outer) → Level 2 → Level 1


## Author
Hikmet Alperen Aydin whose advised by Prof. George Biros
