# HERMES: GPU-Accelerated Multi-Level Heat Solver for Metal Additive Manufacturing

HERMES is a **GPU-accelerated, three-level multi-resolution transient heat solver** for **Laser Powder Bed Fusion (LPBF)** additive manufacturing.  
It efficiently simulates thermal fields during laser scanning with moving nested grids, leveraging **CuPy, Numba-CUDA, and custom CUDA kernels**.

This solver is being developed as part of a research project on **fast, scalable LPBF simulation**, targeting large-scale studies of melt-pool geometry, thermal gradients, and solidification rates. Results are validated against experimental benchmarks and will appear in an upcoming paper.

---

## Features
- Multi-level nested domains (Level-1, Level-2, Level-3) with different grid resolutions.  
- Domains move with the laser in the global frame.  
- Achieves relative errors as low as 1e-5 on thermal gradients and cooling rates for parts on the order of centimeters.  
- Flexible time stepping: **CFL-based** or **fixed dt** (from config). 
- Configurable laser parameters, grid sizes, and material properties via `sim.ini`.  
- Flexible **laser path definition**: straight, raster, segments, explicit waypoints, or traced from a picture.
- Path preview utility to verify trajectories before running simulations.
- Default material = **316L stainless steel**, with optional overrides.  
- Runs on **NVIDIA GPUs** using CuPy + Numba-CUDA  
  (tested on TACC Vista / Grace-Hopper GPUs and TACC Lonestar6 / A100 GPUs). 

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


##  Laser Path Preview

Before running a simulation, you can preview the laser path to ensure it matches your intended trajectory. The script `preview_path.py` (in `src/hermes/laser_path/`) reads a `laser_path.ini` file and generates a figure of the path.
#### Usage
```bash
cd src/hermes/laser_path
python3 preview_path.py --config /path/to/laser_path.ini
```
Options:
``` text
--samples-per-seg   Interpolation samples per segment (default 80)
--label-every       Label every Nth point with index (0 disables, default 200)
```
There are mainly 6 path modes. Examples for each can be found in `configs/paths_all_modes.ini`:
1) **single line** – diagonal or straight scan.
2) **raster (x-major)** – zigzag in x, stepping in y.
3) **raster (y-major)** – zigzag in y, stepping in x.
4) **segments** – sequence of directional moves with lengths.
5) **explicit waypoints** – manually listed coordinates.
6) **picture** – trace a shape from a binary image (e.g. a logo or silhouette).

Example preview showing all six modes (with the starting location marked as 0):


In **picture mode**, the image is converted to grayscale, thresholded, and contours are extracted (using `scikit-image`). A zigzag infill path is then created inside the shape, scaled to user-specified horizontal and/or vertical dimensions.

## Configuration
Simulation parameters are controlled by two files:
- `configs/sim.ini` &rarr; physical, material, solver parameters.
-  `configs/laser_path.ini` &rarr; laser path definition (any of the 6 modes above).


## Running
From the repo root, first move into the `scripts/` directory:
```bash
cd src/hermes/scripts
```
- Default run (uses `configs/sim.ini` and `configs/laser_path.ini`):
```bash
python3 multi_level_solver.py
```
-  Custom config:
```bash
python3 multi_level_solver.py --config /path/to/other_sim.ini --laser_path /path/to/other_laser.ini
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
#### Usage
```bash
cd src/hermes/post/
python3 surface_export.py --help
```
Options:
```text
  -h, --help       show this help message and exit
  --output_path OUTPUT_PATH      Path to output tag dir (contains 'snapshots/'). Example: /abs/.../outputs/demo_run
  --steps STEPS     Which steps: 'all', 'last', 'N', 'N:M', or comma list '10,20,30'. Default: last
  --write-temp     Also write temperature volume as VTK ImageData.
  --skip-G         Do not generate G surface VTK (even if G_flat exists).
  --skip-R         Do not generate R surface VTK (even if R_flat exists).
  --config CONFIG  Path to sim.ini. If omitted, tries PATH/sim.ini; else falls back to repo
                   configs/sim.ini.
  --layers LAYERS  Which layers to export: 'all', 'L', 'L1,L2', or 'A:B'. Default: all              
```
**Important:**
- `--config` must be the same `.ini` file used for the simulation run.
This ensures consistency in material parameters (`Ts`, `Tl`, `ΔT`), length/time scaling, etc.
 - By default G and R VTK files are written.
- The temperature volume (T) is only written if `--write-temp` is provided.
- The resulting VTK files can be opened directly in ParaView for visualization.
## Example Run 1

Example case: 
A **3 mm straight scan in the y-direction** with laser power **Q = 70 W** and beam radius **rb = 25 µm**, velocity **v = 1.0 m/s**.  
The simulation was run on TACC Vista (Grace Hopper GPU) using:

```bash
# Run solver
python3 src/hermes/scripts/multi_level_solver.py \
  --config configs/sim_ex1.ini \
  --laser_path configs/path_laser_ex1.ini

# Post-process outputs into VTK
python3 src/hermes/post/surface_export.py \
  --path outputs/example1_Q70_r25_v1 \
  --config configs/sim_ex1.ini \
  --write-temp
  ```
  The results were then visualized in Paraview:
- **Temperature field** (cut 20 µm below the surface)  
- Profiles of **thermal gradient (G)** and **solidification velocity (R)** along the melt-pool surface  
(from the deepest point of the melt pool to the trailing end)

<p align="center"> <img src="Figs_example1/Temperature_example1.png" alt="Temperature field example" width="500"/> </p> <p align="center"> <img src="Figs_example1/GR_example1.png" alt="G and R profiles example" width="500"/> </p>

Example console output from this run:
```text
[info] Layers available: [1]; selected: [1]
[info] Steps available: [17858]; selected: [17858]
[info] sim.ini: /work/09143/halperen/vista/hermom/HERMOM/configs/sim_ex1.ini
[info] Ts=1658.0 K, Tl=1723.0 K, ΔT=65.0 K, len_scale=0.023057365490123424, time_scale=147.14671993697004
[info] Writing under: /work/09143/halperen/vista/hermom/HERMOM/outputs/example1_Q70_r25_v1/VTK

=== layer 1 — step 000017858 ===
  Melt extents [μm]: x=78.0, y=360.0, z=34.5
  Deepest y-plane index = 185
  Wrote G surface: /work/09143/halperen/vista/hermes-gpu-heat/outputs/example1_Q70_r25_v1/VTK/G/vtkG_layer_1_step_000017858.vtk
  Wrote R surface: /work/09143/halperen/vista/hermes-gpu-heat/outputs/example1_Q70_r25_v1/VTK/R/vtkR_layer_1_step_000017858.vtk
  Wrote temperature volume: /work/09143/halperen/vista/hermes-gpu-heat/outputs/example1_Q70_r25_v1/VTK/T/T_step_000017858.vtk

[done]
```

## Example Run 2
The formation of **UT** and **Longhorn** shapes, on the order of centimeters, was also simulated for 10 successive layers (layer thickness = 100 µm).  
For visualization, multiple picture paths (U, T, Longhorn) were concatenated and the globally melted points across layers were tracked. Example visualizations are shown below:


<p align="center">
  <img src="UT_Longhorn/UT_Longhorn_10Layers_Sim.png" alt="Multi-layer UT-Longhorn simulation (10 layers)" width="700"/>
</p>

<p align="center">
  <video controls width="700">
    <source src="UT_Longhorn/UT_Longhorn_OneLayer_WithZoomInner.mp4" type="video/mp4">
  </video>
</p>


> **Note:** This concatenation/marking steps is not part of the public solver but can be easily replicated by users if desired.


## Notes
- Default material: 316L stainless steel
- Units: m, mm, µm, nm are supported in config
- Timestep: choose either CFL or dt (not both)
- Nested grids: Level 3 (outer) → Level 2 → Level 1


## Author
Hikmet Alperen Aydin whose advised by Prof. George Biros
