# GH Node Reproducible Setup

## 1) Create environment (one-time)

```bash
cd hermes-gpu-heat
bash gh_node_scripts/create_env_gh.sh
```

If `hermes_clean` already exists and you want to refresh it:

```bash
cd hermes-gpu-heat
bash gh_node_scripts/create_env_gh.sh
```

## 2) Run solver on any GH node

Option A: wrapper script

```bash
cd hermes-gpu-heat
bash gh_node_scripts/run_solver_gh.sh sim_ex1.ini path_laser_ex1.ini jacobi none # jacobi preconditioner for level 2, none for level 3
```


Option B: direct Python run (equivalent, but you must set environment variables manually once)

```bash
cd hermes-gpu-heat
source ~/.bashrc >/dev/null 2>&1 || true
conda activate hermes_clean
module purge
module load cuda
module load gcc
export CUDA_HOME="${TACC_CUDA_DIR:-/home1/apps/nvidia/Linux_aarch64/24.7/cuda/12.5}"
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
python3 src/hermes/scripts/multi_level_solver.py --config sim_ex1.ini --laser_path path_laser_ex1.ini -precond-level2 jacobi --precond-level3 none
```


`gh_node_scripts/run_solver_gh.sh` automatically sets:
- `module load cuda`
- `module load gcc`
- `CUDA_HOME`
- `NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so`
- `LD_LIBRARY_PATH` including CUDA, NVVM, and math libs
- `PYTHONPATH` to `src`

## 3) Quick test (optional)

```bash
cd hermes-gpu-heat
timeout 30s bash gh_node_scripts/run_solver_gh.sh sim_ex1.ini path_laser_ex1.ini
```

Expected behavior: startup lines print and process continues until timeout kills it.
