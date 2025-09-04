from __future__ import annotations
import os, re, math, glob
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from tvtk.api import tvtk, write_data
from vtk.util import numpy_support
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator

# our existing config + phys
from hermes.runtime.config import load_config
from hermes.physics.material import phys_parameter

# -------------------------------
# Helpers to build phys from ini
# -------------------------------
def build_phys_from_config(cfg_path: str | Path) -> phys_parameter:
    rc = load_config(cfg_path)
    Q  = rc.laser.Q
    rb = rc.laser.x_span_m            # beam radius (m)
    t  = rc.laser.t_spot_on
    mat_over = rc.material.to_override_dict()
    return phys_parameter(Q, rb, t, mat_over)

# -------------------------------
# Load one saved step
# Names expected:
#   u_s_{xsp}_{step}.npy
#   u_s_old_{xsp}_{step}.npy
#   x_s_{xsp}_{step}.npy, y_s..., z_s...
#   Optional: G_{xsp}_{step}.npy / R_{xsp}_{step}.npy  (3D)
#             G_gpu_{xsp}_{step}.npy / R_gpu_{xsp}_{step}.npy (flat)
# -------------------------------
def load_step(run_dir: str | Path, xsp: str, step: int) -> Dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    def npy(name): return np.load(run_dir / name)
    suffix = f"_{xsp}_{step}.npy"

    data = {
        "x_s":      npy(f"x_s{suffix}"),
        "y_s":      npy(f"y_s{suffix}"),
        "z_s":      npy(f"z_s{suffix}"),
        "u_s":      npy(f"u_s{suffix}"),
        "u_s_old":  npy(f"u_s_old{suffix}"),
    }

    # Try to load G/R (3D first)
    G3 = run_dir / f"G{suffix}"
    R3 = run_dir / f"R{suffix}"
    Gf = run_dir / f"G_gpu{suffix}"
    Rf = run_dir / f"R_gpu{suffix}"

    if G3.exists() and R3.exists():
        data["G"] = np.load(G3)     # (nx-2,ny-2,nz-2)
        data["R"] = np.load(R3)
        data["_gr_is_flat"] = False
    elif Gf.exists() and Rf.exists():
        data["G_flat"] = np.load(Gf)  # (Nint,)
        data["R_flat"] = np.load(Rf)
        data["_gr_is_flat"] = True
    else:
        # It's okay if G/R were not saved yet; caller can skip those outputs
        data["_gr_is_flat"] = None

    return data

def reshape_fortran(vec: np.ndarray, shape: Tuple[int,int,int]) -> np.ndarray:
    return np.reshape(vec, shape, order="F")

# -------------------------------
# Depth-peak plane (your method)
# melt_pool_indices from u>1 (nondim)
# -------------------------------
def deepest_y_plane_index(ny: int,
                          melt_pool_indices: Tuple[np.ndarray,np.ndarray,np.ndarray],
                          z_s: np.ndarray) -> int:
    max_depths = []
    for y_plane in range(1, ny - 1):
        z_idx = melt_pool_indices[2][melt_pool_indices[1] == y_plane]
        if len(z_idx) > 0:
            max_depth = np.max(z_s[z_idx]) - np.min(z_s[z_idx])
        else:
            max_depth = -1
        max_depths.append(max_depth)
    max_depths = np.array(max_depths)
    max_depth_value = np.max(max_depths)
    candidates = np.where(max_depths == max_depth_value)[0]
    return int(candidates[len(candidates)//2])

# -------------------------------
# Units helpers (nondim -> dim)
# -------------------------------
def temp_dim(u_nd: np.ndarray, Ts: float, dT: float) -> np.ndarray:
    return u_nd * dT + Ts  # K

def G_dim_per_um(G_nd: np.ndarray, dT: float, len_scale: float) -> np.ndarray:
    # nondim gradient -> K/m, then to K/µm
    return (G_nd * dT / len_scale) / 1e6

def R_dim_m_per_s(R_nd: np.ndarray, len_scale: float, time_scale: float) -> np.ndarray:
    # nondim R -> (len_scale/time_scale) * R (m/s)
    return R_nd * (len_scale / time_scale)

# -------------------------------
# Write VTK: isosurface with scalar
# -------------------------------
def write_surface_with_scalar(verts: np.ndarray,
                              faces: np.ndarray,
                              scalar: np.ndarray,
                              scalar_name: str,
                              out_path: str | Path) -> None:
    points_vtk = tvtk.Points()
    points_vtk.from_array(verts)

    cells_vtk = tvtk.CellArray()
    for f in faces:
        cells_vtk.insert_next_cell(len(f), f.tolist())

    poly = tvtk.PolyData(points=points_vtk, polys=cells_vtk)
    scalars = numpy_support.numpy_to_vtk(np.asarray(scalar))
    scalars.SetName(scalar_name)
    poly.point_data.scalars = scalars

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_data(poly, str(out_path))

# -------------------------------
# Write VTK: full temperature volume
# -------------------------------
def write_temperature_volume(u_nd_3d: np.ndarray,
                             x_s: np.ndarray, y_s: np.ndarray, z_s: np.ndarray,
                             Ts: float, dT: float,
                             out_path: str | Path) -> None:
    # origin & spacing from axes
    hx = float(x_s[1]-x_s[0]); hy = float(y_s[1]-y_s[0]); hz = float(z_s[1]-z_s[0])
    origin = (float(x_s[0]), float(y_s[0]), float(z_s[0]))
    tempK = temp_dim(u_nd_3d, Ts, dT)

    grid = tvtk.ImageData(spacing=(hx, hy, hz), origin=origin, dimensions=tempK.shape)
    grid.point_data.scalars = tempK.ravel(order="F")
    grid.point_data.scalars.name = "temperature_K"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_data(grid, str(out_path))

# -------------------------------
# Main per-step processor
# -------------------------------
def process_step(run_dir: str | Path,
                 cfg_path: str | Path,
                 xsp: str,
                 step: int,
                 *,
                 make_G=True,
                 make_R=True,
                 make_T=True,
                 plane_align_to_y0=True,
                 out_root: Optional[str | Path] = None) -> Dict[str,str]:
    """
    Returns dict of written file paths (keys: G, R, T) if created.
    """
    phys = build_phys_from_config(cfg_path)
    dT   = phys.deltaT
    Ts   = phys.Ts
    len_scale = phys.len_scale
    time_scale = phys.time_scale

    dat = load_step(run_dir, xsp, step)
    x_s, y_s, z_s = dat["x_s"], dat["y_s"], dat["z_s"]
    u_s = dat["u_s"]; u_old = dat["u_s_old"]

    nx, ny, nz = x_s.size, y_s.size, z_s.size
    u3  = reshape_fortran(u_s, (nx, ny, nz))
    uo3 = reshape_fortran(u_old, (nx, ny, nz))

    # Prepare G/R arrays (interior)
    G_i = None; R_i = None
    if dat["_gr_is_flat"] is True:
        nxi, nyi, nzi = nx-2, ny-2, nz-2
        G_i = reshape_fortran(dat["G_flat"], (nxi, nyi, nzi))
        R_i = reshape_fortran(dat["R_flat"], (nxi, nyi, nzi))
    elif dat["_gr_is_flat"] is False:
        G_i = dat["G"]
        R_i = dat["R"]
    # else: None (not saved) → skip surfaces

    # melt pool (nondim: u>1)
    melt_idx = np.where(u_s > 1)[0]
    u3_nd = u3
    m3 = np.where(u3_nd - 1 > 1e-4)
    if len(m3[0]) == 0:
        # nothing melted; still can write volume T if requested
        mplane = 1
    else:
        mplane = deepest_y_plane_index(ny, m3, z_s)

    # restrict to 0..mplane in y for marching cubes
    u3_cut = u3_nd[:, 0:mplane+1, :]
    # spacing
    hx = float(x_s[1]-x_s[0]); hy = float(y_s[1]-y_s[0]); hz = float(z_s[1]-z_s[0])
    # run marching cubes at isovalue = Ts (in K) on dimensional temperature
    temp_cut_K = temp_dim(u3_cut, Ts, dT)
    verts, faces, normals, values = marching_cubes(
        temp_cut_K, level=Ts, spacing=(hx, hy, hz)
    )
    # shift to true coords (if axes don't start at 0)
    verts[:, 0] += x_s[0]
    verts[:, 1] += y_s[0]
    verts[:, 2] += z_s[0]

    # optionally align y top to 0 for visualization stacks
    if plane_align_to_y0:
        y_offset = 0.0 - np.max(verts[:, 1])
        verts_vis = verts.copy()
        verts_vis[:, 1] += y_offset
    else:
        verts_vis = verts

    # build output folder
    if out_root is None:
        out_root = Path(run_dir) / "vtk"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written = {}

    # ----- G surface -----
    if make_G and (G_i is not None):
        # interpolator expects interior grids
        xg = x_s[1:-1]; yg = y_s[1:-1]; zg = z_s[1:-1]
        interpG = RegularGridInterpolator((xg, yg, zg), G_i, bounds_error=False, fill_value=None)
        G_sample_nd = interpG(verts)   # nondim gradient
        G_sample = G_dim_per_um(G_sample_nd, dT, len_scale)  # K/µm

        outG = out_root / f"G/Q_{int(round(phys.Q))}/G_Q_{int(round(phys.Q))}_v_{phys_parameter.__name__ if False else 'v'}_xsp_{xsp}_{step:07d}.vtk"
        outG.parent.mkdir(parents=True, exist_ok=True)
        write_surface_with_scalar(verts_vis, faces, G_sample, "G_K_per_um", outG)
        written["G"] = str(outG)

    # ----- R surface -----
    if make_R and (R_i is not None):
        xr = x_s[1:-1]; yr = y_s[1:-1]; zr = z_s[1:-1]
        interpR = RegularGridInterpolator((xr, yr, zr), R_i, bounds_error=False, fill_value=None)
        R_sample_nd = interpR(verts)
        R_sample = R_dim_m_per_s(R_sample_nd, len_scale, time_scale)  # m/s
        outR = out_root / f"R/Q_{int(round(phys.Q))}/R_Q_{int(round(phys.Q))}_xsp_{xsp}_{step:07d}.vtk"
        outR.parent.mkdir(parents=True, exist_ok=True)
        write_surface_with_scalar(verts_vis, faces, R_sample, "R_m_per_s", outR)
        written["R"] = str(outR)

    # ----- Temperature volume -----
    if make_T:
        outT = out_root / f"U/Q_{int(round(phys.Q))}/U_Q_{int(round(phys.Q))}_xsp_{xsp}_{step:07d}.vtk"
        outT.parent.mkdir(parents=True, exist_ok=True)
        write_temperature_volume(u3_nd, x_s, y_s, z_s, Ts, dT, outT)
        written["T"] = str(outT)

    return written

