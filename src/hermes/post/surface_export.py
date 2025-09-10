#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hermes post-process to VTK:
- Reads PATH/snapshots/step_XXXXXX.{npz|npy}
- Reconstructs u_s, x_s, y_s, z_s (Fortran order)
- Builds phys from sim.ini via load_config -> phys_parameter
- Finds deepest melt plane (u>1), extracts Tl isosurface, interpolates G/R
- Writes:
    PATH/VTK/G/vtkG_step_XXXXXX.vtk      (unless --skip-G)
    PATH/VTK/R/vtkR_step_XXXXXX.vtk      (unless --skip-R)
    PATH/VTK/T/UT_step_XXXXXX.vtk        (if --write-temp)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# VTK / TVTK
from tvtk.api import tvtk, write_data
from vtk.util import numpy_support

# SciPy / scikit-image
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator

# ---- import project helpers ----
# config loader & phys
from hermes.runtime.config import load_config
from hermes.physics.material import phys_parameter


# --------------------------
# Utils
# --------------------------

def parse_index_spec(spec: str, available: List[int]) -> List[int]:
    """Parse 'all' | 'N' | 'A:B' | 'a,b,c' to a sorted unique list filtered by `available`."""
    s = spec.strip().lower()
    if s == "all":
        return sorted(available)
    if re.match(r"^\d+:\d+$", s):
        a, b = map(int, s.split(":"))
        sel = [v for v in available if a <= v <= b]
        return sorted(sel)
    if re.match(r"^\d+(,\d+)*$", s):
        wanted = set(map(int, s.split(",")))
        sel = [v for v in available if v in wanted]
        return sorted(sel)
    if s.isdigit():
        val = int(s)
        return [val] if val in available else []
    raise ValueError(f"Unrecognized index spec: {spec}")

def detect_layered_scheme(snapshot_dir: Path) -> bool:
    """Return True if files look like 'layer_<L>_step_<STEP>_*.npy' or '.npz'."""
    # any layered npz?
    if any(snapshot_dir.glob("layer_*_step_*.npz")):
        return True
    # any layered npy (we look for u_s as representative)
    if any(snapshot_dir.glob("layer_*_step_*_u_s.npy")):
        return True
    return False

def list_indexed(snapshot_dir: Path) -> Tuple[bool, List[Tuple[Optional[int], int]]]:
    """
    Return (layered, items) where items is a list of (layer, step) tuples.
    If layered == False, layer is None and step is the old scheme's step.
    """
    items: set[Tuple[Optional[int], int]] = set()
    layered = detect_layered_scheme(snapshot_dir)

    if layered:
        # Accept both .npz and .npy sets
        # npz: layer_<L>_step_<STEP>.npz
        for p in snapshot_dir.glob("layer_*_step_*.npz"):
            m = re.match(r"layer_(\d+)_step_(\d+)\.npz$", p.name)
            if m:
                L = int(m.group(1)); S = int(m.group(2))
                items.add((L, S))
        # npy: layer_<L>_step_<STEP>_u_s.npy (presence of u_s means the set exists)
        for p in snapshot_dir.glob("layer_*_step_*_u_s.npy"):
            m = re.match(r"layer_(\d+)_step_(\d+)_u_s\.npy$", p.name)
            if m:
                L = int(m.group(1)); S = int(m.group(2))
                items.add((L, S))
    else:
        # Legacy scheme: step_<STEP>.npz, or step_<STEP>_u_s.npy
        for p in snapshot_dir.glob("step_*.npz"):
            m = re.match(r"step_(\d+)\.npz$", p.name)
            if m:
                S = int(m.group(1))
                items.add((None, S))
        for p in snapshot_dir.glob("step_*_u_s.npy"):
            m = re.match(r"step_(\d+)_u_s\.npy$", p.name)
            if m:
                S = int(m.group(1))
                items.add((None, S))

    return layered, sorted(items, key=lambda t: (t[0] or 0, t[1]))

def load_indexed(snapshot_dir: Path, layer: Optional[int], step: int):
    """
    Load one snapshot. Works with both naming schemes, layer_x_step_y or step_y.

    Returns: dict with keys 'u_s','u_s_old','x_s','y_s','z_s' and optional 'G_flat','R_flat'.
    """
    data = {}
    if layer is not None:
        base_npz = snapshot_dir / f"layer_{layer}_step_{step:09d}.npz"
        if base_npz.exists():
            with np.load(base_npz) as Z:
                for k in ("u_s","u_s_old","x_s","y_s","z_s"):
                    data[k] = Z[k]
                if "G_flat" in Z: data["G_flat"] = Z["G_flat"]
                if "R_flat" in Z: data["R_flat"] = Z["R_flat"]
            return data

        # fall back to individual .npy
        req = {
            "u_s":     snapshot_dir / f"layer_{layer}_step_{step:09d}_u_s.npy",
            "u_s_old": snapshot_dir / f"layer_{layer}_step_{step:09d}_u_s_old.npy",
            "x_s":     snapshot_dir / f"layer_{layer}_step_{step:09d}_x_s.npy",
            "y_s":     snapshot_dir / f"layer_{layer}_step_{step:09d}_y_s.npy",
            "z_s":     snapshot_dir / f"layer_{layer}_step_{step:09d}_z_s.npy",
        }
        for k, p in req.items():
            if not p.exists():
                # Also try 6-digit for robustness
                alt = snapshot_dir / f"layer_{layer}_step_{step:06d}_{k}.npy"
                if not alt.exists():
                    raise FileNotFoundError(f"Missing required file: {p} (or {alt})")
                data[k] = np.load(alt)
            else:
                data[k] = np.load(p)

        for optk in ("G_flat","R_flat"):
            p9  = snapshot_dir / f"layer_{layer}_step_{step:09d}_{optk}.npy"
            p6  = snapshot_dir / f"layer_{layer}_step_{step:06d}_{optk}.npy"
            if p9.exists():
                data[optk] = np.load(p9)
            elif p6.exists():
                data[optk] = np.load(p6)
        return data

    # Legacy scheme (no layer)
    base_npz = snapshot_dir / f"step_{step:06d}.npz"
    if base_npz.exists():
        with np.load(base_npz) as Z:
            for k in ("u_s","u_s_old","x_s","y_s","z_s"):
                data[k] = Z[k]
            if "G_flat" in Z: data["G_flat"] = Z["G_flat"]
            if "R_flat" in Z: data["R_flat"] = Z["R_flat"]
        return data

    req = {
        "u_s":     snapshot_dir / f"step_{step:06d}_u_s.npy",
        "u_s_old": snapshot_dir / f"step_{step:06d}_u_s_old.npy",
        "x_s":     snapshot_dir / f"step_{step:06d}_x_s.npy",
        "y_s":     snapshot_dir / f"step_{step:06d}_y_s.npy",
        "z_s":     snapshot_dir / f"step_{step:06d}_z_s.npy",
    }
    for k, p in req.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for step {step}: {p}")
        data[k] = np.load(p)

    for optk in ("G_flat","R_flat"):
        p = snapshot_dir / f"step_{step:06d}_{optk}.npy"
        if p.exists():
            data[optk] = np.load(p)
    return data

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def shape_back_3d(u_flat: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Flattened Fortran-ordered -> (nx, ny, nz) array."""
    return np.reshape(u_flat, (nx, ny, nz), order="F")



def G2L_3D_arr(idxx: np.ndarray, nx: int, ny: int, nz: int):
    """Flat (Fortran) indices -> (i,j,k)."""
    k = idxx // (nx * ny)
    idx_z = idxx - k * nx * ny
    j = idx_z // nx
    i = idx_z - j * nx
    return i, j, k

def max_depth_calculator(ny_s: int, melt_pool_indices, z_s: np.ndarray) -> int:
    """Pick y-plane with max melt depth; tie -> middle index among maxima."""
    max_depths = []
    for y_plane in range(1, ny_s - 1):
        z_idx = melt_pool_indices[2][melt_pool_indices[1] == y_plane]
        if len(z_idx) > 0:
            max_depth = np.max(z_s[z_idx]) - np.min(z_s[z_idx])
        else:
            max_depth = -1
        max_depths.append(max_depth)
    max_depths = np.array(max_depths)
    y_candidates = np.where(max_depths == np.max(max_depths))[0]
    return int(y_candidates[len(y_candidates) // 2])

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hermes post-process: export Ts isosurface with G/R and temperature volume to VTK (phys auto-loaded from sim.ini)."
    )
    ap.add_argument("--output_path", required=True,
                    help="Path to output tag dir (contains 'snapshots/'). Example: /abs/.../outputs/demo_run")
    ap.add_argument("--steps", default="last",
                    help="Which steps: 'all', 'last', 'N', 'N:M', or comma list '10,20,30'. Default: last")
    ap.add_argument("--write-temp", action="store_true",
                    help="Also write temperature volume as VTK ImageData.")
    ap.add_argument("--skip-G", action="store_true",
                    help="Do not generate G surface VTK (even if G_flat exists).")
    ap.add_argument("--skip-R", action="store_true",
                    help="Do not generate R surface VTK (even if R_flat exists).")
    ap.add_argument("--config", default=None,
                    help="Optional path to sim.ini. If omitted, tries PATH/sim.ini; else falls back to repo configs/sim.ini.")
    ap.add_argument("--layers", default="all",
                help="Which layers to export: 'all', 'L', 'L1,L2', or 'A:B'. Default: all")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    base_path =  (project_root / args.path).resolve()
    snapshot_dir = base_path / "snapshots"
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshots folder not found: {snapshot_dir}")

    # ---- locate config ----
    cfg_path = None
    if args.config:
        c =  (project_root / args.config).resolve()
        if not c.is_file():
            raise FileNotFoundError(f"--config not found: {c}")
        cfg_path = c
    else:
        # try next to outputs tag dir
        local_cfg = base_path / "sim.ini"
        if local_cfg.is_file():
            cfg_path = local_cfg
        else:
            # fallback to repo default (../../../../configs/sim.ini relative to this file)
            repo_root = Path(__file__).resolve().parents[3]
            default_cfg = repo_root / "configs" / "sim.ini"
            if default_cfg.is_file():
                cfg_path = default_cfg
            else:
                raise FileNotFoundError("Could not find sim.ini (tried PATH/sim.ini and repo configs/sim.ini). "
                                        "Pass --config explicitly.")

    # ---- build phys from config (same as solver) ----
    rc = load_config(cfg_path)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)

    Ts = phys.Ts
    Tl = phys.Tl
    deltaT = phys.deltaT
    len_scale = phys.len_scale
    time_scale = phys.time_scale

    def temp_dim(u_nd):    # nondimensional -> K
        return u_nd * deltaT + Ts

    # ---- discover available (layer, step) pairs ----
    
    layered, pairs = list_indexed(snapshot_dir)
    if not pairs:
        raise RuntimeError(f"No snapshots found in {snapshot_dir}")
   
    avail_layers = sorted(set(L for (L, _) in pairs if L is not None)) if layered else []
    avail_steps  = sorted(set(S for (_, S) in pairs))

    sel_steps = args.steps.strip().lower()
    if sel_steps == "last":
        wanted_steps = [avail_steps[-1]]
    else:
        wanted_steps = parse_index_spec(sel_steps, avail_steps)
    if not wanted_steps:
        raise SystemExit(f"No steps matched --steps={args.steps!r} among {avail_steps}")

    if layered:
        wanted_layers = parse_index_spec(args.layers, avail_layers) if args.layers != "all" else avail_layers
        if not wanted_layers:
            raise SystemExit(f"No layers matched --layers={args.layers!r} among {avail_layers}")
    else:
        wanted_layers = [None]  # legacy

    print(f"[info] layered naming: {layered}")
    if layered:
        print(f"[info] Layers available: {avail_layers}; selected: {wanted_layers}")
    print(f"[info] Steps available: {avail_steps[:8]}{'...' if len(avail_steps)>8 else ''}; selected: {wanted_steps}")

    sel_pairs = [(L, S) for (L, S) in pairs
                 if (S in wanted_steps) and ((L in wanted_layers) if layered else True)]
    if not sel_pairs:
        raise SystemExit("No (layer, step) pairs matched your selection.")

    # ---- output dirs ----
    out_root = ensure_dir(base_path / "VTK")
    out_G = ensure_dir(out_root / "G")
    out_R = ensure_dir(out_root / "R")
    out_T = ensure_dir(out_root / "T") if args.write_temp else None

    print(f"[info] sim.ini: {cfg_path}")
    print(f"[info] Ts={Ts} K, Tl={Tl} K, ΔT={deltaT} K, len_scale={len_scale}, time_scale={time_scale}")
    # print(f"[info] Steps: {steps}")
    print(f"[info] Writing under: {out_root}")
    

    for L, step in sel_pairs:
        if layered:
            print(f"\n=== layer {L} — step {step:09d} ===")
        else:
            print(f"\n=== step {step:06d} ===")

        D = load_indexed(snapshot_dir, L, step)

        u_s      = D["u_s"]       # flat (Fortran order)
        u_s_old  = D["u_s_old"]
        x_s      = D["x_s"]
        y_s      = D["y_s"]
        z_s      = D["z_s"]

        nx, ny, nz = len(x_s), len(y_s), len(z_s)
        u3d     = shape_back_3d(u_s, nx, ny, nz)

        # spacings (code units -> dimensional via len_scale when used)
        hx = float(x_s[1] - x_s[0])
        hy = float(y_s[1] - y_s[0])
        hz = float(z_s[1] - z_s[0])

        # Melt pool & deepest plane
        melt_idx_flat = np.where(u_s > 1.0)[0]
        if melt_idx_flat.size == 0:
            print("  [warn] No melted cells (u>1). Skipping isosurface.")
            if args.write_temp:
                Tdim = temp_dim(u3d)
                grid = tvtk.ImageData(
                    spacing=(hx*len_scale, hy*len_scale, hz*len_scale),
                    origin=(x_s[0]*len_scale, y_s[0]*len_scale, z_s[0]*len_scale),
                    dimensions=Tdim.shape,
                )
                scal = numpy_support.numpy_to_vtk(Tdim.ravel(order="F"))
                scal.SetName("temperature_K")
                grid.point_data.scalars = scal
                fname = f"T_{('layer_%d_'%L) if layered else ''}step_{step:09d}.vtk"
                outT = out_T / fname
                write_data(grid, str(outT))
                print(f"  Wrote temperature volume: {outT}")
            continue

        xi, yi, zi = G2L_3D_arr(melt_idx_flat, nx, ny, nz)
        mx = (x_s[max(xi)] - x_s[min(xi)]) * len_scale
        my = (y_s[max(yi)] - y_s[min(yi)]) * len_scale
        mz = (z_s[max(zi)] - z_s[min(zi)]) * len_scale
        print(f"  Melt extents [μm]: x={mx*1e6:.1f}, y={my*1e6:.1f}, z={mz*1e6:.1f}")

        melt_pool_indices = np.where(u3d - 1.0 > 1e-4)
        deepest_y_plane = max_depth_calculator(ny, melt_pool_indices, z_s)
        print(f"  Deepest y-plane index = {deepest_y_plane}")

        # Dimensional coords
        x_dim = x_s * len_scale
        y_dim = y_s * len_scale
        z_dim = z_s * len_scale

        # marching cubes on u up to deepest plane, at dimensional Ts
        u3d_cut = u3d[:, :deepest_y_plane+1, :]
        Tdim_cut = temp_dim(u3d_cut)
        verts, faces, normals, values = marching_cubes(
            Tdim_cut,
            level=Tl,
            spacing=(hx*len_scale, hy*len_scale, hz*len_scale),
        )
        verts[:, 0] += x_dim[0]
        verts[:, 1] += y_dim[0]
        verts[:, 2] += z_dim[0]

        current_max_y = np.max(verts[:, 1])
        common_max_y = 0
        y_offset = common_max_y - current_max_y
        verts_visual = verts.copy()
        verts_visual[:, 1] += y_offset

        # Interior grids (for G/R interpolation)
        nx_i, ny_i, nz_i = nx - 2, ny - 2, nz - 2
        x_i = x_dim[1:-1]
        y_i = y_dim[1:-1]
        z_i = z_dim[1:-1]

        # ---- G surface ----
        if (not args.skip_G) and ("G_flat" in D):
            G_flat = D["G_flat"]
            G = G_flat.reshape((nx_i, ny_i, nz_i), order="F")
            G_interp = RegularGridInterpolator((x_i, y_i, z_i), G,
                                               bounds_error=False, fill_value=None)
            G_vals = G_interp(verts)  # nondimensional gradient magnitude
            # Output G in K/μm: (ΔT / length)
            G_K_per_um = (G_vals * deltaT) / (len_scale * 5e6)

            pts = tvtk.Points(); pts.from_array(verts_visual)
            cells = tvtk.CellArray()
            for f in faces:
                cells.insert_next_cell(len(f), f.tolist())
            poly = tvtk.PolyData(points=pts, polys=cells)
            scal = numpy_support.numpy_to_vtk(G_K_per_um.astype(np.float64))
            scal.SetName("G_K_per_um")
            poly.point_data.scalars = scal
            fname = f"vtkG_{('layer_%d_'%L) if layered else ''}step_{step:09d}.vtk"
            outG = out_G / fname
            write_data(poly, str(outG))
            print(f"  Wrote G surface: {outG}")
        else:
            if args.skip_G:
                print("  [info] Skipping G by user request.")
            else:
                print("  [info] No G_flat in snapshot; skipping G surface.")

        # ---- R surface ----
        if (not args.skip_R) and ("R_flat" in D):
            R_flat = D["R_flat"]
            R = R_flat.reshape((nx_i, ny_i, nz_i), order="F")
            R_interp = RegularGridInterpolator((x_i, y_i, z_i), R,
                                               bounds_error=False, fill_value=None)
            R_vals = R_interp(verts)  # nondim cooling rate along grad direction
            # Output R in m/s: R_dim ≈ (len_scale / time_scale).
            R_um_per_us = (R_vals * len_scale) / time_scale  # (μm/μs) since both scales are SI

            pts = tvtk.Points(); pts.from_array(verts_visual)
            cells = tvtk.CellArray()
            for f in faces:
                cells.insert_next_cell(len(f), f.tolist())
            poly = tvtk.PolyData(points=pts, polys=cells)
            scal = numpy_support.numpy_to_vtk(R_um_per_us.astype(np.float64))
            scal.SetName("R_um_per_us")
            poly.point_data.scalars = scal
            fname = f"vtkR_{('layer_%d_'%L) if layered else ''}step_{step:09d}.vtk"
            outR = out_R / fname
            write_data(poly, str(outR))
            print(f"  Wrote R surface: {outR}")
        else:
            if args.skip_R:
                print("  [info] Skipping R by user request.")
            else:
                print("  [info] No R_flat in snapshot; skipping R surface.")

        # ---- Temperature volume ----
        if args.write_temp:
            Tdim = temp_dim(u3d)
            grid = tvtk.ImageData(
                spacing=(hx*len_scale, hy*len_scale, hz*len_scale),
                origin=(x_dim[0], y_dim[0], z_dim[0]),
                dimensions=Tdim.shape,
            )
            scal = numpy_support.numpy_to_vtk(Tdim.ravel(order="F"))
            scal.SetName("temperature_K")
            grid.point_data.scalars = scal
            outT = out_T / f"T_step_{step:09d}.vtk"
            write_data(grid, str(outT))
            print(f"  Wrote temperature volume: {outT}")

    print("\n[done]")

if __name__ == "__main__":
    main()

