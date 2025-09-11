#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import configparser
import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from path_builders import (
    build_single_line_nd,
    build_segments_nd,
    build_raster_nd,
    build_waypoints_nd,
)
from path_builders import build_picture_nd


# -----------------------------
# Minimal length-unit parsing
# -----------------------------
_SI = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
_num_unit = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$")

def parse_length_expr(s: str) -> float:
    """Supports '96 * 50um', '12*50 um', '0.001', '1e-4m' -> meters"""
    parts = [p.strip() for p in str(s).split("*")]
    val = 1.0
    for p in parts:
        if not p:
            continue
        m = _num_unit.match(p)
        if not m:
            raise ValueError(f"Cannot parse length token: {p!r}")
        num = float(m.group(1))
        unit = (m.group(2) or "m").strip()
        if unit not in _SI:
            raise ValueError(f"Unknown unit {unit!r} in {p!r}; allowed {list(_SI)}")
        val *= num * _SI[unit]
    return val

def parse_tuple_list(s: str) -> List[Tuple[float, float]]:
    """
    Parse waypoints like:
      "(0mm,0mm), (0mm,3mm), (1mm,3mm), (1mm,0mm)"
    into list of (x_m, y_m).
    """
    pts = []
    for part in s.split(")"):
        part = part.strip().lstrip(",").strip()
        if not part:
            continue
        if part[0] == "(":
            part = part[1:]
        xy = [p.strip() for p in part.split(",")]
        if len(xy) != 2:
            continue
        x_m = parse_length_expr(xy[0])
        y_m = parse_length_expr(xy[1])
        pts.append((x_m, y_m))
    return pts

# -----------------------------
# Densify between waypoints
# -----------------------------
def densify_polyline(points_m: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """
    Given waypoints in meters shape (P,2), interpolate linearly
    samples_per_segment between each neighbor.
    """
    if points_m.shape[0] <= 1:
        return points_m.copy()
    segs = []
    for i in range(points_m.shape[0] - 1):
        a = points_m[i]
        b = points_m[i + 1]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        segs.append(a[None, :] * (1.0 - t[:, None]) + b[None, :] * (t[:, None]))
    segs.append(points_m[-1][None, :])
    return np.vstack(segs)

# -----------------------------
# Plot helper
# -----------------------------
def plot_path(ax, XY_m: np.ndarray, title: str, label_every: int = 50):
    ax.plot(XY_m[:, 0], XY_m[:, 1], ".", ms=3)
    if label_every and label_every > 0:
        for idx in range(0, XY_m.shape[0], label_every):
            ax.text(XY_m[idx, 0], XY_m[idx, 1], str(idx), fontsize=7, color="tab:blue")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

# -----------------------------
# Builders per mode
# -----------------------------
def build_path_single(cfg, len_scale):
    sec = cfg["path.single"]
    start = (parse_length_expr(sec.get("x0", "0")), parse_length_expr(sec.get("y0", "0")))
    length_m = parse_length_expr(sec["length"])
    dir_str = sec.get("dir", "+y")
    return build_single_line_nd(start_xy_m=start, length_m=length_m, dir_str=dir_str, len_scale=len_scale)

def build_path_raster_x(cfg, len_scale):
    sec = cfg["path.raster_x"]
    origin = (parse_length_expr(sec.get("x0", "0")), parse_length_expr(sec.get("y0", "0")))
    width_m  = parse_length_expr(sec["width"])
    height_m = parse_length_expr(sec["height"])
    line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
    passes = int(sec["passes"]) if "passes" in sec else None
    x_dir_sign = int(sec.get("x_dir", "1"))
    return build_raster_nd(
        origin_m=origin, width_m=width_m, height_m=height_m,
        line_pitch_m=line_pitch_m, passes=passes,
        x_dir_sign=x_dir_sign, x_major=True, len_scale=len_scale
    )

def build_path_raster_y(cfg, len_scale):
    sec = cfg["path.raster_y"]
    origin = (parse_length_expr(sec.get("x0", "0")), parse_length_expr(sec.get("y0", "0")))
    width_m  = parse_length_expr(sec["width"])
    height_m = parse_length_expr(sec["height"])
    line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
    passes = int(sec["passes"]) if "passes" in sec else None
    y_dir_sign = int(sec.get("y_dir", "1"))
    return build_raster_nd(
        origin_m=origin, width_m=width_m, height_m=height_m,
        line_pitch_m=line_pitch_m, passes=passes,
        x_dir_sign=y_dir_sign, x_major=False, len_scale=len_scale
    )

def build_path_segments(cfg, len_scale):
    sec = cfg["path.segments"]
    start = (parse_length_expr(sec.get("x0", "0")), parse_length_expr(sec.get("y0", "0")))
    repeat = int(sec.get("repeat", "1"))
    segments: List[dict] = []
    for key in sorted(k for k in sec if k.startswith("segment.")):
        line = sec.get(key, "").strip()
        if not line:
            continue
        toks = [t.strip() for t in line.split(",")]
        seg = {}
        for t in toks:
            if "=" not in t:
                continue
            k, v = [x.strip() for x in t.split("=", 1)]
            if k == "dir":
                seg["dir"] = v
            elif k == "length":
                seg["length_m"] = parse_length_expr(v)
            elif k == "vx":
                seg["vx"] = float(v)
            elif k == "vy":
                seg["vy"] = float(v)
        if "length_m" not in seg:
            raise ValueError(f"{key} missing length=...")
        if "dir" not in seg and not {"vx", "vy"} <= set(seg.keys()):
            raise ValueError(f"{key} needs either dir=... or (vx,vy)=...")
        segments.append(seg)
    return build_segments_nd(start_xy_m=start, segments=segments, repeat=repeat, len_scale=len_scale)

def build_path_waypoints(cfg, len_scale):
    sec = cfg["path.waypoints"]
    pts = parse_tuple_list(sec["points"])
    close_loop = sec.get("close_loop", "false").strip().lower() in {"1", "true", "yes", "on"}
    return build_waypoints_nd(waypoints_m=pts, close_loop=close_loop, len_scale=len_scale)

def build_path_picture(cfg, len_scale):
    sec = cfg["path.picture"]
    img_path = sec["image"]
    horizontal_m = parse_length_expr(sec["horizontal"])
    vertical_m = parse_length_expr(sec["vertical"]) if "vertical" in sec else None
    n = int(sec.get("n", 100))  # <--- pick up optional density from INI (default 100)
    print('n = ', n)
    return build_picture_nd(
        img_path=img_path,
        horizontal_length_m=horizontal_m,
        vertical_length_m=vertical_m,
        len_scale=len_scale,
        n=n,  # <--- pass through
    )

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preview laser path(s) defined in a path INI. Output PNG is saved next to this script."
    )
    parser.add_argument("--config", required=True, help="Path to path INI (e.g., configs/paths_demo.ini)")
    parser.add_argument("--samples-per-seg", type=int, default=80,
                        help="Interpolation samples per segment (for visual trace, default is 80).")
    parser.add_argument("--label-every", type=int, default=200,
                        help="Label every Nth point with index (0 to disable, default is 200).")
    args = parser.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with open(args.config, "r") as f:
        cfg.read_file(f)

    # We plot in meters directly; len_scale=1.0 (builders return ND which we treat as meters for preview).
    len_scale = 1.0

    # Collect whichever modes are present
    mode_defs = [
        ("1) single line",              "path.single",     build_path_single),
        ("2) raster (x-major)",         "path.raster_x",   build_path_raster_x),
        ("3) raster (y-major)",         "path.raster_y",   build_path_raster_y),
        ("4) segments",                 "path.segments",   build_path_segments),
        ("5) explicit waypoints",       "path.waypoints",  build_path_waypoints),
        ("6) picture (zig-zag fill)",   "path.picture",    build_path_picture),
    ]
    to_plot = []
    for title, section, builder in mode_defs:
        if section in cfg:
            try:
                w_nd = builder(cfg, len_scale)
                XY_m = w_nd * len_scale
                XY_dense = densify_polyline(XY_m, args.samples_per_seg)
                to_plot.append((title, XY_dense))
            except Exception as e:
                to_plot.append((f"{title} (error)", np.array([[0.0, 0.0]])))
                print(f"[warn] {title}: {e}")

    if not to_plot:
        raise SystemExit(
            "No path sections found in INI. Add one of [path.single], [path.raster_x], "
            "[path.raster_y], [path.segments], [path.waypoints], [path.picture]."
        )

    n = len(to_plot)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (title, XY) in zip(axs, to_plot):
        plot_path(ax, XY, title, args.label_every)

    for ax in axs[len(to_plot):]:
        ax.axis("off")

    fig.tight_layout()

    script_dir = Path(__file__).resolve().parent
    cfg_stem = Path(args.config).stem
    out_path = script_dir / f"preview_{cfg_stem}.png"
    fig.savefig(out_path, dpi=200)
    print(f"[ok] saved figure to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
