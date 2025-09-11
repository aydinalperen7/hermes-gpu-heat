from __future__ import annotations
import configparser
from pathlib import Path

from .path_builders import (
    build_single_line_nd,
    build_segments_nd,
    build_raster_nd,
    build_waypoints_nd,
    build_picture_nd,
)
from hermes.runtime.config import parse_length_expr  

def build_waypoints_nd_from_ini(path_ini: str | Path, len_scale: float) -> "np.ndarray":
    """
    Returns non-dimensional waypoints (shape (P,2), Fortran-compatible array ok)
    built from whichever [path.*] section exists in `path_ini`.

    Priority order (first match wins): single, raster_x, raster_y, segments, waypoints, picture.
    """
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with open(path_ini, "r") as f:
        cfg.read_file(f)

    # 1) single
    if "path.single" in cfg:
        sec = cfg["path.single"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        length_m = parse_length_expr(sec["length"])
        dir_str = sec.get("dir", "+y")
        return build_single_line_nd(
            start_xy_m=(x0, y0),
            length_m=length_m,
            dir_str=dir_str,
            len_scale=len_scale,
        )

    # 2) raster x-major
    if "path.raster_x" in cfg:
        sec = cfg["path.raster_x"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        width_m  = parse_length_expr(sec["width"])
        height_m = parse_length_expr(sec["height"])
        line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
        passes = int(sec["passes"]) if "passes" in sec else None
        x_dir = int(sec.get("x_dir", "1"))
        return build_raster_nd(
            origin_m=(x0, y0), width_m=width_m, height_m=height_m,
            line_pitch_m=line_pitch_m, passes=passes,
            x_dir_sign=x_dir, x_major=True, len_scale=len_scale
        )

    # 3) raster y-major
    if "path.raster_y" in cfg:
        sec = cfg["path.raster_y"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        width_m  = parse_length_expr(sec["width"])
        height_m = parse_length_expr(sec["height"])
        line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
        passes = int(sec["passes"]) if "passes" in sec else None
        y_dir = int(sec.get("y_dir", "1"))
        return build_raster_nd(
            origin_m=(x0, y0), width_m=width_m, height_m=height_m,
            line_pitch_m=line_pitch_m, passes=passes,
            x_dir_sign=y_dir, x_major=False, len_scale=len_scale
        )

    # 4) segments
    if "path.segments" in cfg:
        sec = cfg["path.segments"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        repeat = int(sec.get("repeat", "1"))
        segments = []
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
        return build_segments_nd(
            start_xy_m=(x0, y0),
            segments=segments,
            repeat=repeat,
            len_scale=len_scale,
        )

    # 5) explicit waypoints
    if "path.waypoints" in cfg:
        sec = cfg["path.waypoints"]
        # points: "(0mm,0mm), (0mm,3mm), (1mm,3mm), (1mm,0mm)"
        raw = sec["points"]
        # local parse (kept minimal)
        def _parse_pts(s: str):
            out = []
            for chunk in s.split(")"):
                c = chunk.strip().lstrip(",").strip()
                if not c: continue
                if c[0] == "(":
                    c = c[1:]
                xy = [t.strip() for t in c.split(",")]
                if len(xy) != 2: continue
                out.append((parse_length_expr(xy[0]), parse_length_expr(xy[1])))
            return out
        pts_m = _parse_pts(raw)
        close_loop = sec.get("close_loop", "false").strip().lower() in {"1","true","on","yes"}
        return build_waypoints_nd(waypoints_m=pts_m, close_loop=close_loop, len_scale=len_scale)

    # 6) picture
    if "path.picture" in cfg:
        sec = cfg["path.picture"]
        img_path = sec["image"]
        horizontal_m = parse_length_expr(sec["horizontal"])
        vertical_m = parse_length_expr(sec["vertical"]) if "vertical" in sec else None
        n = int(sec.get("n", "100"))  # your density knob
        return build_picture_nd(
            img_path=img_path,
            horizontal_length_m=horizontal_m,
            vertical_length_m=vertical_m,
            len_scale=len_scale,
            n=n,
        )

    raise ValueError("No [path.*] section found in the provided INI.")
