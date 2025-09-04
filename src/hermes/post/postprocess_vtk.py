#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path

from hermes.post.process import process_step

def parse_steps(s: str):
    # "100", "100,200,300", "0:10000:50"
    if ":" in s:
        a, b, *rest = s.split(":")
        start = int(a); stop = int(b)
        step = int(rest[0]) if rest else 1
        return list(range(start, stop+1, step))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]

def main(argv=None):
    p = argparse.ArgumentParser(description="Post-process saved fields into VTK (G, R surfaces and temperature volume).")
    p.add_argument("--config", required=True, help="Path to sim.ini (to reconstruct phys).")
    p.add_argument("--run-dir", required=True, help="Directory where .npy files were saved.")
    p.add_argument("--xsp", required=True, help="The xsp token used in filenames (e.g. '50').")
    p.add_argument("--steps", required=True, help="Step index or list/range, e.g. '16458' or '100,200' or '0:10000:50'.")
    p.add_argument("--no-G", action="store_true", help="Do not write G surface.")
    p.add_argument("--no-R", action="store_true", help="Do not write R surface.")
    p.add_argument("--no-T", action="store_true", help="Do not write temperature volume.")
    p.add_argument("--out-root", default=None, help="Output root folder (default: <run-dir>/vtk).")
    args = p.parse_args(argv)

    steps = parse_steps(args.steps)
    for s in steps:
        written = process_step(
            run_dir=args.run_dir,
            cfg_path=args.config,
            xsp=args.xsp,
            step=s,
            make_G=not args.no_G,
            make_R=not args.no_R,
            make_T=not args.no_T,
            out_root=args.out_root
        )
        print(f"[step {s}] wrote:", written)

if __name__ == "__main__":
    main()

