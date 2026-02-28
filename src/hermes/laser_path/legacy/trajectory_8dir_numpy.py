#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy 8-direction trajectory stepper.

This implementation constrains motion to axis-aligned or diagonal grid steps
using fixed per-axis increments. It is kept for reference and reproducibility
of older solver behavior.

For current trajectory handling, use:
- `hermes.laser_path.trajectory_numpy`
- `hermes.laser_path.trajectory_cupy`
"""


from __future__ import annotations
import numpy as np


def _to_numpy_waypoints(waypoints_nd) -> np.ndarray:
    """Convert NumPy/CuPy-like waypoint arrays to a contiguous NumPy float array once."""
    if isinstance(waypoints_nd, np.ndarray):
        arr = waypoints_nd
    elif hasattr(waypoints_nd, "get"):
        # CuPy arrays expose .get() for explicit host transfer.
        arr = waypoints_nd.get()
    else:
        arr = np.asarray(waypoints_nd)

    arr = np.asarray(arr, dtype=float, order="C")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("waypoints_nd must have shape (N, 2)")
    return arr


class TrajectoryStepper:
    """
    Axis-wise stepping:
      - On each step, move by exactly ±velocity_nd on each axis *independently* if the
        remaining distance on that axis is >= velocity_nd; otherwise don't move on that axis.
      - Waypoint is considered reached when BOTH axes are within velocity_nd of target.
      - Flag = True  -> not yet reached on both axes (stay on same segment)
              False -> reached on both axes (next step targets next waypoint)
    Returns: (x, y, movement_x, movement_y, Flag)
    """

    def __init__(self, waypoints_nd):
        self.W = _to_numpy_waypoints(waypoints_nd)
        self.i = 0
        self.pos_x = float(self.W[0, 0])
        self.pos_y = float(self.W[0, 1])
        self.done = (self.W.shape[0] < 2)

    def current(self):
        return self.pos_x, self.pos_y

    def advance(self, velocity_nd: float):
        v = float(velocity_nd)

        if self.done:
            return self.pos_x, self.pos_y, 0, 0, False

        # If last segment is already finished, stay put
        if self.i >= self.W.shape[0] - 1:
            self.done = True
            return self.pos_x, self.pos_y, 0, 0, False

        # We may need to retarget immediately if we’re exactly at a waypoint.
        # Loop: if we’re at a waypoint, advance segment and re-evaluate, otherwise move once and return.
        while True:
            # Safety: if we ran out of segments by retargeting, stop
            if self.i >= self.W.shape[0] - 1:
                self.done = True
                return self.pos_x, self.pos_y, 0, 0, False

            qx = float(self.W[self.i + 1, 0])
            qy = float(self.W[self.i + 1, 1])

            dx = qx - self.pos_x
            dy = qy - self.pos_y
            adx = abs(dx)
            ady = abs(dy)

            # If BOTH axes are already within one step, consider waypoint reached.
            # Immediately advance to the next segment and try again (no idle step).
            if adx < v and ady < v:
                self.i += 1
                # If that was the last segment, mark done and return (no move this step)
                if self.i >= self.W.shape[0] - 1:
                    self.done = True
                    return self.pos_x, self.pos_y, 0, 0, False
                continue # retarget in the same call
            # Otherwise, perform axis-wise move by exactly ±v per axis that still needs it
            if adx >= v:
                movement_x = 1 if dx > 0.0 else -1
                newx = self.pos_x + movement_x * v
            else:
                movement_x = 0
                newx = self.pos_x

            if ady >= v:
                movement_y = 1 if dy > 0.0 else -1
                newy = self.pos_y + movement_y * v
            else:
                movement_y = 0
                newy = self.pos_y

            self.pos_x = float(newx)
            self.pos_y = float(newy)

            remx = qx - self.pos_x
            remy = qy - self.pos_y
            flag = (abs(remx) >= v) or (abs(remy) >= v)

            if not flag:
                # We have reached/overshot this waypoint on both axes.
                self.i += 1
                if self.i >= self.W.shape[0] - 1:
                    self.done = True
                return self.pos_x, self.pos_y, movement_x, movement_y, False
            
            # Still not at waypoint; stay on this segment
            return self.pos_x, self.pos_y, movement_x, movement_y, True
