#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import cupy as cp

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
    def __init__(self, waypoints_nd: cp.ndarray):
        assert waypoints_nd.ndim == 2 and waypoints_nd.shape[1] == 2
        self.W = cp.asarray(waypoints_nd, dtype=float, order="C")
        self.i = 0
        self.pos = self.W[0].copy()
        self.done = (self.W.shape[0] < 2)

    def current(self):
        return float(self.pos[0]), float(self.pos[1])

    def advance(self, velocity_nd: float):
        if self.done:
            x, y = self.current()
            return x, y, 0, 0, False
    
        # If last segment is already finished, stay put
        if self.i >= self.W.shape[0] - 1:
            self.done = True
            x, y = self.current()
            return x, y, 0, 0, False
    
        # We may need to retarget immediately if we’re exactly at a waypoint.
        # Loop: if we’re at a waypoint, advance segment and re-evaluate, otherwise move once and return.
        while True:
            # Safety: if we ran out of segments by retargeting, stop
            if self.i >= self.W.shape[0] - 1:
                self.done = True
                x, y = self.current()
                return x, y, 0, 0, False
    
            p = self.pos
            q = self.W[self.i + 1]
            d = q - p
    
            dx = float(d[0]); adx = abs(dx)
            dy = float(d[1]); ady = abs(dy)
    
            # If BOTH axes are already within one step, consider waypoint reached.
            # Immediately advance to the next segment and try again (no idle step).
            if adx < velocity_nd and ady < velocity_nd:
                self.i += 1
                # If that was the last segment, mark done and return (no move this step)
                if self.i >= self.W.shape[0] - 1:
                    self.done = True
                    x, y = self.current()
                    return x, y, 0, 0, False
                continue  # retarget in the same call
    
            # Otherwise, perform axis-wise move by exactly ±velocity_nd per axis that still needs it
            if adx >= velocity_nd:
                movement_x = 1 if dx > 0 else -1
                newx = p[0] + movement_x * velocity_nd
            else:
                movement_x = 0
                newx = p[0]
    
            if ady >= velocity_nd:
                movement_y = 1 if dy > 0 else -1
                newy = p[1] + movement_y * velocity_nd
            else:
                movement_y = 0
                newy = p[1]
    
            # Update position with this step’s movement
            self.pos[0] = newx
            self.pos[1] = newy    
            # Remaining distance to this waypoint after moving
            rem = q - self.pos
            Flag = (abs(rem[0]) >= velocity_nd) or (abs(rem[1]) >= velocity_nd)
    
            if not Flag:
                # We have reached/overshot this waypoint on both axes.
                self.i += 1
                if self.i >= self.W.shape[0] - 1:

                    self.done = True
                return float(newx), float(newy), movement_x, movement_y, False
    
            # Still not at waypoint; stay on this segment
            return float(newx), float(newy), movement_x, movement_y, True


