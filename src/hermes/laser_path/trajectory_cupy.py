#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import cupy as cp




class TrajectoryStepper:
    """
    Vector stepping with constant speed magnitude.
      - Move along the segment direction by `velocity_nd` every step.
      - Last step on a segment can be shorter to land exactly on the waypoint.
    Returns: (x, y, vel_x, vel_y, movement_x, movement_y, Flag)
      vel_x, vel_y are signed per-step components in ND units.
      movement_x/movement_y are in {-1,0,1}.
      Flag=True  means same segment continues next call.
      Flag=False means current waypoint reached this call.
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
        v = float(velocity_nd)
        eps = 1e-14

        if self.done:
            x, y = self.current()
            return x, y, 0.0, 0.0, 0, 0, False

        if self.i >= self.W.shape[0] - 1:
            self.done = True
            x, y = self.current()
            return x, y, 0.0, 0.0, 0, 0, False

        while True:
            if self.i >= self.W.shape[0] - 1:
                self.done = True
                x, y = self.current()
                return x, y, 0.0, 0.0, 0, 0, False

            p = self.pos
            q = self.W[self.i + 1]
            d = q - p

            dx = float(d[0])
            dy = float(d[1])
            dist = (dx * dx + dy * dy) ** 0.5
            # Degenerate/zero-length segment: advance without consuming a time step.
            if dist < eps:
                self.pos[0] = q[0]
                self.pos[1] = q[1]
                self.i += 1
                if self.i >= self.W.shape[0] - 1:
                    self.done = True
                    x, y = self.current()
                    return x, y, 0.0, 0.0, 0, 0, False
                continue

            # Generic vector stepping:
            # - in-segment: move by v along segment direction
            # - endpoint step: snap exactly to waypoint (shorter than v)
            px = float(p[0])
            py = float(p[1])
            reached = dist <= v + eps
            if reached:
                newx = float(q[0])
                newy = float(q[1])
                vel_x = float(newx - px)
                vel_y = float(newy - py)
            else:
                # Keep axis-aligned segments numerically exact step-to-step.
                if abs(dy) <= eps:
                    vel_x = v if dx > 0.0 else -v
                    vel_y = 0.0
                elif abs(dx) <= eps:
                    vel_x = 0.0
                    vel_y = v if dy > 0.0 else -v
                else:
                    ux = dx / dist
                    uy = dy / dist
                    vel_x = v * ux
                    vel_y = v * uy
                newx = px + vel_x
                newy = py + vel_y

            movement_x = 0 if abs(vel_x) < 1e-14 else (1 if vel_x > 0 else -1)
            movement_y = 0 if abs(vel_y) < 1e-14 else (1 if vel_y > 0 else -1)

            self.pos[0] = newx
            self.pos[1] = newy

            if reached:
                self.i += 1
                if self.i >= self.W.shape[0] - 1:
                    self.done = True
                return float(newx), float(newy), vel_x, vel_y, movement_x, movement_y, False

            return float(newx), float(newy), vel_x, vel_y, movement_x, movement_y, True
