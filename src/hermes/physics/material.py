from math import pi
import numpy as np

__all__ = ["phys_parameter"]

class phys_parameter:
    """
    Legacy-compatible container with nondimensional groups.
    Args:
        arg1: Q (W)
        arg2: rb (m)  -> beam radius
        arg3: t_spot_on (s)
        mat_ch:  (optional) dict to override material 
    """

    def __init__(self, arg1, arg2, arg3, mat_ch=None):
        Q = float(arg1)
        rb = float(arg2)
        t_spot_on = float(arg3)

        # --- material and environment defaults (same as your file) ---
        K = 13.5                    # W/(m K)
        rho = 7950                  # kg/m^3
        Cp = 470                    # J/(kg K)
        Lf = 2.7e5                  # J/kg
        A = 0.55                    # absorptivity

        T0 = 25.0 + 273.0           # ambient K
        Ts = 1658.0                 # solidus K
        Tl = 1723.0                 # liquidus K

        hc = 10.0                   # W m^{-2} K^{-1}
        epsilon = 0.1
        sigma = 5.67e-8             # W m^{-2} K^{-4}
        # -------------------------------------------------------------

        # allow overrides (minimal change)
        if mat_ch:
            K       = mat_ch.get("K", K)
            rho     = mat_ch.get("rho", rho)
            Cp      = mat_ch.get("Cp", Cp)
            Lf      = mat_ch.get("Lf", Lf)
            A       = mat_ch.get("A", A)
            T0      = mat_ch.get("T0", T0)
            Ts      = mat_ch.get("Ts", Ts)
            Tl      = mat_ch.get("Tl", Tl)
            hc      = mat_ch.get("hc", hc)
            epsilon = mat_ch.get("epsilon", epsilon)
            sigma   = mat_ch.get("sigma", sigma)

        self.Q = Q
        self.rb = rb
        self.t_spot_on = t_spot_on

        self.Ts = Ts
        self.Tl = Tl
        self.deltaT = self.Tl - self.Ts
        self.T0 = T0

        self.Cp = Cp
        self.kappa = K / (rho * Cp)
        self.Ste = self.Cp * self.deltaT / Lf  # Stefan number

        # scales 
        self.len_scale  = self.Q * self.t_spot_on * self.kappa / (self.rb**2 * K * self.deltaT)
        self.time_scale = self.len_scale**2 / self.kappa

        # nondimensional groups
        self.n1 = 2 * self.Q * A * self.len_scale / (pi * self.rb**2 * K * self.deltaT)
        self.n2 = hc * self.len_scale / K
        self.n3 = (self.Ts - self.T0) / self.deltaT
        self.n4 = epsilon * sigma * self.len_scale * (self.deltaT**3) / K
        self.n5 = self.Ts / self.deltaT
        self.n6 = self.T0 / self.deltaT

        self.u0 = (self.T0 - self.Ts) / self.deltaT
