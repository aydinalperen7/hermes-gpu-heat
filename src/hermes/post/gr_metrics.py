
from __future__ import annotations
import math
import numpy as np
from numba import cuda





# CPU
def compute_G_and_R_cpu(u, unew, dt, hx, hy, hz, eps=1e-12):
    """
    u, unew: (nx, ny, nz) CPU arrays (NumPy)
    returns: G_t, R_t with shape (nx-2, ny-2, nz-2)
    """
    dTdt = np.abs((unew - u) / dt)

    gradTx = ((unew[2:, 1:-1, 1:-1] - unew[:-2, 1:-1, 1:-1]) / (2 * hx) +
              (u[2:,    1:-1, 1:-1] - u[:-2, 1:-1, 1:-1])    / (2 * hx)) * 0.5

    gradTy = ((unew[1:-1, 2:, 1:-1] - unew[1:-1, :-2, 1:-1]) / (2 * hy) +
              (u[1:-1,   2:, 1:-1] - u[1:-1, :-2, 1:-1])     / (2 * hy)) * 0.5

    gradTz = ((unew[1:-1, 1:-1, 2:] - unew[1:-1, 1:-1, :-2]) / (2 * hz) +
              (u[1:-1,   1:-1, 2:] - u[1:-1, 1:-1, :-2])     / (2 * hz)) * 0.5

    G_t = np.sqrt(gradTx**2 + gradTy**2 + gradTz**2)
    R_t = dTdt[1:-1, 1:-1, 1:-1] / (G_t + eps)
    return G_t, R_t

@cuda.jit
def compute_G_and_R_gpu(u_flat, unew_flat,
                         nx, ny, nz,         
                         dt, hx, hy, hz, eps,  
                         G_flat, R_flat):
    """
    Outputs G_flat, R_flat are (nx-2)*(ny-2)*(nz-2), with i-fastest (Fortran-like) interior order.
    """

    # ---- force ints for all index math ----
    nxi = int(nx - 2)
    nyi = int(ny - 2)
    nzi = int(nz - 2)
    px   = np.int32(1)
    py   = np.int32(nx)
    pz   = np.int32(nx * ny)

    if nxi <= 0 or nyi <= 0 or nzi <= 0:
        return

    tid = cuda.grid(1)
    Nint = nxi * nyi * nzi
    if tid >= Nint:
        return

    # interior-flat -> (ii,jj,kk) with i-fastest
    ii = tid % nxi
    tmp = tid // nxi
    jj = tmp % nyi
    kk = tmp // nyi

    # shift to full-domain indices
    i = ii + 1
    j = jj + 1
    k = kk + 1

    # linear indices with *explicit* pitches
    c  = i*px + j*py + k*pz
    ip = c + px
    im = c - px
    jp = c + py
    jm = c - py
    kp = c + pz
    km = c - pz

    # central differences (averaged)
    dux_new = (unew_flat[ip] - unew_flat[im]) * (0.5 / hx)
    dux_old = (   u_flat[ip] -    u_flat[im]) * (0.5 / hx)
    gradTx  = 0.5 * (dux_new + dux_old)

    duy_new = (unew_flat[jp] - unew_flat[jm]) * (0.5 / hy)
    duy_old = (   u_flat[jp] -    u_flat[jm]) * (0.5 / hy)
    gradTy  = 0.5 * (duy_new + duy_old)

    duz_new = (unew_flat[kp] - unew_flat[km]) * (0.5 / hz)
    duz_old = (   u_flat[kp] -    u_flat[km]) * (0.5 / hz)
    gradTz  = 0.5 * (duz_new + duz_old)

    Gval = math.sqrt(gradTx*gradTx + gradTy*gradTy + gradTz*gradTz)
    dTdt = abs((unew_flat[c] - u_flat[c]) / dt)
    Rval = dTdt / (Gval + eps)

    G_flat[tid] = Gval
    R_flat[tid] = Rval
