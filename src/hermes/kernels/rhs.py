from numba import cuda
from math import sin, pi
import numpy as np

@cuda.jit
def rhs_level3_dirichlet(
    nx, ny, nz,
    u,                 # in: current field (1D, column major)
    qs,                # in: surface flux 
    b,                 # out: RHS (1D)
    hixsq, hiysq, hizsq,
    n2, n3,            # material/BC coefficients 
    dt05,
    u0,                # Dirichlet value at boundaries (k==0 or lateral edges)
    hz
):
    
    """
    Level 3 (Dirichlet) RHS assembly.
    Inputs: u[nx*ny*nz], qs[i,j] 
    Output: b[nx*ny*nz]
    Boundary behavior: i==0|nx-1 or j==0|ny-1 or k==0 → Dirichlet value u0.
    Top (k==nz-1): Robin-like closure using qs.
    """
    
    idxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idxx >= nx * ny * nz:
        return

    i = idxx % nx
    j = (idxx // nx) % ny
    k = idxx // (nx * ny)
    

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0:
        b[idxx] = u0
        return

    O = idxx + 1
    I = idxx - 1
    R = idxx + nx
    L = idxx - nx
    T = idxx + nx * ny
    B = idxx - nx * ny

    uC = u[idxx]
    uI = u[I]; uO = u[O]; uR = u[R]; uL = u[L]; uB = u[B]
    # top closure
    four = 4.0; two = 2.0
    uT = u[T] if k < nz - 1 else (uB + (four * hz * (qs[i, j] - n2 * n3)) - (two * hz * n2 * uC))

    dx = (uI + uO) * hixsq * dt05
    dy = (uR + uL) * hiysq * dt05
    dz = (uT + uB) * hizsq * dt05
    p1 = -(2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC
    b[idxx] = dx + dy + dz + p1


@cuda.jit
def rhs_level12_neumann(
    nx, ny, nz,
    u,                   # in: current field
    qs,                  # in: surface flux (ny, nx) as qs[i,j]
    b,                   # out
    hixsq, hiysq, hizsq,
    iSte, n2, n3, dt05,
    p_o, p_i, p_r, p_l, p_b,         # Neumann diffs (current)
    p_o_old, p_i_old, p_r_old, p_l_old, p_b_old,   # Neumann diffs (previous)
    u_non_lin,           # for latent term
    hz,
    n4, n5, n6           # radiation terms for top
):
    """
    Levels 1–2 (Neumann) RHS assembly.

    """
    idxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idxx >= nx * ny * nz:
        return

    i = idxx % nx
    j = (idxx // nx) % ny
    k = idxx // (nx * ny)

    O = idxx + 1
    I = idxx - 1
    R = idxx + nx
    L = idxx - nx
    T = idxx + nx * ny
    B = idxx - nx * ny

    uC = u[idxx]

    if (0 < i < nx - 1) and (0 < j < ny - 1) and (0 < k < nz - 1):
        # interior nodes
        uI = u[I]; uO = u[O]; uR = u[R]; uL = u[L]; uT = u[T]; uB = u[B]
    else:
        # top
        if k == nz - 1:
            qs2 = qs[i, j]
            four = 4.0; two = 2.0
            uB = u[B]
            # radiative term via n4*((u+n5)^4 - n6^4)
            uT = uB +  (four*hz * (qs2 - n2*n3)) - (two*hz*n2*uC) - 4*hz*n4*((u[idxx] + n5)**4 - n6**4  )
            
        elif k == 0:
            uT = u[T]
            bc_idx = j * nx + i
            uB = uT - p_b[bc_idx] - p_b_old[bc_idx]
            
        else:
            uT = u[T]; uB = u[B]

        # j bounds
        if j == ny - 1:
            uL = u[L]
            bc_idx = k * nx + i
            uR = uL + p_r[bc_idx] + p_r_old[bc_idx]
        elif j == 0:
            uR = u[R]
            bc_idx = k * nx + i
            uL = uR - p_l[bc_idx] - p_l_old[bc_idx]
        else:
            uR = u[R]; uL = u[L]

        # i bounds
        if i == nx - 1:
            uI = u[I]
            bc_idx = k * ny + j
            uO = uI + p_o[bc_idx] + p_o_old[bc_idx]
        elif i == 0:
            uO = u[O]
            bc_idx = k * ny + j
            uI = uO - p_i[bc_idx] - p_i_old[bc_idx]
        else:
            uO = u[O]; uI = u[I]

    dx = (uO + uI) * hixsq * dt05
    dy = (uR + uL) * hiysq * dt05
    dz = (uT + uB) * hizsq * dt05

    # latent heat term
    if (u_non_lin[idxx] > 0.0) and (u_non_lin[idxx] < 1.0):
        p1 = -(2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC + (0.5 * pi * (iSte) * np.sin(pi * u_non_lin[idxx]) * uC)
    else:
        p1 = -(2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC

    b[idxx] = dx + dy + dz + p1
