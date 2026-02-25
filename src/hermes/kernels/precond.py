from numba import cuda
from math import sin, pi


@cuda.jit
def build_diag_level3_dirichlet(
    nx, ny, nz,
    hixsq, hiysq, hizsq,
    dt05, n2, hz,
    diag
):
    idxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idxx >= nx * ny * nz:
        return

    i = idxx % nx
    j = (idxx // nx) % ny
    k = idxx // (nx * ny)

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0:
        diag[idxx] = 1.0
        return

    d = 1.0 + (2.0 * (hixsq + hiysq + hizsq)) * dt05
    if k == nz - 1:
        d += (2.0 * hz * n2 * hizsq) * dt05
    diag[idxx] = d


@cuda.jit
def build_diag_level12_neumann(
    nx, ny, nz,
    hixsq, hiysq, hizsq,
    dt05, n2, iSte, hz,
    u_phase,
    diag
):
    idxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idxx >= nx * ny * nz:
        return

    k = idxx // (nx * ny)
    phase = u_phase[idxx]

    d = 1.0 + (2.0 * (hixsq + hiysq + hizsq)) * dt05
    if (phase > 0.0) and (phase < 1.0):
        d += 0.5 * pi * iSte * sin(pi * phase)
    if k == nz - 1:
        d += (2.0 * hz * n2 * hizsq) * dt05

    diag[idxx] = d
