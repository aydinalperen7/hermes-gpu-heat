from numba import cuda, njit

@cuda.jit(device=True, inline=True)
def G2L_3D(idxx, nx, ny, nz):
 # Fortran-style / column-major linearization: idxx = i + j*nx + k*nx*ny
    k = idxx // (nx * ny)
    r = idxx - k * nx * ny
    j = r // nx
    i = r - j * nx
    return i, j, k

@njit
def G2L_3D_CPU(idxx, nx, ny, nz):
    k = idxx // (nx * ny)
    r = idxx - k * nx * ny
    j = r // nx
    i = r - j * nx
    return i, j, k

@cuda.jit(device=True, inline=True)
def L2G_3D(i, j, k, nx, ny):
    return i + j * nx + k * nx * ny

@cuda.jit(device=True, inline=True)
def L2G_3D_CPU(i, j, k, nx, ny):
    return i + j * nx + k * nx * ny


