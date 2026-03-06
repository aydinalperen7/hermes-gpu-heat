def launch_1d(n_elems, threads=128):
    """Return (blocks, threads) for a 1D CUDA launch."""
    blocks = (n_elems + (threads - 1)) // threads
    return blocks, threads


def launch_flattened(nx, ny, nz, threads=128):
    """Return (blocks, threads) for flattened 3D kernels."""
    return launch_1d(nx * ny * nz, threads)


def launch_3d(nx, ny, nz, threads=128):
    """Backward-compatible alias for launch_flattened."""
    return launch_flattened(nx, ny, nz, threads)


def launch_3d_xyz(nx, ny, nz, threads=(8, 4, 4)):
    """Return ((bx, by, bz), (tx, ty, tz)) for native 3D CUDA launches."""
    tx, ty, tz = threads
    blocks = (
        (int(nx) + tx - 1) // tx,
        (int(ny) + ty - 1) // ty,
        (int(nz) + tz - 1) // tz,
    )
    return blocks, threads


def launch_matvec_3d(nx, ny, nz):
    """Convenience launcher for 3D matvec kernels with tuned default threads."""
    return launch_3d_xyz(nx, ny, nz, threads=(8, 4, 4))


def launch_bc(n_face, threads=128):
    """Return (blocks, threads) for face-sized launches (e.g. nx*nz, nx*ny)."""
    return launch_1d(n_face, threads)


def launch_2d(nx, ny, threads=(16, 16)):
    """Return ((bx, by), (tx, ty)) for native 2D CUDA launches."""
    tx, ty = threads
    blocks = (
        (int(nx) + tx - 1) // tx,
        (int(ny) + ty - 1) // ty,
    )
    return blocks, threads


def launch_bc_faces_3d(nx_s_level2, ny_s_level2, nz_s_level2, nx_s, ny_s, nz_s, threads=(16, 16)):
    """
    Return launch tuples for BC face kernels using 2D indexing:
      1) level2 x-z face, 2) level2 x-y face,
      3) level1 x-z face, 4) level1 x-y face
    """
    bc1 = launch_2d(nx_s_level2, nz_s_level2, threads=threads)
    bc2 = launch_2d(nx_s_level2, ny_s_level2, threads=threads)
    bc3 = launch_2d(nx_s, nz_s, threads=threads)
    bc4 = launch_2d(nx_s, ny_s, threads=threads)
    return bc1, bc2, bc3, bc4


def launch_interp_3d(nx, ny, nz, threads=(8, 4, 4)):
    """Return ((bx, by, bz), (tx, ty, tz)) for 3D interpolation kernels."""
    return launch_3d_xyz(nx, ny, nz, threads=threads)


def launch_interp_groups_3d(nx_s2_level2, ny_s2_level2, nz_s2_level2,
                            nx_s2, ny_s2, nz_s2,
                            nx_s, ny_s, nz_s,
                            nx_s_level2, ny_s_level2, nz_s_level2,
                            threads=(8, 4, 4)):
    """
    Return launch tuples for four interpolation buffers:
      int1: s2_level2, int2: s2, int3: s, int4: s_level2
    """
    int1 = launch_interp_3d(nx_s2_level2, ny_s2_level2, nz_s2_level2, threads=threads)
    int2 = launch_interp_3d(nx_s2, ny_s2, nz_s2, threads=threads)
    int3 = launch_interp_3d(nx_s, ny_s, nz_s, threads=threads)
    int4 = launch_interp_3d(nx_s_level2, ny_s_level2, nz_s_level2, threads=threads)
    return int1, int2, int3, int4
