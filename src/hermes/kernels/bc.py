from numba import cuda

@cuda.jit
def extract_neumann_bc_r_l_i_o(
    nx, ny, nz,
    vals,                       # flattened field 
    right_bc, left_bc, out_bc, in_bc,  # outputs
    slice_bc_out0_1d, slice_bc_in0_1d,
    slice_bc_right0_1d, slice_bc_left0_1d,
    nx2                          # stride for the neighbor distance in x
):
    """
    Compute finite-difference gradients on right/left/in/out faces:
    bc = vals[index2] - vals[index1] with precomputed face indices.
    Expects `nx*nz` sized faces (your original sizes).
    """
    
    indxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if indxx >= nx * nz:
        return

    index1_out = slice_bc_out0_1d[indxx]
    index2_out = index1_out + 2
    index1_in  = slice_bc_in0_1d[indxx]
    index2_in  = index1_in + 2

    out_bc[indxx] = vals[index2_out] - vals[index1_out]
    in_bc[indxx]  = vals[index2_in]  - vals[index1_in]

    index1_right = slice_bc_right0_1d[indxx]
    index2_right = index1_right + 2 * nx2
    index1_left  = slice_bc_left0_1d[indxx]
    index2_left  = index1_left + 2 * nx2

    right_bc[indxx] = vals[index2_right] - vals[index1_right]
    left_bc[indxx]  = vals[index2_left]  - vals[index1_left]


@cuda.jit
def extract_neumann_bc_b(
    nx, ny,           # face is nx*ny
    vals,
    bottom_bc,        # out: nx*ny
    slice_bc_bottom0_1d,
    nx2, ny2          # strides on level 2
):
    """
    Compute bottom face gradient using a 2*nx2*ny2 stride jump.
    """
    indxx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if indxx >= nx * ny:
        return
    index1 = slice_bc_bottom0_1d[indxx]
    index2 = index1 + 2 * nx2 * ny2
    bottom_bc[indxx] = vals[index2] - vals[index1]
