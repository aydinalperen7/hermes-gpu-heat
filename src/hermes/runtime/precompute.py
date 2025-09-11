import cupy as cp 

def precompute_for_update(u, nx, ny, nz, y, y_index, x, x_index, slice_1d_in, slice_1d_out, z, float_type):


    u_in  = cp.zeros_like(u[slice_1d_in], dtype=float_type)
    u_out = cp.zeros_like(u[slice_1d_out], dtype=float_type)

    ny_in  = int(y[0:y_index].shape[0])
    ny_out = int(y[y_index:].shape[0])

    nx_in  = int(x[0:x_index].shape[0])
    nx_out = int(x[x_index:].shape[0])

    xoldmin = float_type(x[0].get())
    yoldmin = float_type(y[0].get())
    zoldmin = float_type(z[0].get())

    threads_per_block_in = 128
    blocks_per_grid_in   = (nx * ny_in  * nz + (threads_per_block_in   - 1)) // threads_per_block_in

    threads_per_block_out = 128
    blocks_per_grid_out   = (nx * ny_out * nz + (threads_per_block_out - 1)) // threads_per_block_out

    threads_per_block_in_x = 128
    blocks_per_grid_in_x   = (nx_in  * ny * nz + (threads_per_block_in_x  - 1)) // threads_per_block_in_x

    threads_per_block_out_x = 128
    blocks_per_grid_out_x   = (nx_out * ny * nz + (threads_per_block_out_x - 1)) // threads_per_block_out_x

    return (u_in, u_out, ny_in, ny_out, nx_in, nx_out,
            xoldmin, yoldmin, zoldmin,
            blocks_per_grid_in, blocks_per_grid_out,
            blocks_per_grid_in_x, blocks_per_grid_out_x,
            threads_per_block_in, threads_per_block_out,
            threads_per_block_in_x, threads_per_block_out_x)
