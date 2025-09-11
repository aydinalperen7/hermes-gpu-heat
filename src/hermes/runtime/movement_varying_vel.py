from dataclasses import dataclass

@dataclass
class LevelRefs:
    # arrays and dims for one level
    u: any
    x: any; y: any; z: any
    nx: int; ny: int; nz: int
    x0: any; y0: any

@dataclass
class Kernels:
    grid_movement_index: any
    precompute_for_update: any

class PrecomputeState:
    """Holds the outputs that were globals before."""
    # L3 (lin)
    index_y_lin=None; index_x_lin=None; index_y_lin_neg=None; index_x_lin_neg=None
    slice_1d_in_liny=None; slice_1d_out_liny=None; slice_1d_in_linx=None; slice_1d_out_linx=None
    slice_1d_in_linx_negative=None; slice_1d_out_linx_negative=None
    slice_1d_in_liny_negative=None; slice_1d_out_liny_negative=None
    uin_lin=None; uout_lin=None; ny_lin_in=None; ny_lin_out=None; nx_lin_in=None; nx_lin_out=None
    xoldmin_lin=None; yoldmin_lin=None; zoldmin_lin=None
    blocks_per_grid_in_lin_y=None; blocks_per_grid_out_lin_y=None
    blocks_per_grid_in_lin_x=None; blocks_per_grid_out_lin_x=None
    threads_per_block_in_lin_y=None; threads_per_block_out_lin_y=None
    threads_per_block_in_lin_x=None; threads_per_block_out_lin_x=None

    # L2
    index_y_level2=None; index_x_level2=None; index_y_neg_level2=None; index_x_neg_level2=None
    slice_1d_in_level2y=None; slice_1d_out_level2y=None; slice_1d_in_level2x=None; slice_1d_out_level2x=None
    slice_1d_in_level2x_negative=None; slice_1d_out_level2x_negative=None
    slice_1d_in_level2y_negative=None; slice_1d_out_level2y_negative=None
    uin_s_level2=None; uout_s_level2=None; ny_s_in_level2=None; ny_s_out_level2=None
    nx_s_in_level2=None; nx_s_out_level2=None
    xoldmin_s_level2=None; yoldmin_s_level2=None; zoldmin_s_level2=None
    blocks_per_grid_in_s_level2_y=None; blocks_per_grid_out_s_level2_y=None
    blocks_per_grid_in_s_level2_x=None; blocks_per_grid_out_s_level2_x=None
    threads_per_block_in_s_level2_y=None; threads_per_block_out_s_level2_y=None
    threads_per_block_in_s_level2_x=None; threads_per_block_out_s_level2_x=None

    # L1
    index_y=None; index_x=None; index_y_neg=None; index_x_neg=None
    slice_1d_iny=None; slice_1d_outy=None; slice_1d_inx=None; slice_1d_outx=None
    slice_1d_inx_negative=None; slice_1d_outx_negative=None
    slice_1d_iny_negative=None; slice_1d_outy_negative=None
    uin_s=None; uout_s=None; ny_s_in=None; ny_s_out=None; nx_s_in=None; nx_s_out=None
    xoldmin_s=None; yoldmin_s=None; zoldmin_s=None
    blocks_per_grid_in_s_y=None; blocks_per_grid_out_s_y=None
    blocks_per_grid_in_s_x=None; blocks_per_grid_out_s_x=None
    threads_per_block_in_s_y=None; threads_per_block_out_s_y=None
    threads_per_block_in_s_x=None; threads_per_block_out_s_x=None

    # mins
    xminn=None; yminn=None; zminn=None
    xmin_level2=None; ymin_level2=None; zmin_level2=None

class GridUpdater:
    """Replaces update_grid_and_precompute(velocity) without globals."""
    def __init__(self, level1: LevelRefs, level2: LevelRefs, lin: LevelRefs,
                 kernels: Kernels, float_type, t_len: int):
        self.l1, self.l2, self.lin = level1, level2, lin
        self.k = kernels
        self.float_type = float_type
        self.t_len = t_len
        self.state = PrecomputeState()

    def update(self, velocity):
        # ---- indices/slices (exactly your calls) ----
        (self.state.index_y_lin, self.state.index_x_lin, self.state.index_y_lin_neg, self.state.index_x_lin_neg,
         self.state.slice_1d_in_liny, self.state.slice_1d_out_liny,
         self.state.slice_1d_in_linx, self.state.slice_1d_out_linx,
         self.state.slice_1d_in_linx_negative, self.state.slice_1d_out_linx_negative,
         self.state.slice_1d_in_liny_negative, self.state.slice_1d_out_liny_negative
        ) = self.k.grid_movement_index(self.lin.x0, self.lin.y0, velocity, self.lin.nx, self.lin.ny, self.lin.nz, t_len=self.t_len)

        (self.state.index_y_level2, self.state.index_x_level2, self.state.index_y_neg_level2, self.state.index_x_neg_level2,
         self.state.slice_1d_in_level2y, self.state.slice_1d_out_level2y,
         self.state.slice_1d_in_level2x, self.state.slice_1d_out_level2x,
         self.state.slice_1d_in_level2x_negative, self.state.slice_1d_out_level2x_negative,
         self.state.slice_1d_in_level2y_negative, self.state.slice_1d_out_level2y_negative
        ) = self.k.grid_movement_index(self.l2.x0, self.l2.y0, velocity, self.l2.nx, self.l2.ny, self.l2.nz, t_len=self.t_len)

        (self.state.index_y, self.state.index_x, self.state.index_y_neg, self.state.index_x_neg,
         self.state.slice_1d_iny, self.state.slice_1d_outy,
         self.state.slice_1d_inx, self.state.slice_1d_outx,
         self.state.slice_1d_inx_negative, self.state.slice_1d_outx_negative,
         self.state.slice_1d_iny_negative, self.state.slice_1d_outy_negative
        ) = self.k.grid_movement_index(self.l1.x0, self.l1.y0, velocity, self.l1.nx, self.l1.ny, self.l1.nz, t_len=self.t_len)

        # ---- mins ----
        self.state.xminn = self.float_type(self.lin.x[0].get())
        self.state.yminn = self.float_type(self.lin.y[0].get())
        self.state.zminn = self.float_type(self.lin.z[0].get())
        self.state.xmin_level2 = self.float_type(self.l2.x[0].get())
        self.state.ymin_level2 = self.float_type(self.l2.y[0].get())
        self.state.zmin_level2 = self.float_type(self.l2.z[0].get())

        # ---- precompute_for_update: L1 (dict exact keys) ----
        pre_s = self.k.precompute_for_update(self.l1.u, self.l1.nx, self.l1.ny, self.l1.nz,
                                             self.l1.y, self.state.index_y,
                                             self.l1.x, self.state.index_x,
                                             self.state.slice_1d_iny, self.state.slice_1d_outy,
                                             self.l1.z, float_type=self.float_type)
        self.state.uin_s  = pre_s["u_in"];      self.state.uout_s = pre_s["u_out"]
        self.state.ny_s_in = pre_s["ny_in"];    self.state.ny_s_out = pre_s["ny_out"]
        self.state.nx_s_in = pre_s["nx_in"];    self.state.nx_s_out = pre_s["nx_out"]
        self.state.xoldmin_s = pre_s["xoldmin"]; self.state.yoldmin_s = pre_s["yoldmin"]; self.state.zoldmin_s = pre_s["zoldmin"]
        self.state.blocks_per_grid_in_s_y  = pre_s["blocks_in_y"];  self.state.blocks_per_grid_out_s_y = pre_s["blocks_out_y"]
        self.state.blocks_per_grid_in_s_x  = pre_s["blocks_in_x"];  self.state.blocks_per_grid_out_s_x = pre_s["blocks_out_x"]
        self.state.threads_per_block_in_s_y = pre_s["tpbin"];       self.state.threads_per_block_out_s_y = pre_s["tpbout"]
        self.state.threads_per_block_in_s_x = pre_s["tpbin_x"];     self.state.threads_per_block_out_s_x = pre_s["tpbout_x"]

        # ---- L2 ----
        pre2 = self.k.precompute_for_update(self.l2.u, self.l2.nx, self.l2.ny, self.l2.nz,
                                            self.l2.y, self.state.index_y_level2,
                                            self.l2.x, self.state.index_x_level2,
                                            self.state.slice_1d_in_level2y, self.state.slice_1d_out_level2y,
                                            self.l2.z, float_type=self.float_type)
        self.state.uin_s_level2  = pre2["u_in"];      self.state.uout_s_level2 = pre2["u_out"]
        self.state.ny_s_in_level2 = pre2["ny_in"];    self.state.ny_s_out_level2 = pre2["ny_out"]
        self.state.nx_s_in_level2 = pre2["nx_in"];    self.state.nx_s_out_level2 = pre2["nx_out"]
        self.state.xoldmin_s_level2 = pre2["xoldmin"]; self.state.yoldmin_s_level2 = pre2["yoldmin"]; self.state.zoldmin_s_level2 = pre2["zoldmin"]
        self.state.blocks_per_grid_in_s_level2_y  = pre2["blocks_in_y"];  self.state.blocks_per_grid_out_s_level2_y = pre2["blocks_out_y"]
        self.state.blocks_per_grid_in_s_level2_x  = pre2["blocks_in_x"];  self.state.blocks_per_grid_out_s_level2_x = pre2["blocks_out_x"]
        self.state.threads_per_block_in_s_level2_y = pre2["tpbin"];       self.state.threads_per_block_out_s_level2_y = pre2["tpbout"]
        self.state.threads_per_block_in_s_level2_x = pre2["tpbin_x"];     self.state.threads_per_block_out_s_level2_x = pre2["tpbout_x"]

        # ---- LIN ----
        plin = self.k.precompute_for_update(self.lin.u, self.lin.nx, self.lin.ny, self.lin.nz,
                                            self.lin.y, self.state.index_y_lin,
                                            self.lin.x, self.state.index_x_lin,
                                            self.state.slice_1d_in_liny, self.state.slice_1d_out_liny,
                                            self.lin.z, float_type=self.float_type)
        self.state.uin_lin = plin["u_in"];        self.state.uout_lin = plin["u_out"]
        self.state.ny_lin_in = plin["ny_in"];     self.state.ny_lin_out = plin["ny_out"]
        self.state.nx_lin_in = plin["nx_in"];     self.state.nx_lin_out = plin["nx_out"]
        self.state.xoldmin_lin = plin["xoldmin"]; self.state.yoldmin_lin = plin["yoldmin"]; self.state.zoldmin_lin = plin["zoldmin"]
        self.state.blocks_per_grid_in_lin_y = plin["blocks_in_y"];   self.state.blocks_per_grid_out_lin_y = plin["blocks_out_y"]
        self.state.blocks_per_grid_in_lin_x = plin["blocks_in_x"];   self.state.blocks_per_grid_out_lin_x = plin["blocks_out_x"]
        self.state.threads_per_block_in_lin_y = plin["tpbin"];       self.state.threads_per_block_out_lin_y = plin["tpbout"]
        self.state.threads_per_block_in_lin_x = plin["tpbin_x"];     self.state.threads_per_block_out_lin_x = plin["tpbout_x"]

        return self.state

    #  keep old variable names temporarily
    def as_globals(self):
        s = self.state
        return {name: getattr(s, name) for name in vars(s) if not name.startswith('_')}
