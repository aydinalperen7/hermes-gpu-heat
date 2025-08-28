#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:27:01 2025

@author: aaydin
"""




import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.sparse.linalg as spla
from numba import cuda
from hermes.physics.material import phys_parameter
from hermes.grids.sim_params import init_level3_outer
from hermes.grids.sim_params import init_inner_level
from hermes.runtime.movement import (
    grid_movement_index,
    precompute_for_update
)
from hermes.runtime.gpu_setup import  launch_3d, launch_bc

from hermes.kenrels.rhs import rhs_level3_dirichlet as rhs_lin_dirichlet
from hermes.kenrels.rhs import rhs_level12_neumann as rhs_neumann
from hermes.kenrels.matvec import mv_level3_dirichlet, mv_level12_neumann
from hermes.kenrels.bc import extract_neumann_bc_r_l_i_o, extract_neumann_bc_b
from hermes.kenrels.interp import  trilinear_interpolation





def len_dim(x): return x *  phys.len_scale      # [m]
def len_um(x): return x *  phys.len_scale * 1e6    # [um]
def shape_back_3d(u, nx, ny, nz): return cp.reshape(u, (nx, ny, nz), order='F')
def temp_dim(u): return u * phys.deltaT + phys.Ts  # [K]
def gaussian2d(x, y, sigma,x00,y00): return two * phys.n1 * cp.exp(-((x-x00)**two + (y-y00)**two) / (two * sigma**two)) 
    


def mv_wrapper_lin(v):
    mv_level3_dirichlet[blocks_per_grid_lin,threads_per_block_lin](v, d_result_lin, nx_lin, ny_lin, nz_lin, h_linsq, h_linsq, h_linsq, dt_lin05, n2, u0, h_lin, two)
    return result_lin

def mv_wrapper_s(v):
    mv_level12_neumann[blocks_per_grid_s,threads_per_block_s ](v,d_result_s, nx_s, ny_s, nz_s,h_ix_newsq, h_iy_newsq, h_iz_newsq, dt_lin05, n2, iSte, u_s, h_z_new, two, half, one, pii) 
    return result_s

def mv_wrapper_s_level2(v):
    mv_level12_neumann[blocks_per_grid_s_level2,threads_per_block_s_level2](v, d_result_s_level2, nx_s_level2, ny_s_level2, nz_s_level2, h_ix_newsq_level2, h_iy_newsq_level2, h_iz_newsq_level2, dt_lin05, n2, iSte, u_s_level2, h_z_new_level2, two , half, one, pii)
    return result_s_level2


def sparse_cg(A, b, u0, TOL, P, maxit):

      num_iters = 0

      def callback(xk):
         nonlocal num_iters
         num_iters+=1

      x,status = spla.cg(A, b, x0=u0, tol=TOL, M=P, maxiter = maxit, callback=callback)
      return x,status,num_iters
          

        

### FUNCS FOR MOVING ###

def move_gridy_all(velocity, iii):
    global y_s, Y_s, y_s2, y_s_level2, Y_s_level2, y_s2_level2, y_lin, Y_lin
    if iii != 0:
        y_s[:] += velocity
        y_s2[:] +=  velocity
        Y_s[:] += velocity
        
        y_s_level2[:] += velocity
        y_s2_level2[:] += velocity
        Y_s_level2[:] += velocity
        
        y_lin[:] +=  velocity
        Y_lin[:] +=  velocity

def move_gridx_all(velocity, iii):
    global x_s, X_s, x_s2, x_s_level2, X_s_level2, x_s2_level2, x_lin, X_lin
    if iii != 0:
        x_s[:] += velocity
        x_s2[:] += velocity
        X_s[:] += velocity
        
        x_s_level2[:] += velocity
        x_s2_level2[:] += velocity
        X_s_level2[:] += velocity
        
        x_lin[:] += velocity
        X_lin[:] += velocity

def resety_all(y_s0, Y_s0, y_s20, y_s0_level2, Y_s0_level2, y_s20_level2, y_lin0, Y_lin0 ):
    global y_s, Y_s, y_s2, y_s_level2, Y_s_level2, y_s2_level2, y_lin, Y_lin

    y_s[:] = y_s0.copy()
    y_s2[:] =  y_s20.copy()
    Y_s[:] = Y_s0.copy()
    
    y_s_level2[:] = y_s0_level2.copy()
    y_s2_level2[:] = y_s20_level2.copy()
    Y_s_level2[:] = Y_s0_level2.copy()
    
    y_lin[:] =  y_lin0.copy()
    Y_lin[:] =  Y_lin0.copy()
    
def resetx_all(x_s0, X_s0, x_s20, x_s0_level2, X_s0_level2, x_s20_level2, x_lin0, X_lin0):
    global x_s, X_s, x_s2, x_s_level2, X_s_level2, x_s2_level2, x_lin, X_lin

    x_s[:] = x_s0.copy()
    x_s2[:] = x_s20.copy()
    X_s[:] = X_s0.copy()
    
    x_s_level2[:] = x_s0_level2.copy()
    x_s2_level2[:] = x_s20_level2.copy()
    X_s_level2[:] = X_s0_level2.copy()
    
    x_lin[:] = x_lin0.copy()
    X_lin[:] = X_lin0.copy()





def update_after_movementy2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin,  nx, ny_in, nz, uin, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2,  xold2min, yold2min, zold2min,  ny_out, uout, one, val3, slice_1d_in, slice_1d_out, index_y, threads_per_block_in, blocks_per_grid_in, threads_per_block_out , blocks_per_grid_out):
    '''
    x,y,z: New coordinates after movement
    val:  The current value of the grid before update (u_s or u_s_level2)
    nx_old, ny_old, nz_old: Should match the size of val (nx_s, ny_s, nz_s or level2)
    hxval, hyval, hzval: Should match the spacing of vals coordinates
    xoldmin, yoldmin, zoldmin: Extracted from the coordinate of the level (before movement)
    nx, ny_in, nz , ny_out: Grid size of uin nx*nz*ny_in and grid size of uout is nx*nz*ny_out
    
    
    val2: Value of outer level (before update and movement)
    nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2: Grid of val2
    xold2min, yold2min, zold2min: Extracted from the coordinate of the outer level (before movement)
    
    '''
    yin = y[0:index_y]
    yout = y[index_y:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin, nx, ny_in, nz, uin, one)
    trilinear_interpolation[blocks_per_grid_out, threads_per_block_out](x, yout, z, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2, xold2min, yold2min, zold2min, nx, ny_out, nz, uout, one)
    

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = uout

def update_after_movementy2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin,  nx, ny_in, nz, uin, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2,  xold2min, yold2min, zold2min,  ny_out, uout, one, val3, slice_1d_in, slice_1d_out, index_y, threads_per_block_in, blocks_per_grid_in, threads_per_block_out , blocks_per_grid_out):
    '''
    x,y,z: New coordinates after movement
    val:  The current value of the grid before update (u_s or u_s_level2)
    nx_old, ny_old, nz_old: Should match the size of val (nx_s, ny_s, nz_s or level2)
    hxval, hyval, hzval: Should match the spacing of vals coordinates
    xoldmin, yoldmin, zoldmin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz , nx_out: Grid size of uin nx*nz*ny_in and grid size of uout is ny*nz*nx_out
    
    
    val2: Value of outer level (before update and movement)
    nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2: Grid of val2
    xold2min, yold2min, zold2min: Extracted from the coordinate of the outer level (before movement)
    '''
    yin = y[index_y+1:]
    yout = y[0:index_y+1]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin, nx, ny_in, nz, uin, one)
    trilinear_interpolation[blocks_per_grid_out, threads_per_block_out](x, yout, z, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2, xold2min, yold2min, zold2min, nx, ny_out, nz, uout, one)
    

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = uout



def update_after_movementx2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin,  nx_in, ny, nz, uin, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2,  xold2min, yold2min, zold2min,  nx_out, uout, one, val3, slice_1d_in, slice_1d_out, index_x, threads_per_block_in, blocks_per_grid_in, threads_per_block_out , blocks_per_grid_out):
    '''
    x,y,z: New coordinates after movement
    val:  The current value of the grid before update (u_s or u_s_level2)
    nx_old, ny_old, nz_old: Should match the size of val (nx_s, ny_s, nz_s or level2)
    hxval, hyval, hzval: Should match the spacing of vals coordinates
    xoldmin, yoldmin, zoldmin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz , nx_out: Grid size of uin nx*nz*ny_in and grid size of uout is ny*nz*nx_out
    
    
    val2: Value of outer level (before update and movement)
    nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2: Grid of val2
    xold2min, yold2min, zold2min: Extracted from the coordinate of the outer level (before movement)
    '''
    xin = x[0:index_x]
    xout = x[index_x:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval,xoldmin, yoldmin, zoldmin, nx_in, ny, nz, uin, one)    
    trilinear_interpolation[blocks_per_grid_out, threads_per_block_out](xout, y, z, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2, xold2min, yold2min, zold2min, nx_out, ny, nz, uout, one)
    

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = uout
    
def update_after_movementx2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin, yoldmin, zoldmin,  nx_in, ny, nz, uin, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2,  xold2min, yold2min, zold2min,  nx_out, uout, one, val3, slice_1d_in, slice_1d_out, index_x, threads_per_block_in, blocks_per_grid_in, threads_per_block_out , blocks_per_grid_out):
    '''
    x,y,z: New coordinates after movement
    val:  The current value of the grid before update (u_s or u_s_level2)
    nx_old, ny_old, nz_old: Should match the size of val (nx_s, ny_s, nz_s or level2)
    hxval, hyval, hzval: Should match the spacing of vals coordinates
    xoldmin, yoldmin, zoldmin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz , nx_out: Grid size of uin nx*nz*ny_in and grid size of uout is ny*nz*nx_out
    
    
    val2: Value of outer level (before update and movement)
    nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2: Grid of val2
    xold2min, yold2min, zold2min: Extracted from the coordinate of the outer level (before movement)
    '''
    xin = x[index_x+1:]
    xout = x[0:index_x+1]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval,xoldmin, yoldmin, zoldmin, nx_in, ny, nz, uin, one)    
    trilinear_interpolation[blocks_per_grid_out, threads_per_block_out](xout, y, z, val2, nx_old2, ny_old2, nz_old2, hxval2, hyval2, hzval2, xold2min, yold2min, zold2min, nx_out, ny, nz, uout, one)
    

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = uout


def update_after_movement_level3y_2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx, ny_in, nz, uin,  slice_1d_in, slice_1d_out, index_y, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    nx, ny_in, nz: size of uin
    '''
    
    yin = y[0:index_y]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx, ny_in, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0


def update_after_movement_level3y_2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx, ny_in, nz, uin,  slice_1d_in, slice_1d_out, index_y, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    yin = y[index_y+1:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx, ny_in, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0

def update_after_movement_level3x_2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx_in, ny, nz, uin,  slice_1d_in, slice_1d_out, index_x, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    xin = x[0:index_x]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx_in, ny, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0
    
def update_after_movement_level3x_2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx_in, ny, nz, uin,  slice_1d_in, slice_1d_out, index_x, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    xin = x[index_x+1:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx_in, ny, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0







def update_grid_and_precompute(velocity):
    global index_y_lin, index_x_lin, index_y_lin_neg, index_x_lin_neg
    global slice_1d_in_liny, slice_1d_out_liny, slice_1d_in_linx, slice_1d_out_linx
    global slice_1d_in_linx_negative, slice_1d_out_linx_negative
    global slice_1d_in_liny_negative, slice_1d_out_liny_negative
    global index_y_level2, index_x_level2, index_y_neg_level2, index_x_neg_level2
    global slice_1d_in_level2y, slice_1d_out_level2y, slice_1d_in_level2x, slice_1d_out_level2x
    global slice_1d_in_level2x_negative, slice_1d_out_level2x_negative
    global slice_1d_in_level2y_negative, slice_1d_out_level2y_negative
    global index_y, index_x, index_y_neg, index_x_neg
    global slice_1d_iny, slice_1d_outy, slice_1d_inx, slice_1d_outx
    global slice_1d_inx_negative, slice_1d_outx_negative
    global slice_1d_iny_negative, slice_1d_outy_negative
    global xminn, yminn, zminn
    global xmin_level2, ymin_level2, zmin_level2
    global uin_s, uout_s, ny_s_in, ny_s_out, nx_s_in, nx_s_out
    global xoldmin_s, yoldmin_s, zoldmin_s
    global blocks_per_grid_in_s_y, blocks_per_grid_out_s_y
    global blocks_per_grid_in_s_x, blocks_per_grid_out_s_x
    global threads_per_block_in_s_y, threads_per_block_out_s_y
    global threads_per_block_in_s_x, threads_per_block_out_s_x
    global uin_s_level2, uout_s_level2, ny_s_in_level2, ny_s_out_level2
    global nx_s_in_level2, nx_s_out_level2
    global xoldmin_s_level2, yoldmin_s_level2, zoldmin_s_level2
    global blocks_per_grid_in_s_level2_y, blocks_per_grid_out_s_level2_y
    global blocks_per_grid_in_s_level2_x, blocks_per_grid_out_s_level2_x
    global threads_per_block_in_s_level2_y, threads_per_block_out_s_level2_y
    global threads_per_block_in_s_level2_x, threads_per_block_out_s_level2_x
    global uin_lin, uout_lin, ny_lin_in, ny_lin_out
    global nx_lin_in, nx_lin_out
    global xoldmin_lin, yoldmin_lin, zoldmin_lin
    global blocks_per_grid_in_lin_y, blocks_per_grid_out_lin_y
    global blocks_per_grid_in_lin_x, blocks_per_grid_out_lin_x
    global threads_per_block_in_lin_y, threads_per_block_out_lin_y
    global threads_per_block_in_lin_x, threads_per_block_out_lin_x
    
    ### FOR THE GRID MOVEMENT ###

    ## LEVEL 3: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##
    (
        index_y_lin, index_x_lin, index_y_lin_neg, index_x_lin_neg, 
        slice_1d_in_liny, slice_1d_out_liny, slice_1d_in_linx, slice_1d_out_linx, 
        slice_1d_in_linx_negative, slice_1d_out_linx_negative, 
        slice_1d_in_liny_negative, slice_1d_out_liny_negative
    ) = grid_movement_index(x_lin0, y_lin0, velocity, nx_lin, ny_lin, nz_lin, t_len = len(t_vals_lin))

    ## LEVEL 2: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##
    (
        index_y_level2, index_x_level2, index_y_neg_level2, index_x_neg_level2, 
        slice_1d_in_level2y, slice_1d_out_level2y, slice_1d_in_level2x, slice_1d_out_level2x, 
        slice_1d_in_level2x_negative, slice_1d_out_level2x_negative, 
        slice_1d_in_level2y_negative, slice_1d_out_level2y_negative
    ) = grid_movement_index(x_s0_level2, y_s0_level2, velocity, nx_s_level2, ny_s_level2, nz_s_level2, t_len = len(t_vals_lin))

    ## LEVEL 1: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##
    (
        index_y, index_x, index_y_neg, index_x_neg, 
        slice_1d_iny, slice_1d_outy, slice_1d_inx, slice_1d_outx, 
        slice_1d_inx_negative, slice_1d_outx_negative, 
        slice_1d_iny_negative, slice_1d_outy_negative
    ) = grid_movement_index(x_s0, y_s0, velocity, nx_s, ny_s, nz_s, t_len = len(t_vals_lin))

    ### Pre-Compute for update ###
    xminn = float_type(x_lin[0].get())
    yminn = float_type(y_lin[0].get())
    zminn = float_type(z_lin[0].get())
    
    xmin_level2 = float_type(x_s_level2[0].get())
    ymin_level2 = float_type(y_s_level2[0].get())
    zmin_level2 = float_type(z_s_level2[0].get())

    (
        uin_s, uout_s, ny_s_in, ny_s_out, nx_s_in, nx_s_out,
        xoldmin_s, yoldmin_s, zoldmin_s,
        blocks_per_grid_in_s_y, blocks_per_grid_out_s_y,
        blocks_per_grid_in_s_x, blocks_per_grid_out_s_x,
        threads_per_block_in_s_y, threads_per_block_out_s_y,
        threads_per_block_in_s_x, threads_per_block_out_s_x
    ) = precompute_for_update(u_s, nx_s, ny_s, nz_s, y_s, index_y, x_s, index_x, slice_1d_iny, slice_1d_outy, z_s)

    (
        uin_s_level2, uout_s_level2, ny_s_in_level2, ny_s_out_level2, nx_s_in_level2, nx_s_out_level2,
        xoldmin_s_level2, yoldmin_s_level2, zoldmin_s_level2,
        blocks_per_grid_in_s_level2_y, blocks_per_grid_out_s_level2_y,
        blocks_per_grid_in_s_level2_x, blocks_per_grid_out_s_level2_x,
        threads_per_block_in_s_level2_y, threads_per_block_out_s_level2_y,
        threads_per_block_in_s_level2_x, threads_per_block_out_s_level2_x
    ) = precompute_for_update(u_s_level2, nx_s_level2, ny_s_level2, nz_s_level2, y_s_level2, index_y_level2, x_s_level2, index_x_level2, slice_1d_in_level2y, slice_1d_out_level2y, z_s_level2)

    (
        uin_lin, uout_lin, ny_lin_in, ny_lin_out, nx_lin_in, nx_lin_out,
        xoldmin_lin, yoldmin_lin, zoldmin_lin,
        blocks_per_grid_in_lin_y, blocks_per_grid_out_lin_y,
        blocks_per_grid_in_lin_x, blocks_per_grid_out_lin_x,
        threads_per_block_in_lin_y, threads_per_block_out_lin_y,
        threads_per_block_in_lin_x, threads_per_block_out_lin_x
    ) = precompute_for_update(u_lin, nx_lin, ny_lin, nz_lin, y_lin, index_y_lin, x_lin, index_x_lin, slice_1d_in_liny, slice_1d_out_liny, z_lin)

    ### Pre-Compute for update ###

### FUNCS FOR MOVING ###



use_double_precision = True 
if use_double_precision:
    float_type = cp.float64
    pii = cp.pi
    two = 2
    half = 0.5
    one = 1
    four = 4
else:
    float_type = cp.float32
    pii = float_type(cp.pi)
    two = float_type(2)
    half = float_type(0.5)
    one = float_type(1.0)
    four = float_type(4.0)




### FUNCS FOR MOVING ###
#############################################################        


time_multip = 1/4
time_step = 108
  
  
  
Q = 150;   x_span_m = 50e-6;  v = 1.0; x00_initial , y00_initial = 0 , 0 #(Initial laser pos);    #-- LASER INPUTS; USER INPUT


### PHYSICAL PARAMETERS ### --COMPUTE BASED ON USER INPUT
t_span_s = 2 * x_span_m / v
phys = phys_parameter( Q, x_span_m, t_span_s )
x_span = float_type(x_span_m / phys.len_scale)
t_span = float_type(t_span_s / phys.time_scale)
Ste    = float_type(phys.Ste)
iSte   = float_type(1.0 / Ste)
n1     = float_type(phys.n1)
n2     = float_type(phys.n2)
n3     = float_type(phys.n3)
u0     = float_type(phys.u0)
u0 = float_type(phys.u0)
ttt = 0 # Starting time
### PHYSICAL PARAMETERS ### --COMPUTE BASED ON USER INPUT


lxd_level3 = 96*50e-6; lyd_level3 = 96*50e-6; lzd_level3 = 48*50e-6; h_level3 = 18e-6 #-- USER INPUT FOR LEVEL 3

###  -- COMPUTE BASED ON USER INPUT
outer = init_level3_outer(phys, float_type, lxd_level3, lyd_level3, lzd_level3, h_level3, cp)
nx_lin = outer["sp"].nx; ny_lin = outer["sp"].ny; nz_lin = outer["sp"].nz
x_lin, y_lin, z_lin = outer["x_lin"], outer["y_lin"], outer["z_lin"]
dt_lin, h_lin = outer["dt_lin"], outer["h_lin"]
h_linsq, dt_lin05 = outer["h_linsq"], outer["dt_lin05"]
u_lin, u_new_lin, b_lin = outer["u_lin"], outer["u_new_lin"], outer["b_lin"]
t_vals_lin = outer["t_vals_lin"]
x_lin0 = x_lin.copy()
y_lin0 = y_lin.copy()
z_lin0 = z_lin.copy()
velocity = float_type(v / ( (phys.len_scale / phys.time_scale) / dt_lin)) 
velocity0 = velocity

###  -- COMPUTE BASED ON USER INPUT



###  -- COMPUTE BASED ON USER INPUT
    ### Initialize Variables ###
u_lin =u0 * cp.ones(nx_lin*ny_lin*nz_lin,dtype=float_type) 
u_new_lin = u0 * cp.ones(nx_lin*ny_lin*nz_lin,dtype=float_type) 
b_lin = cp.ones(nx_lin*ny_lin*nz_lin, dtype=float_type)
qs_lin = cp.zeros([nx_lin, ny_lin],dtype=float_type) 
y00 = y00_initial
x00 = x00_initial

z00_end = z_lin[-1]
t00 = float_type(outer["sp"].t_end/phys.time_scale * 0.5)
Y_lin, X_lin = cp.meshgrid(y_lin, x_lin) 
Zs2_xy_lin =  gaussian2d(X_lin, Y_lin, x_span, x00_initial, y00_initial)
Y_lin0, X_lin0 = Y_lin.copy(), X_lin.copy()
    ### Initialize Variables ###
###  -- COMPUTE BASED ON USER INPUT



lxd_level1 = 12*50e-6; lyd_level1 = 12*50e-6; lzd_level1 = 6*50e-6; h_level1_factor = 4.0 #-- USER INPUT FOR LEVEL 1
to_code = lambda Lm: float_type(Lm / phys.len_scale)

###  -- COMPUTE BASED ON USER INPUT; Initialize Level 1
inner1 = init_inner_level(
    outer, x_span,                       # x_span already in code units
    size=(to_code(lxd_level1), to_code(lyd_level1), to_code(lzd_level1)),
    h_factor=h_level1_factor,
    center=(float_type(x00_initial), float_type(y00_initial)),
    z_end=outer["z_lin"][-1],
    float_type=float_type,
    xp=cp,
)

nx_s, ny_s, nz_s = inner1["nx_s"], inner1["ny_s"], inner1["nz_s"]
x_s, y_s, z_s = inner1["x_s"], inner1["y_s"], inner1["z_s"]
x_s2, y_s2, z_s2 = inner1["x_s2"], inner1["y_s2"], inner1["z_s2"]
h_x_new, h_y_new, h_z_new = inner1["h_x_new"], inner1["h_y_new"], inner1["h_z_new"]
u_s, u_new_s, b_s = inner1["u_s"], inner1["u_new_s"], inner1["b_s"]
p_r, p_l, p_b, p_o, p_i = inner1["p_r"], inner1["p_l"], inner1["p_b"], inner1["p_o"], inner1["p_i"]
p_r_old, p_l_old, p_b_old, p_o_old, p_i_old = inner1["p_r_old"], inner1["p_l_old"], inner1["p_b_old"], inner1["p_o_old"], inner1["p_i_old"]
slice_bc_right0_1d = inner1["slice_bc_right0_1d"]
slice_bc_left0_1d  = inner1["slice_bc_left0_1d"]
slice_bc_bottom0_1d= inner1["slice_bc_bottom0_1d"]
slice_bc_in0_1d    = inner1["slice_bc_in0_1d"]
slice_bc_out0_1d   = inner1["slice_bc_out0_1d"]
qs_s = cp.zeros([nx_s, ny_s],dtype=float_type) 






Y_s, X_s = inner1["Y_s"], inner1["X_s"]
inner1["Zs2_xy_s"] = gaussian2d(X_s, Y_s, x_span, float_type(x00_initial), float_type(y00_initial))

h_ix_new = 1/h_x_new
h_iy_new = 1/h_y_new
h_iz_new = 1/h_z_new
h_ix_newsq, h_iy_newsq, h_iz_newsq = float_type(h_ix_new**2), float_type(h_iy_new**2), float_type(h_iz_new**2)


x_s0, y_s0, z_s0 = x_s.copy(), y_s.copy(), z_s.copy()
x_s20, y_s20, z_s20 = x_s2.copy(), y_s2.copy(), z_s2.copy()

X_s0, Y_s0 = X_s.copy(), Y_s.copy()
uinteold = cp.zeros([x_s2.shape[0] * y_s2.shape[0] * z_s2.shape[0]], dtype=float_type)
uinte = cp.zeros([x_s2.shape[0] * y_s2.shape[0] * z_s2.shape[0]], dtype=float_type)
nx_s2, ny_s2, nz_s2  = nx_s +2, ny_s+2, nz_s +1
###  -- COMPUTE BASED ON USER INPUT; Initialize Level 1


lxd_level2_m = 20 * 50e-6 ; lyd_level2_m = 20 * 50e-6 ; lzd_level2_m = 10 * 50e-6 ; h_level2_factor = 2.0 #-- USER INPUT FOR LEVEL 2


###  -- COMPUTE BASED ON USER INPUT; Initialize Level 2
inner2 = init_inner_level(
    outer, x_span,
    size=(to_code(lxd_level2_m), to_code(lyd_level2_m), to_code(lzd_level2_m)),
    h_factor=h_level2_factor,
    center=(float_type(x00_initial), float_type(y00_initial)),
    z_end=outer["z_lin"][-1],
    float_type=float_type,
    xp=cp,
)

nx_s_level2, ny_s_level2, nz_s_level2 = inner2["nx_s"], inner2["ny_s"], inner2["nz_s"]
x_s_level2,  y_s_level2,  z_s_level2  = inner2["x_s"], inner2["y_s"], inner2["z_s"]
x_s2_level2, y_s2_level2, z_s2_level2 = inner2["x_s2"], inner2["y_s2"], inner2["z_s2"]

h_x_new_level2 = inner2["h_x_new"]
h_y_new_level2 = inner2["h_y_new"]
h_z_new_level2 = inner2["h_z_new"]

u_s_level2      = inner2["u_s"]
u_new_s_level2  = inner2["u_new_s"]
b_s_level2      = inner2["b_s"]

p_r_level2 = inner2["p_r"];  p_l_level2 = inner2["p_l"]
p_b_level2 = inner2["p_b"];  p_o_level2 = inner2["p_o"];  p_i_level2 = inner2["p_i"]

p_r_old_level2 = inner2["p_r_old"]; p_l_old_level2 = inner2["p_l_old"]
p_b_old_level2 = inner2["p_b_old"]; p_o_old_level2 = inner2["p_o_old"]; p_i_old_level2 = inner2["p_i_old"]

slice_bc_right0_1d_level2  = inner2["slice_bc_right0_1d"]
slice_bc_left0_1d_level2   = inner2["slice_bc_left0_1d"]
slice_bc_bottom0_1d_level2 = inner2["slice_bc_bottom0_1d"]
slice_bc_in0_1d_level2     = inner2["slice_bc_in0_1d"]
slice_bc_out0_1d_level2    = inner2["slice_bc_out0_1d"]
qs_s_level2 = cp.zeros([nx_s_level2, ny_s_level2],dtype=float_type) 





Y_s_level2, X_s_level2 = inner2["Y_s"], inner2["X_s"]
inner2["Zs2_xy_s"] = gaussian2d(X_s_level2, Y_s_level2, x_span, float_type(x00_initial), float_type(y00_initial))

h_ix_new_level2 = 1 / h_x_new_level2
h_iy_new_level2 = 1 / h_y_new_level2
h_iz_new_level2 = 1 / h_z_new_level2

h_ix_newsq_level2 = float_type(h_ix_new_level2**2)
h_iy_newsq_level2 = float_type(h_iy_new_level2**2)
h_iz_newsq_level2 = float_type(h_iz_new_level2**2)


x_s0_level2, y_s0_level2, z_s0_level2   = x_s_level2.copy(), y_s_level2.copy(), z_s_level2.copy()
x_s20_level2, y_s20_level2, z_s20_level2 = x_s2_level2.copy(), y_s2_level2.copy(), z_s2_level2.copy()

X_s0_level2, Y_s0_level2 = X_s_level2.copy(), Y_s_level2.copy()

uinteold_level2 = cp.zeros([x_s2_level2.shape[0] * y_s2_level2.shape[0] * z_s2_level2.shape[0]], dtype=float_type)
uinte_level2 = cp.zeros([x_s2_level2.shape[0] * y_s2_level2.shape[0] * z_s2_level2.shape[0]], dtype=float_type)
nx_s2_level2, ny_s2_level2, nz_s2_level2  = nx_s_level2 +2, ny_s_level2+2, nz_s_level2 +1
###  -- COMPUTE BASED ON USER INPUT; Initialize Level 2





#############################################################


### FOR THE GRID MOVEMENT ###

    ## LEVEL 3: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##
(
    index_y_lin, index_x_lin, index_y_lin_neg, index_x_lin_neg, 
    slice_1d_in_liny, slice_1d_out_liny, slice_1d_in_linx, slice_1d_out_linx, 
    slice_1d_in_linx_negative, slice_1d_out_linx_negative, 
    slice_1d_in_liny_negative, slice_1d_out_liny_negative
) = grid_movement_index(x_lin0, y_lin0, velocity, nx_lin, ny_lin, nz_lin)
    ## LEVEL 3: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ## 

    ## LEVEL 2: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##    
(
    index_y_level2, index_x_level2, index_y_neg_level2, index_x_neg_level2, 
    slice_1d_in_level2y, slice_1d_out_level2y, slice_1d_in_level2x, slice_1d_out_level2x, 
    slice_1d_in_level2x_negative, slice_1d_out_level2x_negative, 
    slice_1d_in_level2y_negative, slice_1d_out_level2y_negative
) = grid_movement_index(x_s0_level2, y_s0_level2, velocity, nx_s_level2, ny_s_level2, nz_s_level2)

    ## LEVEL 2: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##    

    ## LEVEL 1: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##
(
    index_y, index_x, index_y_neg, index_x_neg, 
    slice_1d_iny, slice_1d_outy, slice_1d_inx, slice_1d_outx, 
    slice_1d_inx_negative, slice_1d_outx_negative, 
    slice_1d_iny_negative, slice_1d_outy_negative
) = grid_movement_index(x_s0, y_s0, velocity, nx_s, ny_s, nz_s)

    ## LEVEL 1: FIND THE CELLS FALL INSIDE AND OUTSIDE AFTER EVERY MOVEMENT ##




### FOR THE GRID MOVEMENT ###


### Pre-Compute for update after shifting (stays the same throughout the simulation if velocity always constant) ###
xminn = float_type(x_lin[0].get()); yminn = float_type(y_lin[0].get()); zminn = float_type(z_lin[0].get())
xmin_level2 = float_type(x_s_level2[0].get()); ymin_level2 = float_type(y_s_level2[0].get()); zmin_level2 = float_type(z_s_level2[0].get())

# --- Level 1 (inner1)
pre_s = precompute_for_update(
    u_s, nx_s, ny_s, nz_s,
    y_s, index_y,
    x_s, index_x,
    slice_1d_iny, slice_1d_outy,
    z_s,
    float_type=float_type,
)

uin_s                   = pre_s["u_in"]
uout_s                  = pre_s["u_out"]
ny_s_in                 = pre_s["ny_in"]
ny_s_out                = pre_s["ny_out"]
nx_s_in                 = pre_s["nx_in"]
nx_s_out                = pre_s["nx_out"]
xoldmin_s               = pre_s["xoldmin"]
yoldmin_s               = pre_s["yoldmin"]
zoldmin_s               = pre_s["zoldmin"]
blocks_per_grid_in_s_y  = pre_s["blocks_in_y"]
blocks_per_grid_out_s_y = pre_s["blocks_out_y"]
blocks_per_grid_in_s_x  = pre_s["blocks_in_x"]
blocks_per_grid_out_s_x = pre_s["blocks_out_x"]
threads_per_block_in_s_y  = pre_s["tpbin"]
threads_per_block_out_s_y = pre_s["tpbout"]
threads_per_block_in_s_x  = pre_s["tpbin_x"]
threads_per_block_out_s_x = pre_s["tpbout_x"]

# --- Level 2 (inner2)
pre_s_level2 = precompute_for_update(
    u_s_level2, nx_s_level2, ny_s_level2, nz_s_level2,
    y_s_level2, index_y_level2,
    x_s_level2, index_x_level2,
    slice_1d_in_level2y, slice_1d_out_level2y,
    z_s_level2,
    float_type=float_type,
)

uin_s_level2                   = pre_s_level2["u_in"]
uout_s_level2                  = pre_s_level2["u_out"]
ny_s_in_level2                 = pre_s_level2["ny_in"]
ny_s_out_level2                = pre_s_level2["ny_out"]
nx_s_in_level2                 = pre_s_level2["nx_in"]
nx_s_out_level2                = pre_s_level2["nx_out"]
xoldmin_s_level2               = pre_s_level2["xoldmin"]
yoldmin_s_level2               = pre_s_level2["yoldmin"]
zoldmin_s_level2               = pre_s_level2["zoldmin"]
blocks_per_grid_in_s_level2_y  = pre_s_level2["blocks_in_y"]
blocks_per_grid_out_s_level2_y = pre_s_level2["blocks_out_y"]
blocks_per_grid_in_s_level2_x  = pre_s_level2["blocks_in_x"]
blocks_per_grid_out_s_level2_x = pre_s_level2["blocks_out_x"]
threads_per_block_in_s_level2_y  = pre_s_level2["tpbin"]
threads_per_block_out_s_level2_y = pre_s_level2["tpbout"]
threads_per_block_in_s_level2_x  = pre_s_level2["tpbin_x"]
threads_per_block_out_s_level2_x = pre_s_level2["tpbout_x"]

# --- Level 3 (outer/linear)
pre_lin = precompute_for_update(
    u_lin, nx_lin, ny_lin, nz_lin,
    y_lin, index_y_lin,
    x_lin, index_x_lin,
    slice_1d_in_liny, slice_1d_out_liny,
    z_lin,
    float_type=float_type,
)

uin_lin                   = pre_lin["u_in"]
uout_lin                  = pre_lin["u_out"]
ny_lin_in                 = pre_lin["ny_in"]
ny_lin_out                = pre_lin["ny_out"]
nx_lin_in                 = pre_lin["nx_in"]
nx_lin_out                = pre_lin["nx_out"]
xoldmin_lin               = pre_lin["xoldmin"]
yoldmin_lin               = pre_lin["yoldmin"]
zoldmin_lin               = pre_lin["zoldmin"]
blocks_per_grid_in_lin_y  = pre_lin["blocks_in_y"]
blocks_per_grid_out_lin_y = pre_lin["blocks_out_y"]
blocks_per_grid_in_lin_x  = pre_lin["blocks_in_x"]
blocks_per_grid_out_lin_x = pre_lin["blocks_out_x"]
threads_per_block_in_lin_y  = pre_lin["tpbin"]
threads_per_block_out_lin_y = pre_lin["tpbout"]
threads_per_block_in_lin_x  = pre_lin["tpbin_x"]
threads_per_block_out_lin_x = pre_lin["tpbout_x"]
### Pre-Compute for update after shifting (stays the same throughout the simulation if velocity always constant) ###


#### GPU PARAMETERS ###
# Level 3 (outer / linear)
blocks_per_grid_lin, threads_per_block_lin = launch_3d(nx_lin, ny_lin, nz_lin)

# Level 1
blocks_per_grid_s, threads_per_block_s = launch_3d(nx_s, ny_s, nz_s)

# Level 2
blocks_per_grid_s_level2, threads_per_block_s_level2 = launch_3d(nx_s_level2, ny_s_level2, nz_s_level2)

# BC launches (faces)
blocks_per_grid_s_bc1, threads_per_block_bc1 = launch_bc(nx_s_level2 * nz_s_level2)
blocks_per_grid_s_bc2, threads_per_block_bc2 = launch_bc(nx_s_level2 * ny_s_level2)
blocks_per_grid_s_bc3, threads_per_block_bc3 = launch_bc(nx_s * nz_s)
blocks_per_grid_s_bc4, threads_per_block_bc4 = launch_bc(nx_s * ny_s)

# Interpolation buffers
blocks_per_grid_s_int1, threads_per_block_int1 = launch_3d(nx_s2_level2, ny_s2_level2, nz_s2_level2)
blocks_per_grid_s_int2, threads_per_block_int2 = launch_3d(nx_s2, ny_s2, nz_s2)
blocks_per_grid_s_int3, threads_per_block_int3 = launch_3d(nx_s, ny_s, nz_s)
blocks_per_grid_s_int4, threads_per_block_int4 = launch_3d(nx_s_level2, ny_s_level2, nz_s_level2)
### GPU PARAMETERS ###  


### MATVECS FOR LEVEL 3,1 AND 2 ###
result_lin = cp.zeros_like(u_lin, dtype=float_type)
result_s = cp.zeros_like(u_s, dtype=float_type)
result_s_level2 = cp.zeros_like(u_s_level2, dtype=float_type)

d_result_lin = cuda.to_device(result_lin)
d_result_s = cuda.to_device(result_s)
d_result_s_level2 = cuda.to_device(result_s_level2)

Alinear_lin = spla.LinearOperator((nx_lin*ny_lin*nz_lin , nx_lin*ny_lin*nz_lin), matvec=mv_wrapper_lin)
Alinear_s = spla.LinearOperator((nx_s*ny_s*nz_s, nx_s*ny_s*nz_s), matvec=mv_wrapper_s)
Alinear_s_level2  = spla.LinearOperator((nx_s_level2 *ny_s_level2 *nz_s_level2 , nx_s_level2 *ny_s_level2 *nz_s_level2 ), matvec=mv_wrapper_s_level2 )
### MATVECS FOR LEVEL 3, 1 AND 2 ###



### Variables for fixed-point it ###
non_lin_it_lv1 = 2
w_lv1 = float_type(2/3) 
u_s_level2_old = u_s_level2.copy()
u_s_old = u_s.copy()
### Variables for fixed-point it ###



### For multi-layer ###
num_layers = 7; layer_thickness = 50e-6 ### -- USER INPUT

### For multi-layer ### -- COMPUTED BASED ON USER INPUT
lt_sc = float_type(float_type(layer_thickness)/phys.len_scale) #layer_thickness scaled
lay_ind_lin = int(lt_sc/h_lin)
lay_ind_s = int((h_lin*lay_ind_lin)/h_z_new) 
lay_ind_s_level2 = int((h_lin*lay_ind_lin)/h_z_new_level2 )

u_s3d = shape_back_3d(u_s.copy(), nx_s, ny_s, nz_s)
u_s_level23d = shape_back_3d(u_s_level2.copy(), nx_s_level2, ny_s_level2, nz_s_level2) 
u_lin3d = shape_back_3d(u_lin.copy(), nx_lin, ny_lin, nz_lin) 

u_s3d_new = u_s3d.copy()
u_s_level23d_new = u_s_level23d.copy()
u_lin3d_new = u_lin3d.copy()
### For multi-layer ### -- COMPUTED BASED ON USER INPUT
### For multi-layer ###





### CREATE CUDA ARRAYS ###
# --- level 3 (linear) ---
(d_u_lin,) = (cuda.to_device(u_lin),)
d_qs_lin, = (cuda.to_device(qs_lin),)
d_x_lin, d_y_lin, d_z_lin = map(cuda.to_device, (x_lin, y_lin, z_lin))
d_b_lin, = (cuda.to_device(b_lin),)

# --- level 1 ---
d_u_s, = (cuda.to_device(u_s),)
d_u_s_old, = (cuda.to_device(u_s_old),)
d_b_s, = (cuda.to_device(b_s),)
d_qs_s, = (cuda.to_device(qs_s),)
d_x_s, d_y_s, d_z_s = map(cuda.to_device, (x_s, y_s, z_s))
d_x_s2, d_y_s2, d_z_s2 = map(cuda.to_device, (x_s2, y_s2, z_s2))

# BC current
d_p_o, d_p_i, d_p_r, d_p_l, d_p_b = map(cuda.to_device, (p_o, p_i, p_r, p_l, p_b))
# BC old
d_p_o_old, d_p_i_old, d_p_r_old, d_p_l_old, d_p_b_old = map(
    cuda.to_device, (p_o_old, p_i_old, p_r_old, p_l_old, p_b_old)
)

# interpolation buffers
d_uinte, d_uinteold = map(cuda.to_device, (uinte, uinteold))
# in/out buffers for movement
d_uin_s, d_uout_s = map(cuda.to_device, (uin_s, uout_s))

# --- level 2 ---
d_u_s_level2, = (cuda.to_device(u_s_level2),)
d_u_s_level2_old, = (cuda.to_device(u_s_level2_old),)
d_b_s_level2, = (cuda.to_device(b_s_level2),)
d_qs_s_level2, = (cuda.to_device(qs_s_level2),)
d_x_s_level2, d_y_s_level2, d_z_s_level2 = map(cuda.to_device, (x_s_level2, y_s_level2, z_s_level2))
d_x_s2_level2, d_y_s2_level2, d_z_s2_level2 = map(cuda.to_device, (x_s2_level2, y_s2_level2, z_s2_level2))

# level 2 BC current
d_p_o_level2, d_p_i_level2, d_p_r_level2, d_p_l_level2, d_p_b_level2 = map(
    cuda.to_device, (p_o_level2, p_i_level2, p_r_level2, p_l_level2, p_b_level2)
)
# level 2 BC old
d_p_o_old_level2, d_p_i_old_level2, d_p_r_old_level2, d_p_l_old_level2, d_p_b_old_level2 = map(
    cuda.to_device, (p_o_old_level2, p_i_old_level2, p_r_old_level2, p_l_old_level2, p_b_old_level2)
)

# level 2 interpolation
d_uinte_level2, d_uinteold_level2 = map(cuda.to_device, (uinte_level2, uinteold_level2))
# level 2 in/out movement buffers
d_uin_s_level2, d_uout_s_level2 = map(cuda.to_device, (uin_s_level2, uout_s_level2))
### CREATE CUDA ARRAYS ###


movement = 3 #y movement
iii2 = 0





for layers in range(num_layers):

    
    
    cntrr = 0
    movement = 3
    y00 = y00_initial
    x00 = x00_initial
    
    qs_lin[:] = gaussian2d(X_lin, Y_lin, x_span, x00, y00)
    qs_s[:] = gaussian2d(X_s, Y_s, x_span, x00, y00)
    qs_s_level2[:] = gaussian2d(X_s_level2, Y_s_level2, x_span, x00, y00)


    if layers > 0 :
        resety_all(y_s0, Y_s0, y_s20, y_s0_level2, Y_s0_level2, y_s20_level2, y_lin0, Y_lin0 )
        resetx_all(x_s0, X_s0, x_s20, x_s0_level2, X_s0_level2, x_s20_level2, x_lin0, X_lin0 )
        xminn = float_type(x_lin[0]); yminn = float_type(y_lin[0]); zminn = float_type(z_lin[0])
        xmin_level2 = float_type(x_s_level2[0]); ymin_level2 = float_type(y_s_level2[0]); zmin_level2 = float_type(z_s_level2[0])
        


        
        zoldmin_s = float_type(z_s[0].get())
        zoldmin_s_level2 = float_type(z_s_level2[0].get())
        zoldmin_lin = float_type(z_lin[0].get())
        
        ### Pre-Compute for update ###
        xminn = float_type(x_lin[0].get()); yminn = float_type(y_lin[0].get()); zminn = float_type(z_lin[0].get())
        xmin_level2 = float_type(x_s_level2[0].get()); ymin_level2 = float_type(y_s_level2[0].get()); zmin_level2 = float_type(z_s_level2[0].get())

        uin_s = cp.zeros_like(u_s[slice_1d_iny], dtype = float_type)
        uin_s_level2 =  cp.zeros_like(u_s_level2[slice_1d_in_level2y], dtype = float_type)
        uout_s = cp.zeros_like(u_s[slice_1d_outy], dtype = float_type)
        uout_s_level2 = cp.zeros_like(u_s_level2[slice_1d_out_level2y], dtype = float_type)
        uin_lin =  cp.zeros_like(u_lin[slice_1d_in_liny], dtype = float_type)

        ny_lin_in = int(y_lin[0:index_y_lin].shape[0])
        ny_s_in  = int(y_s[0:index_y].shape[0])
        ny_s_out  = int(y_s[index_y:].shape[0])
        ny_s_in_level2  = int(y_s_level2[0:index_y_level2].shape[0])
        ny_s_out_level2  = int(y_s_level2[index_y_level2:].shape[0])

        nx_lin_in = int(x_lin[0:index_x_lin].shape[0])
        nx_s_in  = int(x_s[0:index_x].shape[0])
        nx_s_out  = int(x_s[index_x:].shape[0])
        nx_s_in_level2  = int(x_s_level2[0:index_x_level2].shape[0])
        nx_s_out_level2  = int(x_s_level2[index_x_level2:].shape[0])


        xoldmin_s = float_type(x_s[0].get())
        yoldmin_s = float_type(y_s[0].get())
        zoldmin_s = float_type(z_s[0].get())

        xoldmin_s_level2 = float_type(x_s_level2[0].get())
        yoldmin_s_level2 = float_type(y_s_level2[0].get())
        zoldmin_s_level2 = float_type(z_s_level2[0].get())

        xoldmin_lin = float_type(x_lin[0].get())
        yoldmin_lin = float_type(y_lin[0].get())
        zoldmin_lin = float_type(z_lin[0].get())

        threads_per_block_in_s_y = 128
        blocks_per_grid_in_s_y =  (nx_s*ny_s_in*nz_s + (threads_per_block_in_s_y-1)) // threads_per_block_in_s_y
        threads_per_block_out_s_y = 128
        blocks_per_grid_out_s_y =  (nx_s*ny_s_out*nz_s + (threads_per_block_out_s_y-1)) // threads_per_block_out_s_y

        threads_per_block_in_lin_y = 128
        blocks_per_grid_in_lin_y =  (nx_lin*ny_lin_in*nz_lin + (threads_per_block_in_lin_y-1)) // threads_per_block_in_lin_y

        threads_per_block_in_s_level2_y = 128
        blocks_per_grid_in_s_level2_y =  (nx_s_level2*ny_s_in_level2*nz_s_level2 + (threads_per_block_in_s_level2_y-1)) // threads_per_block_in_s_level2_y

        threads_per_block_out_s_level2_y = 128
        blocks_per_grid_out_s_level2_y =  (nx_s_level2*ny_s_out_level2*nz_s_level2 + (threads_per_block_out_s_level2_y-1)) // threads_per_block_out_s_level2_y


        threads_per_block_in_s_x = 128
        blocks_per_grid_in_s_x =  (nx_s*ny_s_in*nz_s + (threads_per_block_in_s_x-1)) // threads_per_block_in_s_x

        threads_per_block_out_s_x = 128
        blocks_per_grid_out_s_x =  (nx_s*ny_s_out*nz_s + (threads_per_block_out_s_x-1)) // threads_per_block_out_s_x

        threads_per_block_in_lin_x = 128
        blocks_per_grid_in_lin_x =  (nx_lin*ny_lin_in*nz_lin + (threads_per_block_in_lin_x-1)) // threads_per_block_in_lin_x

        threads_per_block_in_s_level2_x = 128
        blocks_per_grid_in_s_level2_x =  (nx_s_level2*ny_s_in_level2*nz_s_level2 + (threads_per_block_in_s_level2_x-1)) // threads_per_block_in_s_level2_x

        threads_per_block_out_s_level2_x = 128
        blocks_per_grid_out_s_level2_x =  (nx_s_level2*ny_s_out_level2*nz_s_level2 + (threads_per_block_out_s_level2_x-1)) // threads_per_block_out_s_level2_x
        ### Pre-Compute for update ###



  
 
    for iii, t_val in enumerate(t_vals_lin):

        ttt += dt_lin*phys.time_scale*1e3
        if velocity != velocity0:
            velocity0 = velocity

            update_grid_and_precompute(velocity)
            print('velocity updated at iii = ', iii)
            print(index_y, 'is index_y')
            d_u_lin = cuda.to_device(u_lin)
            d_p_o = cuda.to_device(p_o)
            d_p_i = cuda.to_device(p_i)
            d_p_r = cuda.to_device(p_r)
            d_p_l = cuda.to_device(p_l)
            d_p_b = cuda.to_device(p_b)

            d_p_o_level2 = cuda.to_device(p_o_level2)
            d_p_i_level2 = cuda.to_device(p_i_level2)
            d_p_r_level2 = cuda.to_device(p_r_level2)
            d_p_l_level2 = cuda.to_device(p_l_level2)
            d_p_b_level2 = cuda.to_device(p_b_level2)

            d_u_s = cuda.to_device(u_s)
            d_u_s_level2 = cuda.to_device(u_s_level2)

            d_b_lin = cuda.to_device(b_lin)
            d_b_s = cuda.to_device(b_s)
            d_b_s_level2 = cuda.to_device(b_s_level2)

            d_uinte_level2 = cuda.to_device(uinte_level2)
            d_uinte = cuda.to_device(uinte)

            d_p_o_old = cuda.to_device(p_o_old)
            d_p_i_old = cuda.to_device(p_i_old)
            d_p_r_old = cuda.to_device(p_r_old)
            d_p_l_old = cuda.to_device(p_l_old)
            d_p_b_old = cuda.to_device(p_b_old)

            d_p_o_old_level2 = cuda.to_device(p_o_old_level2)
            d_p_i_old_level2 = cuda.to_device(p_i_old_level2)
            d_p_r_old_level2 = cuda.to_device(p_r_old_level2)
            d_p_l_old_level2 = cuda.to_device(p_l_old_level2)
            d_p_b_old_level2 = cuda.to_device(p_b_old_level2)

            d_qs_lin = cuda.to_device(qs_lin)
            d_qs_s_level2 = cuda.to_device(qs_s_level2)
            d_qs_s = cuda.to_device(qs_s)


            d_x_lin = cuda.to_device(x_lin)
            d_y_lin = cuda.to_device(y_lin)
            d_z_lin = cuda.to_device(z_lin)

            d_x_s = cuda.to_device(x_s)
            d_y_s = cuda.to_device(y_s)
            d_z_s = cuda.to_device(z_s)

            d_x_s2 = cuda.to_device(x_s2)
            d_y_s2 = cuda.to_device(y_s2)
            d_z_s2 = cuda.to_device(z_s2)

            d_x_s_level2 = cuda.to_device(x_s_level2)
            d_y_s_level2 = cuda.to_device(y_s_level2)
            d_z_s_level2 = cuda.to_device(z_s_level2)

            d_x_s2_level2 = cuda.to_device(x_s2_level2)
            d_y_s2_level2 = cuda.to_device(y_s2_level2)
            d_z_s2_level2 = cuda.to_device(z_s2_level2)

            d_uout_s = cuda.to_device(uout_s)
            d_uout_s_level2 = cuda.to_device(uout_s_level2)

            d_uin_s = cuda.to_device(uin_s)
            d_uin_s_level2 = cuda.to_device(uin_s_level2)

            d_u_s_level2_old = cuda.to_device(u_s_level2_old)
            d_u_s_old = cuda.to_device(u_s_old)

            d_uinteold = cuda.to_device(uinteold)
            d_uinteold_level2 = cuda.to_device(uinteold_level2)

        if movement==1 and iii and velocity : # means y-movement
        ### Move the Laser Source ###
            #print('here = ')
            y00 += velocity
        ### Move the Laser Source ###
        
        ### Move Grid and update variables  ###
    
            move_gridy_all(velocity, iii)
            
            update_after_movementy2(x=d_x_s, y=y_s, z=d_z_s, val=d_u_s, nx_old=nx_s, ny_old=ny_s, nz_old=nz_s, hxval=h_x_new, hyval=h_y_new, hzval=h_z_new, xoldmin=xoldmin_s, yoldmin=yoldmin_s, zoldmin=zoldmin_s,  nx=nx_s, ny_in=ny_s_in, nz=nz_s, uin=uin_s, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  ny_out=ny_s_out, uout=d_uout_s, one=one, val3=u_s, slice_1d_in=slice_1d_iny, slice_1d_out=slice_1d_outy, index_y=index_y, threads_per_block_in=threads_per_block_in_s_y, blocks_per_grid_in=blocks_per_grid_in_s_y, threads_per_block_out=threads_per_block_out_s_y , blocks_per_grid_out=blocks_per_grid_out_s_y)
            update_after_movementy2(x=d_x_s_level2, y=y_s_level2, z=d_z_s_level2, val=d_u_s_level2, nx_old=nx_s_level2, ny_old=ny_s_level2, nz_old=nz_s_level2, hxval=h_x_new_level2, hyval=h_y_new_level2, hzval=h_z_new_level2, xoldmin=xoldmin_s_level2, yoldmin=yoldmin_s_level2, zoldmin=zoldmin_s_level2,  nx=nx_s_level2, ny_in=ny_s_in_level2, nz=nz_s_level2, uin=uin_s_level2, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  ny_out=ny_s_out_level2, uout=d_uout_s_level2, one=one, val3=u_s_level2, slice_1d_in=slice_1d_in_level2y, slice_1d_out=slice_1d_out_level2y, index_y=index_y_level2, threads_per_block_in=threads_per_block_in_s_level2_y, blocks_per_grid_in=blocks_per_grid_in_s_level2_y, threads_per_block_out=threads_per_block_out_s_level2_y , blocks_per_grid_out=blocks_per_grid_out_s_level2_y)
            update_after_movement_level3y_2(x=d_x_lin, y=y_lin, z=d_z_lin, val=u_lin, nx_old=nx_lin, ny_old=ny_lin, nz_old=nz_lin, hxval=h_lin, hyval=h_lin, hzval=h_lin, xoldmin_lin=xoldmin_lin, yoldmin_lin=yoldmin_lin, zoldmin_lin=zoldmin_lin,  nx=nx_lin, ny_in=ny_lin_in, nz=nz_lin, uin=uin_lin,  slice_1d_in=slice_1d_in_liny, slice_1d_out=slice_1d_out_liny, index_y=index_y_lin, u0=u0 , one=one,  val3=u_lin, threads_per_block_in=threads_per_block_in_lin_y , blocks_per_grid_in=blocks_per_grid_in_lin_y)
            
            ### Updated Laser Source ###
            qs_lin[:] = gaussian2d(X_lin, Y_lin, x_span, x00, y00)
            qs_s[:] = gaussian2d(X_s, Y_s, x_span, x00, y00)
            qs_s_level2[:] = gaussian2d(X_s_level2, Y_s_level2, x_span, x00, y00)
            ### Updated Laser Source ###
            
            ### New old mins for update ###
            yoldmin_s += velocity
            yoldmin_lin += velocity
            yoldmin_s_level2 += velocity
            ### New old mins for update ###
            
            ### New min coords for interpolation ###)
            yminn +=  velocity
            ymin_level2 += velocity
            ### New min coords for interpolation ###
        
        ### Move Grid and update variables  ###
        elif  movement == 3 and iii and velocity: 
            ### Move the Laser Source ###
                y00 -= velocity
            ### Move the Laser Source ###
            
            ### Move Grid and update variables  ###

                move_gridy_all(-velocity, iii)
                                                
                update_after_movementy2_negative(x=d_x_s, y=y_s, z=d_z_s, val=d_u_s, nx_old=nx_s, ny_old=ny_s, nz_old=nz_s, hxval=h_x_new, hyval=h_y_new, hzval=h_z_new, xoldmin=xoldmin_s, yoldmin=yoldmin_s, zoldmin=zoldmin_s,  nx=nx_s, ny_in=ny_s_in, nz=nz_s, uin=uin_s, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  ny_out=ny_s_out, uout=d_uout_s, one=one, val3=u_s, slice_1d_in=slice_1d_iny_negative, slice_1d_out=slice_1d_outy_negative, index_y=0, threads_per_block_in=threads_per_block_in_s_y, blocks_per_grid_in=blocks_per_grid_in_s_y, threads_per_block_out=threads_per_block_out_s_y , blocks_per_grid_out=blocks_per_grid_out_s_y)                                                
                update_after_movementy2_negative(x=d_x_s_level2, y=y_s_level2, z=d_z_s_level2, val=d_u_s_level2, nx_old=nx_s_level2, ny_old=ny_s_level2, nz_old=nz_s_level2, hxval=h_x_new_level2, hyval=h_y_new_level2, hzval=h_z_new_level2, xoldmin=xoldmin_s_level2, yoldmin=yoldmin_s_level2, zoldmin=zoldmin_s_level2,  nx=nx_s_level2, ny_in=ny_s_in_level2, nz=nz_s_level2, uin=uin_s_level2, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  ny_out=ny_s_out_level2, uout=d_uout_s_level2, one=one, val3=u_s_level2, slice_1d_in=slice_1d_in_level2y_negative, slice_1d_out=slice_1d_out_level2y_negative, index_y=0, threads_per_block_in=threads_per_block_in_s_level2_y, blocks_per_grid_in=blocks_per_grid_in_s_level2_y, threads_per_block_out=threads_per_block_out_s_level2_y , blocks_per_grid_out=blocks_per_grid_out_s_level2_y)                                                      
                update_after_movement_level3y_2_negative(x=d_x_lin, y=y_lin, z=d_z_lin, val=d_u_lin, nx_old=nx_lin, ny_old=ny_lin, nz_old=nz_lin,  hxval=h_lin, hyval=h_lin, hzval=h_lin, xoldmin_lin=xoldmin_lin, yoldmin_lin=yoldmin_lin, zoldmin_lin=zoldmin_lin,  nx=nx_lin, ny_in=ny_lin_in, nz=nz_lin, uin=uin_lin,  slice_1d_in=slice_1d_in_liny_negative, slice_1d_out=slice_1d_out_liny_negative, index_y=0, u0=u0 , one=one,  val3=u_lin, threads_per_block_in=threads_per_block_in_lin_y , blocks_per_grid_in=blocks_per_grid_in_lin_y)            


                ### Updated Laser Source ###
                qs_lin[:] = gaussian2d(X_lin, Y_lin, x_span, x00, y00)
                qs_s[:] = gaussian2d(X_s, Y_s, x_span, x00, y00)
                qs_s_level2[:] = gaussian2d(X_s_level2, Y_s_level2, x_span, x00, y00)
                ### Updated Laser Source ###
                
                ### New old mins for update ###
                yoldmin_s -= velocity
                yoldmin_lin -= velocity
                yoldmin_s_level2 -= velocity
                ### New old mins for update ###
                
                ### New min coords for interpolation ###
                yminn -=  velocity
                ymin_level2 -= velocity
                ### New min coords for interpolation ###

            
        ### Move Grid and update variables###
        elif not movement and iii and velocity: # means x-movement
        ### Move the Laser Source ###
            x00 += velocity
        ### Move the Laser Source ###
        
        ### Move Grid and update variables  ###

            move_gridx_all(velocity, iii)
            
            update_after_movementx2(x=x_s, y=d_y_s, z=d_z_s, val=d_u_s, nx_old=nx_s, ny_old=ny_s, nz_old=nz_s, hxval=h_x_new, hyval=h_y_new, hzval=h_z_new, xoldmin=xoldmin_s, yoldmin=yoldmin_s, zoldmin=zoldmin_s,  nx_in=nx_s_in, ny=ny_s, nz=nz_s, uin=uin_s, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  nx_out=nx_s_out, uout=d_uout_s, one=one, val3=u_s, slice_1d_in=slice_1d_inx, slice_1d_out=slice_1d_outx, index_x=index_x, threads_per_block_in=threads_per_block_in_s_x, blocks_per_grid_in=blocks_per_grid_in_s_x, threads_per_block_out=threads_per_block_out_s_x , blocks_per_grid_out=blocks_per_grid_out_s_x)
            update_after_movementx2(x=x_s_level2, y=d_y_s_level2, z=d_z_s_level2, val=d_u_s_level2, nx_old=nx_s_level2, ny_old=ny_s_level2, nz_old=nz_s_level2, hxval=h_x_new_level2, hyval=h_y_new_level2, hzval=h_z_new_level2, xoldmin=xoldmin_s_level2, yoldmin=yoldmin_s_level2, zoldmin=zoldmin_s_level2,  nx_in=nx_s_in_level2, ny=ny_s_level2, nz=nz_s_level2, uin=uin_s_level2, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  nx_out=nx_s_out_level2, uout=d_uout_s_level2, one=one, val3=u_s_level2, slice_1d_in=slice_1d_in_level2x, slice_1d_out=slice_1d_out_level2x, index_x=index_x_level2, threads_per_block_in=threads_per_block_in_s_level2_x, blocks_per_grid_in=blocks_per_grid_in_s_level2_x, threads_per_block_out=threads_per_block_out_s_level2_x , blocks_per_grid_out=blocks_per_grid_out_s_level2_x)
            update_after_movement_level3x_2(x=x_lin, y=d_y_lin, z=d_z_lin, val=d_u_lin, nx_old=nx_lin, ny_old=ny_lin, nz_old=nz_lin,  hxval=h_lin, hyval=h_lin, hzval=h_lin, xoldmin_lin=xoldmin_lin, yoldmin_lin=yoldmin_lin, zoldmin_lin=zoldmin_lin,  nx_in=nx_lin_in, ny=ny_lin,nz=nz_lin, uin=uin_lin,  slice_1d_in=slice_1d_in_linx, slice_1d_out=slice_1d_out_linx, index_x=index_x_lin, u0=u0 , one=one,  val3=u_lin, threads_per_block_in=threads_per_block_in_lin_x , blocks_per_grid_in=blocks_per_grid_in_lin_x)            
            

            ### Updated Laser Source ###
            qs_lin[:] = gaussian2d(X_lin, Y_lin, x_span, x00, y00)
            qs_s[:] = gaussian2d(X_s, Y_s, x_span, x00, y00)
            qs_s_level2[:] = gaussian2d(X_s_level2, Y_s_level2, x_span, x00, y00)
            ### Updated Laser Source ###
            
            ### New old mins for update ###
            xoldmin_s += velocity
            xoldmin_lin += velocity
            xoldmin_s_level2 += velocity
            ### New old mins for update ###
            
            ### New min coords for interpolation ###
            xminn +=  velocity
            xmin_level2 += velocity
            ### New min coords for interpolation ###
            
        ### Move Grid and update variables  ###
        elif  movement == 2 and iii and velocity: 
            ### Move the Laser Source ###
                x00 -= velocity
            ### Move the Laser Source ###
            
            ### Move Grid and update variables  ###
                move_gridx_all(-velocity, iii)
                
                update_after_movementx2_negative(x=x_s, y=d_y_s, z=d_z_s, val=d_u_s, nx_old=nx_s, ny_old=ny_s, nz_old=nz_s, hxval=h_x_new, hyval=h_y_new, hzval=h_z_new, xoldmin=xoldmin_s, yoldmin=yoldmin_s, zoldmin=zoldmin_s,  nx_in=nx_s_in, ny=ny_s, nz=nz_s, uin=uin_s, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  nx_out=nx_s_out, uout=d_uout_s, one=one, val3=u_s, slice_1d_in=slice_1d_inx_negative, slice_1d_out=slice_1d_outx_negative, index_x=0, threads_per_block_in=threads_per_block_in_s_x, blocks_per_grid_in=blocks_per_grid_in_s_x, threads_per_block_out=threads_per_block_out_s_x , blocks_per_grid_out=blocks_per_grid_out_s_x)
                update_after_movementx2_negative(x=x_s_level2, y=d_y_s_level2, z=d_z_s_level2, val=d_u_s_level2, nx_old=nx_s_level2, ny_old=ny_s_level2, nz_old=nz_s_level2, hxval=h_x_new_level2, hyval=h_y_new_level2, hzval=h_z_new_level2, xoldmin=xoldmin_s_level2, yoldmin=yoldmin_s_level2, zoldmin=zoldmin_s_level2,  nx_in=nx_s_in_level2, ny=ny_s_level2, nz=nz_s_level2, uin=uin_s_level2, val2=d_u_lin, nx_old2=nx_lin, ny_old2=ny_lin, nz_old2=nz_lin, hxval2=h_lin, hyval2=h_lin, hzval2=h_lin,  xold2min=xoldmin_lin, yold2min=yoldmin_lin, zold2min=zoldmin_lin,  nx_out=nx_s_out_level2, uout=d_uout_s_level2, one=one, val3=u_s_level2, slice_1d_in=slice_1d_in_level2x_negative, slice_1d_out=slice_1d_out_level2x_negative, index_x=0, threads_per_block_in=threads_per_block_in_s_level2_x, blocks_per_grid_in=blocks_per_grid_in_s_level2_x, threads_per_block_out=threads_per_block_out_s_level2_x , blocks_per_grid_out=blocks_per_grid_out_s_level2_x)
                update_after_movement_level3x_2_negative(x=x_lin, y=d_y_lin, z=d_z_lin, val=d_u_lin, nx_old=nx_lin, ny_old=ny_lin, nz_old=nz_lin,  hxval=h_lin, hyval=h_lin, hzval=h_lin, xoldmin_lin=xoldmin_lin, yoldmin_lin=yoldmin_lin, zoldmin_lin=zoldmin_lin,  nx_in=nx_lin_in, ny=ny_lin,nz=nz_lin, uin=uin_lin,  slice_1d_in=slice_1d_in_linx_negative, slice_1d_out=slice_1d_out_linx_negative, index_x=0, u0=u0 , one=one,  val3=u_lin, threads_per_block_in=threads_per_block_in_lin_x , blocks_per_grid_in=blocks_per_grid_in_lin_x)            

                ### Updated Laser Source ###
                qs_lin[:] = gaussian2d(X_lin, Y_lin, x_span, x00, y00)
                qs_s[:] = gaussian2d(X_s, Y_s, x_span, x00, y00)
                qs_s_level2[:] = gaussian2d(X_s_level2, Y_s_level2, x_span, x00, y00)
                ### Updated Laser Source ###
                
                ### New old mins for update ###
                xoldmin_s -= velocity
                xoldmin_lin -= velocity
                xoldmin_s_level2 -= velocity
                ### New old mins for update ###
                
                ### New min coords for interpolation ###
                xminn -=  velocity
                xmin_level2 -= velocity
                ### New min coords for interpolation ###


        
        u_s_old[:] = u_s.copy()
        u_s_level2_old[:] = u_s_level2.copy()

    

    
        
        ### Exract BC from previous time step for Level 2 ###
        trilinear_interpolation[blocks_per_grid_s_int1, threads_per_block_int1](d_x_s2_level2, d_y_s2_level2, d_z_s2_level2, d_u_lin, nx_lin, ny_lin, nz_lin, h_lin, h_lin, h_lin, xminn, yminn, zminn, nx_s2_level2, ny_s2_level2, nz_s2_level2, d_uinteold_level2, one)  
        extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc1, threads_per_block_bc1](d_uinteold_level2, nx_s_level2, nz_s_level2, ny_s_level2, d_p_r_old_level2, d_p_l_old_level2 ,  d_p_o_old_level2, d_p_i_old_level2, slice_bc_out0_1d_level2, slice_bc_in0_1d_level2, slice_bc_right0_1d_level2, slice_bc_left0_1d_level2,  nx_s2_level2)
        extract_neumann_bc_b[blocks_per_grid_s_bc2, threads_per_block_bc2](d_uinteold_level2, ny_s_level2, nx_s_level2, p_b_old_level2, slice_bc_bottom0_1d_level2, nx_s2_level2, ny_s2_level2)
        ### Exract BC from previous time step for Level 2 ###
    
        ### Solve the 3rd Level ###
        rhs_lin_dirichlet[blocks_per_grid_lin, threads_per_block_lin](u_lin, nx_lin, ny_lin, nz_lin, qs_lin, b_lin, h_linsq, h_linsq, h_linsq, n2, n3, dt_lin05, u0, h_lin, two, four)    
        u_new_lin,stat,num_iter = sparse_cg(Alinear_lin, b_lin, u_lin, 1e-5, None, maxit=1000)
        u_lin[:] = u_new_lin.copy()
        ### Solve the 3rd Level ###
        

        ### Exract BC from current time step for Level 2 ###
        trilinear_interpolation[blocks_per_grid_s_int1, threads_per_block_int1](d_x_s2_level2, d_y_s2_level2, d_z_s2_level2, d_u_lin, nx_lin, ny_lin, nz_lin, h_lin, h_lin, h_lin, xminn, yminn, zminn, nx_s2_level2, ny_s2_level2, nz_s2_level2, d_uinte_level2, one)
        extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc1, threads_per_block_bc1](d_uinte_level2, nx_s_level2, nz_s_level2, ny_s_level2, d_p_r_level2, d_p_l_level2, d_p_o_level2, d_p_i_level2, slice_bc_out0_1d_level2, slice_bc_in0_1d_level2, slice_bc_right0_1d_level2, slice_bc_left0_1d_level2,  nx_s2_level2)
        extract_neumann_bc_b[blocks_per_grid_s_bc2, threads_per_block_bc2](d_uinte_level2, ny_s_level2, nx_s_level2, d_p_b_level2, slice_bc_bottom0_1d_level2, nx_s2_level2, ny_s2_level2)
        ### Exract BC from curren time step for Level 2 ###
     
        ### Solve the 2nd Level ###
        rhs_neumann[blocks_per_grid_s_level2, threads_per_block_s_level2](u_s_level2_old, nx_s_level2, ny_s_level2, nz_s_level2, t_val,  qs_s_level2, b_s_level2, h_ix_newsq_level2, h_iy_newsq_level2, h_iz_newsq_level2, iSte, n2, n3, dt_lin05, p_o_level2, p_i_level2, p_r_level2, p_l_level2, p_b_level2, p_o_old_level2, p_i_old_level2, p_r_old_level2, p_l_old_level2, p_b_old_level2, u_s_level2, h_z_new_level2, two , half, four, pii)
        u_new_s_level2,stat,num_iter = sparse_cg(Alinear_s_level2, b_s_level2, u_s_level2, 1e-6, None, maxit=1000)
        ### Solve the 2nd Level ###



        ### MOVE ON TO 1ST LEVEL ###
        
        ### Exract BC from previous time step for Level 1 ###
        trilinear_interpolation[blocks_per_grid_s_int2, threads_per_block_int2](d_x_s2, d_y_s2, d_z_s2, d_u_s_level2_old, nx_s_level2, ny_s_level2, nz_s_level2, h_x_new_level2, h_y_new_level2, h_z_new_level2, xmin_level2, ymin_level2, zmin_level2, nx_s2, ny_s2, nz_s2, d_uinteold, one)  
        extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc3, threads_per_block_bc3](d_uinteold, nx_s, nz_s, ny_s, d_p_r_old, d_p_l_old ,  d_p_o_old, d_p_i_old, slice_bc_out0_1d, slice_bc_in0_1d, slice_bc_right0_1d, slice_bc_left0_1d,  nx_s2)
        extract_neumann_bc_b[blocks_per_grid_s_bc4, threads_per_block_bc4](d_uinteold, ny_s, nx_s, d_p_b_old, slice_bc_bottom0_1d, nx_s2, ny_s2)
        ### Exract BC from previous time step for Level 1 ###
        
        u_s_level2[:] = u_new_s_level2.copy()
        
        ### Exract BC from current time step for Level 1 ###
        trilinear_interpolation[blocks_per_grid_s_int2, threads_per_block_int2](d_x_s2, d_y_s2, d_z_s2, d_u_s_level2, nx_s_level2, ny_s_level2, nz_s_level2, h_x_new_level2, h_y_new_level2, h_z_new_level2, xmin_level2, ymin_level2, zmin_level2, nx_s2, ny_s2, nz_s2, d_uinte, one)
        extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc3, threads_per_block_bc3](d_uinte, nx_s, nz_s, ny_s, d_p_r, d_p_l, d_p_o, d_p_i, slice_bc_out0_1d, slice_bc_in0_1d, slice_bc_right0_1d, slice_bc_left0_1d,  nx_s2)
        extract_neumann_bc_b[blocks_per_grid_s_bc4, threads_per_block_bc4](d_uinte, ny_s, nx_s, d_p_b, slice_bc_bottom0_1d, nx_s2, ny_s2)
        ### Exract BC from current time step for Level 1 ###
    
        
        ### Solve the 1st Level ###
        for nl_lv1 in range(non_lin_it_lv1):
            rhs_neumann[blocks_per_grid_s, threads_per_block_s](d_u_s_old, nx_s, ny_s, nz_s, t_val,  d_qs_s, d_b_s, h_ix_newsq, h_iy_newsq, h_iz_newsq, iSte, n2, n3, dt_lin05, d_p_o, d_p_i, d_p_r, d_p_l, d_p_b, d_p_o_old, d_p_i_old, d_p_r_old, d_p_l_old, d_p_b_old, d_u_s, h_z_new, two, half, four, pii)
            u_new_s,stat,num_iter = sparse_cg(Alinear_s, b_s, u_s, 1e-6, None, maxit=1000)

            if nl_lv1 != nl_lv1-1:
                u_s[:] = (one-w_lv1)*u_s + w_lv1*u_new_s
            u_s[:] = u_new_s.copy()

        iii2 += 1
        
        if iii == 30000:
            print(index_y, 'was index_y')
            velocity *= 15
            print('velocity changed from ', velocity0, 'to ', velocity)
        if iii != 0 and iii % 1200 == 0:
            cntrr += 1
            if cntrr == 3:
                break
            if movement == 0:
                movement = 1
                
            elif cntrr==2:
                movement = 2
                negative = True
            
        if iii == 0:
            u_s_store = u_s.copy()
            u_s_level2_store = u_s_level2.copy()
            u_lin_store = u_lin.copy()
            
            
    if layers < num_layers-1:
        
        velocity = velocity0
        x00 = x00_initial
        y00 = y00_initial
        
        u_s = u_s_store.copy()
        u_s_level2 = u_s_level2_store.copy()
        u_lin = u_lin_store.copy()
        
        u_s3d = shape_back_3d(u_s, nx_s, ny_s, nz_s)
        u_s_level23d = shape_back_3d(u_s_level2, nx_s_level2, ny_s_level2, nz_s_level2)
        u_lin3d = shape_back_3d(u_lin, nx_lin, ny_lin, nz_lin)
        
        u_s3d_new[:, :, :-lay_ind_s] = u_s3d[:, :, lay_ind_s:]
        u_s_level23d_new[:, :, :-lay_ind_s_level2] = u_s_level23d[:, :, lay_ind_s_level2:]
        u_lin3d_new[:, :, :-lay_ind_lin] = u_lin3d[:, :, lay_ind_lin:]
        
        u_s3d_new[:, :, -lay_ind_s:] = u0
        u_s_level23d_new[:, :, -lay_ind_s_level2:] = u0
        u_lin3d_new[:, :, -lay_ind_lin:] = u0
        
        u_lin[:] = u_lin3d_new.ravel(order='F')
        u_s_level2[:] = u_s_level23d_new.ravel(order='F')
        u_s[:] = u_s3d_new.ravel(order='F')


        z_s +=  lt_sc 
        z_lin += lt_sc
        z_s_level2 += lt_sc
    


        z_max = float(z_s[-1])

        qs_lin *= 0 
        qs_s *= 0
        qs_s_level2 *= 0
        
        resety_all(y_s0, Y_s0, y_s20, y_s0_level2, Y_s0_level2, y_s20_level2, y_lin0, Y_lin0 )
        resetx_all(x_s0, X_s0, x_s20, x_s0_level2, X_s0_level2, x_s20_level2, x_lin0, X_lin0 )
        xminn = float_type(x_lin[0]); yminn = float_type(y_lin[0]); zminn = float_type(z_lin[0])
        xmin_level2 = float_type(x_s_level2[0]); ymin_level2 = float_type(y_s_level2[0]); zmin_level2 = float_type(z_s_level2[0])


        zoldmin_s = float_type(z_s[0].get())
        zoldmin_s_level2 = float_type(z_s_level2[0].get())
        zoldmin_lin = float_type(z_lin[0].get())

        ### Pre-Compute for update ###
        xminn = float_type(x_lin[0].get()); yminn = float_type(y_lin[0].get()); zminn = float_type(z_lin[0].get())
        xmin_level2 = float_type(x_s_level2[0].get()); ymin_level2 = float_type(y_s_level2[0].get()); zmin_level2 = float_type(z_s_level2[0].get())
        
        uin_s_level2 =  cp.zeros_like(u_s_level2[slice_1d_in_level2y], dtype = float_type)
        uout_s = cp.zeros_like(u_s[slice_1d_outy], dtype = float_type)
        uout_s_level2 = cp.zeros_like(u_s_level2[slice_1d_out_level2y], dtype = float_type)
        uin_lin =  cp.zeros_like(u_lin[slice_1d_in_liny], dtype = float_type)

        ny_lin_in = int(y_lin[0:index_y_lin].shape[0])
        ny_s_in  = int(y_s[0:index_y].shape[0])
        ny_s_out  = int(y_s[index_y:].shape[0])
        ny_s_in_level2  = int(y_s_level2[0:index_y_level2].shape[0])
        ny_s_out_level2  = int(y_s_level2[index_y_level2:].shape[0])

        nx_lin_in = int(x_lin[0:index_x_lin].shape[0])
        nx_s_in  = int(x_s[0:index_x].shape[0])
        nx_s_out  = int(x_s[index_x:].shape[0])
        nx_s_in_level2  = int(x_s_level2[0:index_x_level2].shape[0])
        nx_s_out_level2  = int(x_s_level2[index_x_level2:].shape[0])


        xoldmin_s = float_type(x_s[0].get())
        yoldmin_s = float_type(y_s[0].get())
        zoldmin_s = float_type(z_s[0].get())

        xoldmin_s_level2 = float_type(x_s_level2[0].get())
        yoldmin_s_level2 = float_type(y_s_level2[0].get())
        zoldmin_s_level2 = float_type(z_s_level2[0].get())

        xoldmin_lin = float_type(x_lin[0].get())
        yoldmin_lin = float_type(y_lin[0].get())
        zoldmin_lin = float_type(z_lin[0].get())
        
        ### Let it cool for balance ###
        for iii, t_val in enumerate(t_vals_lin):
    
            ttt += dt_lin*phys.time_scale*1e3
            
            u_s_level2_old[:] = u_s_level2.copy()
            u_s_old[:] = u_s.copy()
    
    
            
            ### Exract BC from previous time step for Level 2 ###
            trilinear_interpolation[blocks_per_grid_s_int1, threads_per_block_int1](x_s2_level2, y_s2_level2, z_s2_level2, u_lin, nx_lin, ny_lin, nz_lin, h_lin, h_lin, h_lin, xminn, yminn, zminn, nx_s2_level2, ny_s2_level2, nz_s2_level2, uinteold_level2, one)  
            extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc1, threads_per_block_bc1](uinteold_level2, nx_s_level2, nz_s_level2, ny_s_level2, p_r_old_level2, p_l_old_level2 ,  p_o_old_level2, p_i_old_level2, slice_bc_out0_1d_level2, slice_bc_in0_1d_level2, slice_bc_right0_1d_level2, slice_bc_left0_1d_level2,  nx_s2_level2)
            extract_neumann_bc_b[blocks_per_grid_s_bc2, threads_per_block_bc2](uinteold_level2, ny_s_level2, nx_s_level2, p_b_old_level2, slice_bc_bottom0_1d_level2, nx_s2_level2, ny_s2_level2)
            ### Exract BC from previous time step for Level 2 ###
        
            ### Solve the 3rd Level ###
            rhs_lin_dirichlet[blocks_per_grid_lin, threads_per_block_lin](d_u_lin, nx_lin, ny_lin, nz_lin, d_qs_lin, d_b_lin, h_linsq, h_linsq, h_linsq, n2, n3, dt_lin05, u0, h_lin, two, four)
            u_new_lin,stat,num_iter = sparse_cg(Alinear_lin, b_lin, u_lin, 1e-6, None, maxit=1000)
            u_lin[:] = u_new_lin.copy()
            ### Solve the 3rd Level ###
            
            ### Exract BC from current time step for Level 2 ###
            trilinear_interpolation[blocks_per_grid_s_int1, threads_per_block_int1](x_s2_level2, y_s2_level2, z_s2_level2, d_u_lin, nx_lin, ny_lin, nz_lin, h_lin, h_lin, h_lin, xminn, yminn, zminn, nx_s2_level2, ny_s2_level2, nz_s2_level2, d_uinte_level2, one)
            extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc1, threads_per_block_bc1](d_uinte_level2, nx_s_level2, nz_s_level2, ny_s_level2, d_p_r_level2, d_p_l_level2, d_p_o_level2, d_p_i_level2, slice_bc_out0_1d_level2, slice_bc_in0_1d_level2, slice_bc_right0_1d_level2, slice_bc_left0_1d_level2,  nx_s2_level2)
            extract_neumann_bc_b[blocks_per_grid_s_bc2, threads_per_block_bc2](d_uinte_level2, ny_s_level2, nx_s_level2, d_p_b_level2, slice_bc_bottom0_1d_level2, nx_s2_level2, ny_s2_level2)
            ### Exract BC from curren time step for Level 2 ###
         
            ### Solve the 2nd Level ###
            rhs_neumann[blocks_per_grid_s_level2, threads_per_block_s_level2](d_u_s_level2_old, nx_s_level2, ny_s_level2, nz_s_level2, t_val,  d_qs_s_level2, d_b_s_level2, h_ix_newsq_level2, h_iy_newsq_level2, h_iz_newsq_level2, iSte, n2, n3, dt_lin05, d_p_o_level2, d_p_i_level2, d_p_r_level2, d_p_l_level2, d_p_b_level2, d_p_o_old_level2, d_p_i_old_level2, d_p_r_old_level2, d_p_l_old_level2, d_p_b_old_level2, d_u_s_level2, h_z_new_level2, two , half, four, pii)
            u_new_s_level2,stat,num_iter = sparse_cg(Alinear_s_level2, b_s_level2, u_s_level2, 1e-6, None, maxit=1000)
            ### Solve the 2nd Level ###
            
            
            ### MOVE ON TO 1ST LEVEL ###
            
            ### Exract BC from previous time step for Level 1 ###
            trilinear_interpolation[blocks_per_grid_s_int2, threads_per_block_int2](x_s2, y_s2, z_s2, u_s_level2_old, nx_s_level2, ny_s_level2, nz_s_level2, h_x_new_level2, h_y_new_level2, h_z_new_level2, xmin_level2, ymin_level2, zmin_level2, nx_s2, ny_s2, nz_s2, uinteold, one)  
            extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc3, threads_per_block_bc3](uinteold, nx_s, nz_s, ny_s, p_r_old, p_l_old ,  p_o_old, p_i_old, slice_bc_out0_1d, slice_bc_in0_1d, slice_bc_right0_1d, slice_bc_left0_1d,  nx_s2)
            extract_neumann_bc_b[blocks_per_grid_s_bc4, threads_per_block_bc4](uinteold, ny_s, nx_s, p_b_old, slice_bc_bottom0_1d, nx_s2, ny_s2)
            ### Exract BC from previous time step for Level 1 ###
            
            u_s_level2[:] = u_new_s_level2.copy()
            
            ### Exract BC from current time step for Level 1 ###
            trilinear_interpolation[blocks_per_grid_s_int2, threads_per_block_int2](x_s2, y_s2, z_s2, d_u_s_level2, nx_s_level2, ny_s_level2, nz_s_level2, h_x_new_level2, h_y_new_level2, h_z_new_level2, xmin_level2, ymin_level2, zmin_level2, nx_s2, ny_s2, nz_s2, d_uinte, one)
            extract_neumann_bc_r_l_i_o[blocks_per_grid_s_bc3, threads_per_block_bc3](d_uinte, nx_s, nz_s, ny_s, d_p_r, d_p_l, d_p_o, d_p_i, slice_bc_out0_1d, slice_bc_in0_1d, slice_bc_right0_1d, slice_bc_left0_1d,  nx_s2)
            extract_neumann_bc_b[blocks_per_grid_s_bc4, threads_per_block_bc4](d_uinte, ny_s, nx_s, d_p_b, slice_bc_bottom0_1d, nx_s2, ny_s2)
            ### Exract BC from current time step for Level 1 ###
        
            
            ### Solve the 1st Level ###
            rhs_neumann[blocks_per_grid_s, threads_per_block_s](d_u_s_old, nx_s, ny_s, nz_s, t_val,  d_qs_s, d_b_s, h_ix_newsq, h_iy_newsq, h_iz_newsq, iSte, n2, n3, dt_lin05, d_p_o, d_p_i, d_p_r, d_p_l, d_p_b, d_p_o_old, d_p_i_old, d_p_r_old, d_p_l_old, d_p_b_old, d_u_s, h_z_new, two, half, four, pii)
            u_new_s,stat,num_iter = sparse_cg(Alinear_s, b_s, u_s, 1e-6, None, maxit=1000)
            u_s[:] = u_new_s.copy()          
            ### Solve the 1st Level ###
            
            
            if iii == 100:
                break     
    ### Let it cool for balance ###













