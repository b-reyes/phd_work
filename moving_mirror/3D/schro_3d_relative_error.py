import numpy as np
import time
from pyDOE import lhs
import scipy.io

from train_3d_moving_tanh_delta import PINN
from train_3d_moving_tanh_delta import construct_interior_sampling_points
from train_3d_moving_tanh_delta import exact_val
from train_3d_moving_tanh_delta import exact_soln

# Setup GPU for training
import tensorflow as tf
import os
import pickle
import math
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)

PI = math.pi
T_max = 1.5
betax = -2.0
betay = -2.0
betaz = -2.0
kx = 1.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
ky = kx
kz = kx
nu = 0.5 # Changed from 1.0 because of change in alpha formula below
c = 1.0
k_c = 1.0
aconx = 0.25
acony = 0.25
aconz = 0.25

# Parameters and relations:
# Connection of two constants nu and c above with m and h in Moshinky's paper:
alpha = c/(2.0*nu) # = h / (2 * m), so can think of c = h and m = nu
w = (kx**2)*alpha

# Note that psi_b = exp[i(kx - wt)] solves our considered Schrodinger's equation with F = 0
# Here "k" is the wavenumber and "w" is the angular frequency
#  \lambda = (2 * pi) / k is the wave
# particle momentum is " p = k*h = k*c"
# Energy is "E = w*h = w*c"
# If psi = \psi_0 * psi_b then \psi_0 is the (constant) amplitude of the wavefunction
# For test using the inhomogeneous equation, taking \psi_0 to be a function of space
# and this taken to ensure zero boundary condition.

# k = 1
nodes = 60
hid_lay = 4
iterations_lbfgs = 375000
n_sp_str = '25_'
n_tm_str = '2500_'

n_int_sp = 1
n_int_tm = 10

N_fx = 2 
N_fy = 2 
N_fz = 2 
M_f = 2 
XYZT_exact = exact_soln(T_max, N_fx, N_fy, N_fz, M_f, aconx, acony, aconz, kx, ky, kz, w, betax, betay, betay, k_c, PI)

XYZT_c = construct_interior_sampling_points(T_max, betax, betay, betaz, n_int_tm, n_int_sp, aconx, acony, aconz)

u_layers = [4] + hid_lay*[nodes] + [1]
v_layers = [4] + hid_lay*[nodes] + [1]

path = './k_' + str(int(kx)) + '_final/'

udir_name = 'uNN_3d_k_' + str(int(kx)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'

vdir_name = 'vNN_3d_k_' + str(int(kx)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'


with tf.device('/device:GPU:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Provide directory (second init) for pretrained networks if you have
    model = PINN(XYZT_exact, u_layers, v_layers, betax, betay, betaz, aconx, acony, aconz, alpha, kx, ky, kz, w, k_c, iterations_lbfgs,
                 uDir=path + udir_name,
                 vDir=path + vdir_name)
    
N_f_x = 200
N_f_y = 200
N_f_z = 200
M_f = 200

print(f"m_f = {M_f}")

n_x = np.linspace(0, N_f_x, N_f_x+1, dtype=int)
n_y = np.linspace(0, N_f_y, N_f_y+1, dtype=int)
n_z = np.linspace(0, N_f_z, N_f_z+1, dtype=int)
t_exact = np.linspace(0.0, T_max, M_f+1)

error_vec_u = np.zeros_like(t_exact)
error_vec_v = np.zeros_like(t_exact)

for i in range(M_f+1):

    print(f"iteration = {i}")

    h_exact_x = (aconx*t_exact[i]**2 - betax)/N_f_x
    h_exact_y = (acony * t_exact[i] ** 2 - betay)/N_f_y
    h_exact_z = (aconz * t_exact[i] ** 2 - betaz)/N_f_z
    x_pnts = betax + n_x*h_exact_x 
    y_pnts = betay + n_y*h_exact_y 
    z_pnts = betaz + n_z * h_exact_z 

    x_exact, y_exact, z_exact = np.meshgrid(x_pnts, y_pnts, z_pnts)

    net_x = x_exact.reshape(((N_f_x+1)*(N_f_y+1)*(N_f_z+1), 1))
    net_y = y_exact.reshape(((N_f_x + 1) * (N_f_y + 1)*(N_f_z+1), 1))
    net_z = z_exact.reshape(((N_f_x + 1) * (N_f_y + 1) * (N_f_z + 1), 1))

    pred_u, pred_v = model.predict(net_x, net_y, net_z, np.ones_like(net_x)*t_exact[i])

    Exact_full = exact_val(x_exact, y_exact, z_exact, np.ones_like(x_exact)*t_exact[i], betax, betay, betaz, aconx,
                           acony, aconz, kx, ky, kz, w, PI, k_c)

    rl_prt = Exact_full[:, 0:1].reshape((N_f_x+1, N_f_y+1, N_f_z+1))
    img_prt = Exact_full[:, 1:2].reshape((N_f_x+1, N_f_y+1, N_f_z+1))

    error_vec_u[i] = np.linalg.norm(pred_u.reshape((N_f_x+1, N_f_y+1, N_f_z+1)) - rl_prt) / np.linalg.norm(rl_prt)
    error_vec_v[i] = np.linalg.norm(pred_v.reshape((N_f_x+1, N_f_y+1, N_f_z+1)) - img_prt) / np.linalg.norm(img_prt)


rel_txt = 'relative_errors_3d_k_' + str(int(kx)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '_dyn.txt'

file = open(rel_txt, "w")
file.write('max relative error u = ' + str(np.amax(error_vec_u)) + '\n')
file.write('max relative error v = ' + str(np.amax(error_vec_v)) + '\n')
file.close()

print(f"max relative error u = {np.amax(error_vec_u)}")
print(f"max relative error v = {np.amax(error_vec_v)}")

