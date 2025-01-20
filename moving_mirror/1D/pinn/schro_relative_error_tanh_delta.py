import numpy as np
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy.io

from train_1d_moving_tanh_delta_dyn import PINN
from train_1d_moving_tanh_delta_dyn import construct_interior_sampling_points
from train_1d_moving_tanh_delta_dyn import exact_val
from train_1d_moving_tanh_delta_dyn import exact_soln


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
beta = -2.0
k = 5.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
nu = 0.5 # Changed from 1.0 because of change in alpha formula below
c = 1.0
k_c = 1.0
acon = 0.25

# Parameters and relations:
# Connection of two constants nu and c above with m and h in Moshinky's paper:
alpha = c/(2.0*nu) # = h / (2 * m), so can think of c = h and m = nu
w = (k**2)*alpha

# Network configuration
# k = 1
#nodes = 20
#hid_lay = 4
#iterations_lbfgs = 15000
#n_sp_str = '5_'
#n_tm_str = '500_'

# k = 2
#nodes = 40
#hid_lay = 4
#iterations_lbfgs = 30000
#n_sp_str = '10_'
#n_tm_str = '1000_'

# k = 3
#nodes = 60
#hid_lay = 4
#iterations_lbfgs = 60000
#n_sp_str = '15_'
#n_tm_str = '1500_'

# k = 4
#nodes = 80
#hid_lay = 4
#iterations_lbfgs = 120000
#n_sp_str = '20_'
#n_tm_str = '2000_'

# k = 5
nodes = 100
hid_lay = 4
iterations_lbfgs = 240000
n_sp_str = '25_'
n_tm_str = '2500_'

n_int_sp = 1
n_int_tm = 10

N_f = 10
M_f = 10 
XT_exact = exact_soln(T_max, N_f, M_f, acon, k, w, beta, k_c, PI)
XT_c = construct_interior_sampling_points(T_max, beta, n_int_tm, n_int_sp, acon)

u_layers = [2] + hid_lay*[nodes] + [1]
v_layers = [2] + hid_lay*[nodes] + [1]

path = './k_' + str(int(k)) + '/'

udir_name = 'uNN_k_' + str(int(k)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'

vdir_name = 'vNN_k_' + str(int(k)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str+ 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'

with tf.device('/device:GPU:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Provide directory (second init) for pretrained networks if you have
    model = PINN(XT_exact, u_layers, v_layers, beta, acon, alpha, k, w, k_c, iterations_lbfgs,
                 uDir= path + udir_name,
                 vDir= path + vdir_name)
                 

print(f"diff runs k  = {k}") 
N_f = 1280
M_f = 1280

n = np.linspace(0, N_f, N_f+1, dtype=int)
t_exact = np.linspace(0.0, T_max, M_f+1)

error_vec_u = np.zeros_like(t_exact)
error_vec_v = np.zeros_like(t_exact)

h_exact = np.zeros_like(t_exact)

for i in range(M_f+1):

    h_exact[i] = (acon*t_exact[i]**2 - beta)/N_f
    x_pnts = beta + n*h_exact[i]

    x_pnts = x_pnts.reshape((N_f + 1, 1))

    pred_u, pred_v = model.predict(x_pnts, np.ones_like(x_pnts)*t_exact[i])

    Exact_full = exact_val(x_pnts, np.ones_like(x_pnts)*t_exact[i], beta, acon, k, w, PI, k_c)

    rl_prt = Exact_full[:, 0:1]
    img_prt = Exact_full[:, 1:2]

    error_vec_u[i] = np.linalg.norm(pred_u - rl_prt) / np.linalg.norm(rl_prt)
    error_vec_v[i] = np.linalg.norm(pred_v - img_prt) / np.linalg.norm(img_prt)


rel_txt = 'relative_errors_k_' + str(int(k)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '_dyn.txt'

file = open(rel_txt, "w")
file.write('max relative error u = ' + str(np.amax(error_vec_u)) + '\n')
file.write('max relative error v = ' + str(np.amax(error_vec_v)) + '\n')
file.close()

print(f"max relative error u = {np.amax(error_vec_u)}")
print(f"max relative error v = {np.amax(error_vec_v)}")
