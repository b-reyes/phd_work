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

from train_1d_moving_ex_dist_part_tanh_delta_two_nets_dyn_time_pnts import PINN
from train_1d_moving_ex_dist_part_tanh_delta_two_nets_dyn_time_pnts import construct_interior_sampling_points
from train_1d_moving_ex_dist_part_tanh_delta_two_nets_dyn_time_pnts import exact_val
from train_1d_moving_ex_dist_part_tanh_delta_two_nets_dyn_time_pnts import exact_soln


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
k = 3.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
nu = 0.5 # Changed from 1.0 because of change in alpha formula below
c = 1.0
k_c = 1.0
acon = 0.25

# Parameters and relations:
# Connection of two constants nu and c above with m and h in Moshinky's paper:
alpha = c/(2.0*nu) # = h / (2 * m), so can think of c = h and m = nu
w = (k**2)*alpha

##############################################################################################
# Tp = total number of time points
# n = number of initial spatial points n = 5 (maybe make n even, i.e. n1 = 4), n = 10
# Tp_1 = Tp/2
# Tp_2 = Tp_1 + Tp/4
# Tp_3 = Tp_2 + Tp/4
#
# (0, Tp_1] --> n1 = n
# (Tp_1, Tp_2] --> n2 = n1 + floor(n/2)   # maybe play around with the floor(n/2), 2 value
# (Tp_2, Tp_3] --> n3 = n2 + floor(n/2)
##############################################################################################

# Note that psi_b = exp[i(kx - wt)] solves our considered Schrodinger's equation with F = 0
# Here "k" is the wavenumber and "w" is the angular frequency
#  \lambda = (2 * pi) / k is the wave
# particle momentum is " p = k*h = k*c"
# Energy is "E = w*h = w*c"
# If psi = \psi_0 * psi_b then \psi_0 is the (constant) amplitude of the wavefunction
# For test using the inhomogeneous equation, taking \psi_0 to be a function of space
# and this taken to ensure zero boundary condition.

# Network configuration
# k = 1
#u_layers = [2] + 4*[20] + [1]
#v_layers = [2] + 4*[20] + [1]

# k = 2
#u_layers = [2] + 4*[40] + [1]
#v_layers = [2] + 4*[40] + [1]

# k = 3
u_layers = [2] + 4*[60] + [1]
v_layers = [2] + 4*[60] + [1]

# k = 4
# u_layers = [2] + 4*[80] + [1]
# v_layers = [2] + 4*[80] + [1]

# k = 5
# u_layers = [2] + 4*[100] + [1]
# v_layers = [2] + 4*[100] + [1]

# n_int_sp = 30
# n_int_tm = 20000

n_int_sp = 1
n_int_tm = 10

N_f = 1280
M_f = 1280
XT_exact = exact_soln(T_max, N_f, M_f, acon, k, w, beta, k_c, PI)
XT_c = construct_interior_sampling_points(T_max, beta, n_int_tm, n_int_sp, acon)

with tf.device('/device:GPU:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Provide directory (second init) for pretrained networks if you have
    model = PINN(XT_exact, u_layers, v_layers, beta, acon, alpha, k, w, k_c,
                 #uDir='./uNN_switch_k_1_15k_5_500_pnts_tanh_delta_4_X_20_diff.pickle',
                 #vDir='./vNN_switch_k_1_15k_5_500_pnts_tanh_delta_4_X_20_diff.pickle')
                 #uDir='./uNN_switch_k_2_30k_10_1000_pnts_tanh_delta_4_X_40_diff.pickle',
                 #vDir='./vNN_switch_k_2_30k_10_1000_pnts_tanh_delta_4_X_40_diff.pickle')
                 #uDir='./uNN_switch_k_1_15k_5_500_pnts_tanh_delta_4_X_20.pickle',
                 #vDir='./vNN_switch_k_1_15k_5_500_pnts_tanh_delta_4_X_20.pickle')
                 #uDir='./uNN_switch_k_2_30k_10_1000_pnts_tanh_delta_4_X_40.pickle',
                 #vDir='./vNN_switch_k_2_30k_10_1000_pnts_tanh_delta_4_X_40.pickle')
                 uDir='./uNN_switch_k_3_60k_15_1500_pnts_tanh_delta_4_X_60.pickle',
                 vDir='./vNN_switch_k_3_60k_15_1500_pnts_tanh_delta_4_X_60.pickle')
                 

print("diff runs k  = 3") 
N_f = 1280
M_f = 1280

np_pred_soln_h = np.zeros((N_f + 1, M_f + 1))
Exact = np.zeros((N_f + 1, M_f + 1))

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
    # pred_u_part, pred_v_part, _, _, _, _ = model.predict_P(x_pnts, np.ones_like(x_pnts) * t_exact[i])

    np_pred_soln_h[:, i] = (pred_u**2 + pred_v**2).flatten()
    # np_pred_soln_h_part[:, i] = (pred_u_part ** 2 + pred_v_part ** 2).flatten()

    # rl_prt, img_prt = funk(x_pnts, acon*t_exact[i]**2, k)

    Exact_full = exact_val(x_pnts, np.ones_like(x_pnts)*t_exact[i], beta, acon, k, w, PI, k_c)

    rl_prt = Exact_full[:, 0:1]
    img_prt = Exact_full[:, 1:2]
    Exact[:, i] = (rl_prt**2 + img_prt**2).flatten()

    error_vec_u[i] = np.linalg.norm(pred_u - rl_prt) / np.linalg.norm(rl_prt)
    error_vec_v[i] = np.linalg.norm(pred_v - img_prt) / np.linalg.norm(img_prt)


file = open("relative_errors_k_3_60k_4_X_60_15_1500_dyn_time_pnts.txt", "w")
file.write('max relative error u = ' + str(np.amax(error_vec_u)) + '\n')
file.write('max relative error v = ' + str(np.amax(error_vec_v)) + '\n')
file.close()

print(error_vec_u)
print("")
print(error_vec_v)
print(f"max relative error u = {np.amax(error_vec_u)}")
print(f"max relative error v = {np.amax(error_vec_v)}")
# k = 1
# max relative error u = 0.0011697463150611472
# max relative error v = 0.000986866469200426


# max relative error u = 0.001493062102905764 dfdf
# max relative error v = 0.0008055724226955253


# k = 2
# max relative error u = 0.0007585069687347645
# max relative error v = 0.0010918295548114703

# k = 3
# max relative error u = 0.0009177224394282196
# max relative error v = 0.0010950023589681572

sys.exit()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0, 3.2)


def update_plot(i):
    ax.clear()
    ax.set_ylim(0, 3.2)
    abs_E_mat = np.linalg.norm(np_pred_soln_h[:, i] - Exact[:, i]) / np.linalg.norm(Exact[:, i])
    ax.set_xlabel("$x$", fontsize=15.0)
    ax.set_ylabel("$| h(x,t) |^2$", fontsize=15.0)
    ax.set_title("$ Relative Error = $" + f"{abs_E_mat:.4E} at t = {t_exact[i]:.4E}", fontsize=15.0)

    a_vec = beta + n*h_exact[i]

    ax.plot(a_vec, np_pred_soln_h[:, i], 'r-')
    # ax.plot(a_vec, np_pred_soln_h_part[:, i], 'g-')
    ax.plot(a_vec, Exact[:, i], 'b--')
    # ax.plot(a_vec, Exact_part[:, i], 'k--')
    return ax,

def ini():
    a_vec = beta + n*h_exact[0]

    ax.plot(a_vec, np_pred_soln_h[:, 0], 'r-')
    # ax.plot(a_vec, np_pred_soln_h_part[:, 0], 'g-')
    ax.plot(a_vec, Exact[:, 0], 'b--')
    # ax.plot(a_vec, Exact_part[:, 0], 'k--')
    return ax


ani = animation.FuncAnimation(fig, update_plot, frames=range(M_f+1), init_func=ini, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

ani.save('1d_schro_exact_soln_k_1.mp4', writer=writer)
