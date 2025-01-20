import numpy as np
from scipy.io import savemat

from train_2d_moving_tanh_delta_dyn import PINN
from train_2d_moving_tanh_delta_dyn import construct_interior_sampling_points
from train_2d_moving_tanh_delta_dyn import exact_val
from train_2d_moving_tanh_delta_dyn import exact_soln

# Setup GPU for training
import tensorflow as tf
import math

np.random.seed(1111)
tf.set_random_seed(1111)


PI = math.pi
T_max = 1.5
betax = -2.0
betay = -2.0
kx = 3.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
ky = kx
nu = 0.5 # Changed from 1.0 because of change in alpha formula below
c = 1.0
k_c = 1.0
aconx = 0.25
acony = 0.25

# Parameters and relations:
# Connection of two constants nu and c above with m and h in Moshinky's paper:
alpha = c/(2.0*nu) # = h / (2 * m), so can think of c = h and m = nu
w = (kx**2)*alpha

# k = 1
# nodes = 40
# hid_lay = 4
# iterations_lbfgs = 75000
# n_sp_str = '10_'
# n_tm_str = '1000_'

# k = 2
# nodes = 60
# hid_lay = 4
# iterations_lbfgs = 150000
# n_sp_str = '15_'
# n_tm_str = '1500_'

# k = 3
nodes = 80
hid_lay = 4
iterations_lbfgs = 300000
n_sp_str = '20_'
n_tm_str = '2000_'

n_int_sp = 1
n_int_tm = 10 

N_fx = 10 
N_fy = 10 
M_f = 10 
XYT_exact = exact_soln(T_max, N_fx, N_fy, M_f, aconx, acony, kx, ky, w, betax, betay, k_c, PI)
XYT_c = construct_interior_sampling_points(T_max, betax, betay, n_int_tm, n_int_sp, aconx, acony)

u_layers = [2] + hid_lay*[nodes] + [1]
v_layers = [2] + hid_lay*[nodes] + [1]

path = './results_2d/'

udir_name = 'uNN_2d_k_' + str(int(kx)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'

vdir_name = 'vNN_2d_k_' + str(int(kx)) + '_' + str(iterations_lbfgs) + '_' + n_sp_str + n_tm_str + 'pnts_tanh_delta_' + str(hid_lay) + '_X_' + str(nodes) + '.pickle'


with tf.device('/device:GPU:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Provide directory (second init) for pretrained networks if you have
    model = PINN(XYT_exact, u_layers, v_layers, betax, betay, aconx, acony, alpha, kx, ky, w, k_c,
                 iterations_lbfgs, uDir=path + udir_name, vDir=path + vdir_name)

print(f"k = {kx}")

N_f_x = 160 #1280
N_f_y = 160 #1280
M_f = 160 #1280

n_x = np.linspace(0, N_f_x, N_f_x+1, dtype=int)
n_y = np.linspace(0, N_f_y, N_f_y+1, dtype=int)
t_exact = np.linspace(0.0, T_max, M_f+1)
nn_pred_u = np.zeros((N_f_x + 1, N_f_y + 1, M_f+1))
nn_pred_v = np.zeros((N_f_x + 1, N_f_y + 1, M_f+1))

all_xpnts = np.zeros((N_f_x + 1, N_f_y + 1, M_f+1))
all_ypnts = np.zeros((N_f_x + 1, N_f_y + 1, M_f+1))

times = np.zeros((M_f+1, 1))

for i in range(M_f+1):
    times[i] = t_exact[i]
    h_exact_x = (aconx*t_exact[i]**2 - betax)/N_f_x
    h_exact_y = (acony * t_exact[i] ** 2 - betay) / N_f_y
    x_pnts = betax + n_x*h_exact_x
    y_pnts = betay + n_y*h_exact_y
    x_exact, y_exact = np.meshgrid(x_pnts, y_pnts)
    x_pnts1 = x_exact
    y_pnts1 = y_exact
    all_xpnts[:, :, i] = x_exact
    all_ypnts[:, :, i] = y_exact
    net_x = x_exact.reshape(((N_f_x+1)*(N_f_y+1), 1))
    net_y = y_exact.reshape(((N_f_x + 1) * (N_f_y + 1), 1))
    pred_u, pred_v = model.predict(net_x, net_y, np.ones_like(net_x)*t_exact[i])
    nn_pred_u[:, :, i] = pred_u.reshape((N_f_x+1, N_f_y+1))
    nn_pred_v[:, :, i] = pred_v.reshape((N_f_x+1, N_f_y+1))

mdic = {"nn_pred_u": nn_pred_u, "nn_pred_v": nn_pred_v, "all_xpnts": all_xpnts, "all_ypnts": all_ypnts, "times": times}

savemat("2d_matlab_movie_data_k_" + str(int(kx)) + ".mat", mdic)
