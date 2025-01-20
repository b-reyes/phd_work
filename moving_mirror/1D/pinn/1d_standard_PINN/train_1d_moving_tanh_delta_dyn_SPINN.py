import numpy as np
import time
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import pandas as pd
import shutil
import pickle
import math
import scipy.io
import sys

# Setup GPU for training
import tensorflow as tf
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)

# Note: TensorFlow 1.15 version is used
class PINN:
    # Initialize the class
    def __init__(self, XT_exact, u_layers, v_layers, beta, acon, alpha, k, w, k_c, iters, uDir='', vDir=''):

        # Count for callback function
        self.count = 0

        self.PI = math.pi
        self.k = k 

        # Bounds
        self.beta = beta
        self.acon = acon
        self.w = w
        self.beta = beta
        self.k_c = k_c
        self.alpha = alpha

        self.x_exact = XT_exact[:, 0:1]
        self.t_exact = XT_exact[:, 1:2]
        self.u_exact = XT_exact[:, 2:3]
        self.v_exact = XT_exact[:, 3:4]

        # Define layers config
        self.u_layers = u_layers
        self.v_layers = v_layers

        if uDir=='':
            self.u_weights, self.u_biases, self.u_multiplier = self.initialize_NN(self.u_layers)
        else:
            print("Loading u NN ...")
            self.u_weights, self.u_biases, self.u_multiplier = self.load_NN(uDir, self.u_layers)

        if vDir=='':
            self.v_weights, self.v_biases, self.v_multiplier = self.initialize_NN(self.v_layers)
        else:
            print("Loading v NN ...")
            self.v_weights, self.v_biases, self.v_multiplier = self.load_NN(vDir, self.v_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, 1])  # Point for postprocessing
        self.t_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_c_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_c_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_bcs_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_bcs_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_ic_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_ic_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.u_ic_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.v_ic_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.force_u_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.force_v_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_exact_tf = tf.placeholder(tf.float64, shape=[None, self.x_exact.shape[1]])
        self.t_exact_tf = tf.placeholder(tf.float64, shape=[None, self.t_exact.shape[1]])

        # predictions for the final output
        self.u_pred, self.v_pred, self.u_x_pred, self.v_x_pred = self.net_uv(self.x_tf, self.t_tf)

        self.u_pred_ex, self.v_pred_ex, _, _ = self.net_uv(self.x_exact_tf, self.t_exact_tf)

        self.loss_u_ex = tf.norm(self.u_pred_ex - self.u_exact)/tf.norm(self.u_exact)
        self.loss_v_ex = tf.norm(self.v_pred_ex - self.v_exact)/tf.norm(self.v_exact)

        # Governing eqn residual on collocation points
        self.f_pred_u, self.f_pred_v = self.net_f_sig(self.x_c_tf, self.t_c_tf)

        # predictions for the initial condition
        self.u_ic_pred, self.v_ic_pred, _, _ = self.net_uv(self.x_ic_tf, self.t_ic_tf)

        # predictions for the boundary conditions
        self.u_bcs_pred, self.v_bcs_pred, _, _ = self.net_uv(self.x_bcs_tf, self.t_bcs_tf)

        # loss for the initial and boundary conditions
        self.loss_ic_bcs = tf.reduce_mean(tf.square(self.u_ic_pred - self.u_ic_tf)) + \
                           tf.reduce_mean(tf.square(self.v_ic_pred - self.v_ic_tf)) + \
                           tf.reduce_mean(tf.square(self.u_bcs_pred)) + tf.reduce_mean(tf.square(self.v_bcs_pred))

        # Construct loss for the PDE
        self.loss_f_u = tf.reduce_mean(tf.square(self.f_pred_u - self.force_u_tf)) \
                        + tf.reduce_mean(tf.square(self.f_pred_v - self.force_v_tf))

        # full loss without scaling
        self.loss_full = self.loss_ic_bcs + self.loss_f_u

        self.loss = 1000 * (self.loss_full)

        # Optimizer for final solution (while the dist, particular network freezed)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.u_weights + self.u_biases + self.v_weights + self.v_biases + self.u_multiplier + self.v_multiplier,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': iters,
                                                                         'maxfun': iters,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 0.00001 * np.finfo(float).eps})

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []

        multiplier = [self.xavier_init(size=[1, 1])] 
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases, multiplier

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def save_NN(self, ufileDir, TYPE=''):
        if TYPE == 'U':
            u_weights = self.sess.run(self.u_weights)
            u_biases = self.sess.run(self.u_biases)
            u_multiplier = self.sess.run(self.u_multiplier)
        elif TYPE == 'V':
            u_weights = self.sess.run(self.v_weights)
            u_biases = self.sess.run(self.v_biases)
            u_multiplier = self.sess.run(self.v_multiplier)
        else: 
            pass

        with open(ufileDir, 'wb') as f:
            pickle.dump([u_weights, u_biases, u_multiplier], f)
            print("Save " + TYPE + " NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        multiplier = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases, uv_multiplier = pickle.load(f)
            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights) + 1)
            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float64)
                b = tf.Variable(uv_biases[num], dtype=tf.float64)
                weights.append(W)
                biases.append(b)
            
            mm = tf.Variable(uv_multiplier[0], dtype=tf.float64)
            multiplier.append(mm)
            print("Load NN parameters successfully...")
        return weights, biases, multiplier

    def neural_net(self, X, weights, biases, multiplier):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            # Creating a swish function
            delta = multiplier[0]
            xx = tf.add(tf.matmul(H, W), b)
            H = tf.tanh(delta*xx)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_uv(self, x, t):
        # Output for composite network        
        u = self.neural_net(tf.concat([x, t], 1), self.u_weights, self.u_biases, self.u_multiplier)
        v = self.neural_net(tf.concat([x, t], 1), self.v_weights, self.v_biases, self.v_multiplier)

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    def net_f_sig(self, x, t):

        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_u = u_t + self.alpha*v_xx
        f_v = v_t - self.alpha*u_xx

        return f_u, f_v    

    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train_bfgs(self, Collo, IC, BCS):

        x_c = Collo[:, 0:1]
        t_c = Collo[:, 1:2]

        x_ic = IC[:, 0:1]
        t_ic = IC[:, 1:2]
        u_ic = IC[:, 2:3]
        v_ic = IC[:, 3:4]

        x_bcs = BCS[:, 0:1]
        t_bcs = BCS[:, 1:2]

        out = self.forcing_funk(x_c, t_c)

        tf_dict = {self.x_c_tf: x_c, self.t_c_tf: t_c, self.x_ic_tf: x_ic, self.t_ic_tf: t_ic, self.u_ic_tf: u_ic,
                   self.v_ic_tf: v_ic, self.x_bcs_tf: x_bcs, self.t_bcs_tf: t_bcs, self.x_exact_tf: self.x_exact,
                   self.t_exact_tf: self.t_exact,
                   self.force_u_tf: out[:, 0:1], self.force_v_tf: out[:, 1:2]}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict) #,
                                #fetches=[self.loss],
                                #loss_callback=self.callback)

    def forcing_funk(self, x, t):

        force = -np.exp(1j*(-t*self.w + self.k*x))*(2.0*1j*self.k_c*self.PI*(self.alpha*(1j + self.k*(self.beta + self.acon*(t**2) - 2.0*x)) + self.acon*t*(-self.beta + x))*np.cos(self.k_c*self.PI*(self.beta-x)*(-self.acon*(t**2)+x)) + (-self.w+self.alpha*(self.k**2 + (self.k_c**2)*(self.PI**2)*(self.beta + self.acon*(t**2) - 2.0*x)**2))*np.sin(self.k_c*self.PI*(-self.beta+x)*(-self.acon*(t**2) + x)))

        out = np.zeros((x.shape[0], 2))
        out[:, 0] = np.imag(force).flatten()
        out[:, 1] = np.real(-force).flatten()

        return out

    def predict(self, x_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.t_tf: t_star})
        return u_star, v_star

    def getloss(self):
        tf_dict = {self.x_exact_tf: self.x_exact, self.t_exact_tf: self.t_exact}
        loss_value_u = self.sess.run(self.loss_u_ex, tf_dict)
        loss_value_v = self.sess.run(self.loss_v_ex, tf_dict)
        u_mult = self.sess.run(self.u_multiplier, tf_dict)
        v_mult = self.sess.run(self.v_multiplier, tf_dict)
        print('loss_value_u', loss_value_u)
        print('loss_value_v', loss_value_v)
        print('u_mult', u_mult)
        print('v_mult', v_mult)

    def GetFinalLoss(self, Collo, IC, BCS):

        x_c = Collo[:, 0:1]
        t_c = Collo[:, 1:2]

        x_ic = IC[:, 0:1]
        t_ic = IC[:, 1:2]
        u_ic = IC[:, 2:3]
        v_ic = IC[:, 3:4]

        x_bcs = BCS[:, 0:1]
        t_bcs = BCS[:, 1:2]

        out = self.forcing_funk(x_c, t_c)

        tf_dict = {self.x_c_tf: x_c, self.t_c_tf: t_c, self.x_ic_tf: x_ic, self.t_ic_tf: t_ic, self.u_ic_tf: u_ic,
                   self.v_ic_tf: v_ic, self.x_bcs_tf: x_bcs, self.t_bcs_tf: t_bcs, self.x_exact_tf: self.x_exact,
                   self.t_exact_tf: self.t_exact, self.force_u_tf: out[:, 0:1], self.force_v_tf: out[:, 1:2]}

        loss_value_u = self.sess.run(self.loss_u_ex, tf_dict)
        loss_value_v = self.sess.run(self.loss_v_ex, tf_dict)
        loss_f_u = self.sess.run(self.loss_f_u, tf_dict)
        u_mult = self.sess.run(self.u_multiplier, tf_dict)
        v_mult = self.sess.run(self.v_multiplier, tf_dict)

        print('loss_value_u', loss_value_u)
        print('loss_value_v', loss_value_v)
        print('loss_f_u', loss_f_u)
        print('u_mult', u_mult)
        print('v_mult', v_mult)

        return loss_value_u, loss_value_v, loss_f_u, u_mult, v_mult


def get_points(time_pnts, beta, nx, nt, acon):

    gamma = acon * time_pnts[0]**2

    # Domain bounds for x and t
    lb = np.array([beta])
    ub = np.array([gamma])

    tmp = (lb + (ub - lb) * lhs(1, nx))

    XT_c = np.concatenate((tmp, np.ones_like(tmp) * time_pnts[0]), 1)

    for i in range(1, nt):

        gamma = acon * time_pnts[i] ** 2

        # Domain bounds for x and t
        lb = np.array([beta])
        ub = np.array([gamma])

        tmp = (lb + (ub - lb) * lhs(1, nx))

        out = np.concatenate((tmp, np.ones_like(tmp) * time_pnts[i]), 1)
        XT_c = np.concatenate((XT_c, out), 0)

    return XT_c


def construct_interior_sampling_points(Max_T, beta, nt, nx, acon):

    r = nt % 6
    tp = nt // 6

    # [0.0, Max_T/3) --> 3*tp points
    # [Max_T/3, 2*(Max_T/3)] --> 2*tp points
    # (2*(Max_T/3), Max_T] --> tp + r points
    first_inter = np.linspace(0.0, Max_T/3, 3*tp + 1)[0:-1]
    second_inter = np.linspace(Max_T/3, 2*(Max_T/3), 2*tp)
    third_inter = np.linspace(2*(Max_T/3), Max_T, tp + r + 1)[1:]

    time_pnts = np.concatenate((first_inter, second_inter, third_inter), 0)

    # dt = Max_T/nt
    # time_pnts = dt*np.linspace(0, nt, nt+1, dtype=int)

    XT_c = get_points(time_pnts, beta, nx, nt, acon)

    return XT_c


def construct_initial_boundary_sampling_points(n_ic, n_bcs, beta, acon, k, w, pi, k_c, Max_T):

    # creating the initial condition points
    gamma = acon * (0**2)

    lb = np.array([beta])
    ub = np.array([gamma])

    tmp = (lb + (ub - lb) * lhs(1, n_ic))

    # getting initial condition values
    uv_ic = exact_val(tmp, np.zeros_like(tmp), beta, acon, k, w, pi, k_c)

    # putting all of the initial condition sampling points into an array
    IC_c = np.concatenate((tmp, np.zeros_like(tmp), uv_ic), 1)

    # creating the bc time points using LHS
    lb = np.array([0.0])
    ub = np.array([Max_T])
    t_bcs = (lb + (ub - lb) * lhs(1, n_bcs))

    gamma = acon * t_bcs[0]**2
    temp_bcs = np.array([[beta], [gamma[0]]])

    BCS_c = np.concatenate((temp_bcs, np.ones_like(temp_bcs) * t_bcs[0]), 1)

    for i in range(1, n_bcs):

        gamma = acon * t_bcs[i]**2
        temp_bcs = np.array([[beta], [gamma[0]]])

        out = np.concatenate((temp_bcs, np.ones_like(temp_bcs) * t_bcs[i]), 1)

        BCS_c = np.concatenate((BCS_c, out), 0)

    return IC_c, BCS_c


def exact_val(x, t, beta, acon, k, w, pi, k_c):

    ex = np.exp(1j*(k*x - w*t))*np.sin(k_c*pi*(x - acon*(t**2))*(x - beta))

    out = np.zeros((x.shape[0], 2))
    out[:, 0] = np.real(ex).flatten()
    out[:, 1] = np.imag(ex).flatten()

    return out


def gamma(tt, acon):
    return acon*(tt**2)


def exact_soln(T_max, N_f, M_f, acon, k, w, beta, k_c, pi):

    n = np.linspace(0, N_f, N_f + 1, dtype=int)

    t_exact = np.linspace(0, T_max, M_f + 1)

    h = (gamma(t_exact[0], acon) - beta)/N_f
    x_pnts = beta + n*h
    x_pnts = x_pnts.reshape((N_f + 1, 1))

    Exact = exact_val(x_pnts, np.ones_like(x_pnts)*t_exact[0], beta, acon, k, w, pi, k_c)

    XT_exact = np.concatenate((x_pnts, np.ones_like(x_pnts) * t_exact[0],
                               Exact[:, 0:1], Exact[:, 1:2]), 1)

    for i in range(1, t_exact.shape[0]):

        if i%1 == 0:

            h = (gamma(t_exact[i], acon) - beta) / N_f
            x_pnts = beta + n * h
            x_pnts = x_pnts.reshape((N_f + 1, 1))

            Exact = exact_val(x_pnts, np.ones_like(x_pnts)*t_exact[i], beta, acon, k, w, pi, k_c)

            out = np.concatenate((x_pnts, np.ones_like(x_pnts) * t_exact[i],
                                  Exact[:, 0:1], Exact[:, 1:2]), 1)

            XT_exact = np.concatenate((XT_exact, out), 0)

    return XT_exact


if __name__ == "__main__":

    PI = math.pi
    T_max = 1.5
    beta = -2.0
    k = 1.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
    nu = 0.5 # Changed from 1.0 because of change in alpha formula below
    c = 1.0
    k_c = 1.0
    acon = 0.25

    # Parameters and relations:
    # Connection of two constants nu and c above with m and h in Moshinky's paper:
    alpha = c/(2.0*nu) # = h / (2 * m), so can think of c = h and m = nu
    w = (k**2)*alpha

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
    nodes = 20
    hid_lay = 4
    iterations_lbfgs = 15000

    # k = 2
    #nodes = 40
    #hid_lay = 4

    # k = 3
    #nodes = 60
    #hid_lay = 4

    # k = 4
    #nodes = 80
    #hid_lay = 4

    # k = 1
    n_int_sp = 5
    n_int_tm = 500
    n_ic = 1000
    n_bcs = 1000

    # k = 2
    #n_int_sp = 10
    #n_int_tm = 1000

    # k = 3
    #n_int_sp = 15
    #n_int_tm = 1500

    # k = 4
    #n_int_sp = 20
    #n_int_tm = 2000

    u_layers = [2] + hid_lay * [nodes] + [1]
    v_layers = [2] + hid_lay * [nodes] + [1]

    N_f = 1280
    M_f = 1280
    XT_exact = exact_soln(T_max, N_f, M_f, acon, k, w, beta, k_c, PI)
    XT_c = construct_interior_sampling_points(T_max, beta, n_int_tm, n_int_sp, acon)

    IC_c, BCS_c = construct_initial_boundary_sampling_points(n_ic, n_bcs, beta, acon, k, w, PI, k_c, T_max)

    iterstr = "_" + str(iterations_lbfgs) + "_"
    activ_str = "_tanh_delta_"

    pathu = "uNN_k_" + str(k)[0] + iterstr + str(n_int_sp) + "_" + str(n_int_tm) + "_pnts" + activ_str + str(hid_lay) + "_X_" + str(nodes) + ".pickle"
    pathv = "vNN_k_" + str(k)[0] + iterstr + str(n_int_sp) + "_" + str(n_int_tm) + "_pnts" + activ_str + str(hid_lay) + "_X_" + str(nodes) + ".pickle"

    path_loss = "loss_k_" + str(k)[0] + iterstr + str(n_int_sp) + "_" + str(n_int_tm) + "_pnts" + activ_str + str(hid_lay) + "_X_" + str(nodes) + ".txt"

    with tf.device('/device:GPU:0'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XT_exact, u_layers, v_layers, beta, acon, alpha, k, w, k_c, iterations_lbfgs)

        print(activ_str)
        print(iterstr)
        print(f"u_layers = {u_layers}")
        print(f"v_layers = {v_layers}")
        print(f"spatial = {n_int_sp}")
        print(f"time pnts = {n_int_tm}")
        print(f"k = {k}")
        # Train the composite network
        start_time = time.time()
        model.train_bfgs(XT_c, IC_c, BCS_c)
        elapsed = time.time() - start_time
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save the trained model
        model.save_NN(pathu, TYPE='U')
        model.save_NN(pathv, TYPE='V')

        # Check the loss for each part
        model.getloss()

        loss_value_u, loss_value_v, loss_f_u, u_mult, v_mult = model.GetFinalLoss(XT_c, IC_c, BCS_c)

        file = open(path_loss, "w")
        file.write('loss_value_u= ' + str(loss_value_u) + '\n')
        file.write('loss_value_v = ' + str(loss_value_v) + '\n')
        file.write('loss_f_u = ' + str(loss_f_u) + '\n')
        file.write('u_mult = ' + str(u_mult) + '\n')
        file.write('v_mult = ' + str(v_mult) + '\n')
        file.write('elapsed time = ' + str(elapsed) + '\n')
        file.close()
