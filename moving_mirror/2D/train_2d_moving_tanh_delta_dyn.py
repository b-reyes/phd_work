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
    def __init__(self, XYT_exact, u_layers, v_layers, betax, betay, aconx, acony, alpha, kx, ky, w, k_c, iters, uDir='', vDir=''):

        # Count for callback function
        self.count = 0

        self.PI = math.pi
        self.kx = kx
        self.ky = ky

        # Bounds
        self.betax = betax
        self.betay = betay
        self.aconx = aconx
        self.acony = acony
        self.w = w
        self.k_c = k_c
        self.alpha = alpha

        self.x_exact = XYT_exact[:, 0:1]
        self.y_exact = XYT_exact[:, 1:2]
        self.t_exact = XYT_exact[:, 2:3]
        self.u_exact = XYT_exact[:, 3:4]
        self.v_exact = XYT_exact[:, 4:5]

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
        self.x_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_c_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.y_c_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_c_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.force_u_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.force_v_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.x_exact_tf = tf.placeholder(tf.float64, shape=[None, self.x_exact.shape[1]])
        self.y_exact_tf = tf.placeholder(tf.float64, shape=[None, self.y_exact.shape[1]])
        self.t_exact_tf = tf.placeholder(tf.float64, shape=[None, self.t_exact.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred = self.net_uv(self.x_tf, self.y_tf, self.t_tf)

        self.u_pred_ex, self.v_pred_ex = self.net_uv(self.x_exact_tf, self.y_exact_tf, self.t_exact_tf)

        self.loss_u_ex = tf.norm(self.u_pred_ex - self.u_exact)/tf.norm(self.u_exact)
        self.loss_v_ex = tf.norm(self.v_pred_ex - self.v_exact)/tf.norm(self.v_exact)

        # Governing eqn residual on collocation points
        self.f_pred_u, self.f_pred_v = self.net_f_sig(self.x_c_tf, self.y_c_tf, self.t_c_tf)

        # Construct loss to optimize
        self.loss_f_u = tf.reduce_mean(tf.square(self.f_pred_u - self.force_u_tf)) \
                        + tf.reduce_mean(tf.square(self.f_pred_v - self.force_v_tf))

        self.loss = 1000 * (self.loss_f_u)

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
        multiplier = [self.xavier_init(size=[1, 1]) ]
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
            delta = multiplier[0]
            xx = tf.add(tf.matmul(H, W), b)
            H = tf.tanh(delta*xx)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_uv(self, x, y, t):
        # Output for composite network
        
        u = self.neural_net(tf.concat([x, y, t], 1), self.u_weights, self.u_biases, self.u_multiplier)
        v = self.neural_net(tf.concat([x, y, t], 1), self.v_weights, self.v_biases, self.v_multiplier)

        D_uv = t*(x - self.betax)*(self.aconx*(t**2) - x)*(y - self.betay)*(self.acony*(t**2) - y)

        mu = tf.sin(self.k_c*self.PI*(x - self.betax)*(x - self.aconx*(t**2))*(y - self.betay)*(y - self.acony*(t**2)))

        P_u = tf.cos(self.kx*x + self.ky*y)*mu
        P_v = tf.sin(self.kx*x + self.ky*y)*mu

        #####p(x,y)+D(x,y)*u(x,y)######
        u = P_u + D_uv*u
        v = P_v + D_uv*v

        return u, v

    def net_f_sig(self, x, y, t):

        u, v = self.net_uv(x, y, t)

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        f_u = u_t + self.alpha*(v_xx + v_yy)
        f_v = v_t - self.alpha*(u_xx + u_yy)

        return f_u, f_v    

    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train_bfgs(self, Collo):

        x_c = Collo[:, 0:1]
        y_c = Collo[:, 1:2]
        t_c = Collo[:, 2:3]

        out = self.forcing_funk(x_c, y_c, t_c)

        tf_dict = {self.x_c_tf: x_c, self.y_c_tf: y_c, self.t_c_tf: t_c, self.x_exact_tf: self.x_exact,
                   self.y_exact_tf: self.y_exact, self.t_exact_tf: self.t_exact, self.force_u_tf: out[:, 0:1],
                   self.force_v_tf: out[:, 1:2]}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict) #,
                                #fetches=[self.loss],
                                #loss_callback=self.callback)

    def forcing_funk(self, x, y, t):

        force = 1j*(np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*(-2.0*self.acony*self.k_c*self.PI*t*(-self.betax+x)*(-self.aconx*(t**2) + x)*(-self.betay + y) - 2.0*self.aconx*self.k_c*self.PI*t*(-self.betax + x)*(-self.betay + y)*(-self.acony*(t**2) + y))*np.cos(self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.betay + y)*(-self.acony*(t**2) + y)) - 1j*np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*self.w*np.sin(self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.betay + y)*(-self.acony*(t**2) + y))) + self.alpha*(2.0*1j*np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*self.ky*(self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.betay + y) + self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.acony*(t**2) + y))*np.cos(self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.betay + y)*(-self.acony*(t**2) + y)) + 2.0*1j*np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*self.kx*(self.k_c*self.PI*(-self.betax + x)*(-self.betay + y)*(-self.acony*(t**2) + y) + self.k_c*self.PI*(-self.aconx*(t**2) + x)*(-self.betay + y)*(-self.acony*(t**2) + y))*np.cos(self.k_c*self.PI*(-self.betax + x)*(-self.aconx*(t**2) + x)*(-self.betay + y)*(-self.acony*(t**2) + y)) - np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*(self.kx**2)*np.sin(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y)) - np.exp(1j*(-t*self.w + self.kx*x + self.ky*y))*(ky**2)*np.sin(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y)) + np.exp(1j*(-t*self.w + self.kx*x+ self.ky*y))*(2.0*self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*np.cos(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y)) - (self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y) + self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.acony*(t**2)+y))**2*np.sin(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y))) + np.exp(1j*(-t*self.w + self.kx*x+ self.ky*y))*(2.0*self.k_c*self.PI*(-self.betay+y)*(-self.acony*(t**2)+y)*np.cos(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y)) - (self.k_c*self.PI*(-self.betax+x)*(-self.betay+y)*(-self.acony*(t**2)+y) + self.k_c*self.PI*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y))**2*np.sin(self.k_c*self.PI*(-self.betax+x)*(-self.aconx*(t**2)+x)*(-self.betay+y)*(-self.acony*(t**2)+y))))
        out = np.zeros((x.shape[0], 2))
        out[:, 0] = np.imag(force).flatten()
        out[:, 1] = np.real(-force).flatten()

        return out

    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        return u_star, v_star

    def getloss(self):
        tf_dict = {self.x_exact_tf: self.x_exact, self.y_exact_tf: self.y_exact, self.t_exact_tf: self.t_exact}
        loss_value_u = self.sess.run(self.loss_u_ex, tf_dict)
        loss_value_v = self.sess.run(self.loss_v_ex, tf_dict)
        print('loss_value_u', loss_value_u)
        print('loss_value_v', loss_value_v)
        
    def GetFinalLoss(self, Collo):

        x_c = Collo[:, 0:1]
        y_c = Collo[:, 1:2]
        t_c = Collo[:, 2:3]

        out = self.forcing_funk(x_c, y_c, t_c)

        tf_dict = {self.x_c_tf: x_c, self.y_c_tf: y_c, self.t_c_tf: t_c, self.x_exact_tf: self.x_exact,
                   self.y_exact_tf: self.y_exact, self.t_exact_tf: self.t_exact, self.force_u_tf: out[:, 0:1],
                   self.force_v_tf: out[:, 1:2]}

        loss_value_u = self.sess.run(self.loss_u_ex, tf_dict)
        loss_value_v = self.sess.run(self.loss_v_ex, tf_dict)
        u_mult = self.sess.run(self.u_multiplier, tf_dict)
        v_mult = self.sess.run(self.v_multiplier, tf_dict)
        loss_f_u = self.sess.run(self.loss_f_u, tf_dict)

        return loss_value_u, loss_value_v, loss_f_u, u_mult, v_mult


def get_points(time_pnts, betax, betay, n_sp, nt, aconx, acony):

    gammax = aconx * time_pnts[0]**2
    gammay = acony * time_pnts[0]**2

    # Domain bounds for x and t
    lb = np.array([betax, betay])
    ub = np.array([gammax, gammay])

    tmp = (lb + (ub - lb) * lhs(2, n_sp))

    XYT_c = np.concatenate((tmp, np.ones_like(tmp) * time_pnts[0]), 1)

    for i in range(1, nt):

        gammax = aconx * time_pnts[i] ** 2
        gammay = acony * time_pnts[i] ** 2

        # Domain bounds for x and t
        lb = np.array([betax, betay])
        ub = np.array([gammax, gammay])

        tmp = (lb + (ub - lb) * lhs(2, n_sp))

        out = np.concatenate((tmp, np.ones_like(tmp) * time_pnts[i]), 1)
        XYT_c = np.concatenate((XYT_c, out), 0)

    return XYT_c


def construct_interior_sampling_points(Max_T, betax, betay, nt, n_sp, aconx, acony):

    # dt = Max_T/nt
    # time_pnts = dt*np.linspace(0, nt, nt+1, dtype=int)

    r = nt % 6
    tp = nt // 6

    # [0.0, Max_T/3) --> 3*tp points
    # [Max_T/3, 2*(Max_T/3)] --> 2*tp points
    # (2*(Max_T/3), Max_T] --> tp + r points
    first_inter = np.linspace(0.0, Max_T/3, 3*tp + 1)[0:-1]
    second_inter = np.linspace(Max_T/3, 2*(Max_T/3), 2*tp)
    third_inter = np.linspace(2*(Max_T/3), Max_T, tp + r + 1)[1:]

    time_pnts = np.concatenate((first_inter, second_inter, third_inter), 0)

    XYT_c = get_points(time_pnts, betax, betay, n_sp, nt, aconx, acony)

    return XYT_c


def exact_val(x, y, t, betax, betay, aconx, acony, kx, ky, w, pi, k_c):

    m = x.shape[0]
    n = x.shape[1]

    ex = np.exp(1j*(kx*x + ky*y - w*t))*np.sin(k_c*pi*(x - aconx*(t**2))*(x - betax)*(y - acony*(t**2))*(y - betay))

    rl_part = np.real(ex).reshape((m * n, 1))
    img_part = np.imag(ex).reshape((m * n, 1))

    out = np.zeros((m * n, 2))
    out[:, 0] = rl_part.flatten()
    out[:, 1] = img_part.flatten()

    return out


def gamma(tt, acon):
    return acon*(tt**2)


def exact_soln(T_max, N_fx, N_fy, M_f, aconx, acony, kx, ky, w, betax, betay, k_c, pi):

    n_x = np.linspace(0, N_fx, N_fx + 1, dtype=int)
    n_y = np.linspace(0, N_fy, N_fy + 1, dtype=int)

    t_exact = np.linspace(0, T_max, M_f + 1)

    h_exact_x = (aconx * t_exact[0] ** 2 - betax) / N_fx
    h_exact_y = (acony * t_exact[0] ** 2 - betay) / N_fy
    x_pnts = betax + n_x * h_exact_x
    y_pnts = betay + n_y * h_exact_y

    x_exact, y_exact = np.meshgrid(x_pnts, y_pnts)

    Exact = exact_val(x_exact, y_exact, np.ones_like(x_exact)*t_exact[0], betax, betay, aconx, acony, kx, ky, w, pi, k_c)

    XYT_exact = np.concatenate((x_exact.reshape(((N_fx + 1) * (N_fy + 1), 1)),
                               y_exact.reshape(((N_fx + 1) * (N_fy + 1), 1)),
                               np.ones(((N_fx + 1) * (N_fy + 1), 1)) * t_exact[0],
                               Exact[:, 0:1], Exact[:, 1:2]), 1)

    for i in range(1, t_exact.shape[0]):

        if i%1 == 0:

            h_exact_x = (aconx * t_exact[i] ** 2 - betax) / N_fx
            h_exact_y = (acony * t_exact[i] ** 2 - betay) / N_fy
            x_pnts = betax + n_x * h_exact_x
            y_pnts = betay + n_y * h_exact_y

            x_exact, y_exact = np.meshgrid(x_pnts, y_pnts)

            Exact = exact_val(x_exact, y_exact, np.ones_like(x_exact) * t_exact[i], betax, betay, aconx, acony, kx, ky,
                              w, pi, k_c)

            out = np.concatenate((x_exact.reshape(((N_fx + 1) * (N_fy + 1), 1)),
                                       y_exact.reshape(((N_fx + 1) * (N_fy + 1), 1)),
                                       np.ones(((N_fx + 1) * (N_fy + 1), 1)) * t_exact[i],
                                       Exact[:, 0:1], Exact[:, 1:2]), 1)

            XYT_exact = np.concatenate((XYT_exact, out), 0)

    return XYT_exact


if __name__ == "__main__":

    PI = math.pi
    T_max = 1.5
    betax = -2.0
    betay = -2.0
    kx = 4.0 # Take k = 1, 2, 3, ...., 10(as large as possible to test samples)
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
    #nodes = 40
    #hid_lay = 4
    #iterations_lbfgs = 75000

    # k = 2
    #nodes = 60
    #hid_lay = 4
    #iterations_lbfgs = 150000

    # k = 3
    #nodes = 80
    #hid_lay = 4
    #iterations_lbfgs = 300000

    # k = 4
    nodes = 100
    hid_lay = 4
    iterations_lbfgs = 600000

    # k = 1
    #n_int_sp = 10
    #n_int_tm = 1000

    # k = 2
    #n_int_sp = 15
    #n_int_tm = 1500

    # k = 3
    #n_int_sp = 20
    #n_int_tm = 2000

    # k = 4
    n_int_sp = 25
    n_int_tm = 2500

    N_fx = 160 # 1280
    N_fy = 160 # 1280
    M_f = 160 # 1280
    XYT_exact = exact_soln(T_max, N_fx, N_fy, M_f, aconx, acony, kx, ky, w, betax, betay, k_c, PI)

    XYT_c = construct_interior_sampling_points(T_max, betax, betay, n_int_tm, n_int_sp, aconx, acony)

    u_layers = [3] + hid_lay * [nodes] + [1]
    v_layers = [3] + hid_lay * [nodes] + [1]

    iterstr = "_" + str(iterations_lbfgs) + "_"

    pathu = "uNN_2d_k_" + str(kx)[0] + iterstr + str(n_int_sp) + "_" + str(n_int_tm) + "_pnts_tanh_delta_" + str(
        hid_lay) + "_X_" + str(nodes) + ".pickle"
    pathv = "vNN_2d_k_" + str(kx)[0] + iterstr + str(n_int_sp) + "_" + str(n_int_tm) + "_pnts_tanh_delta_" + str(
        hid_lay) + "_X_" + str(nodes) + ".pickle"

    path_loss = "loss_2d_k_" + str(kx)[0] + iterstr + str(n_int_sp) + "_" + str(
        n_int_tm) + "_pnts_tanh_delta_" + str(hid_lay) + "_X_" + str(nodes) + ".txt"

    with tf.device('/device:GPU:0'):

    #if tf.config.experimental.list_logical_devices('GPU'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XYT_exact, u_layers, v_layers, betax, betay, aconx, acony, alpha, kx, ky, w, k_c, iterations_lbfgs)

        print(iterstr)
        print("start diff ")
        print(f"u_layers = {u_layers}")
        print(f"v_layers = {v_layers}")
        print(f"spatial = {n_int_sp}")
        print(f"time pnts = {n_int_tm}")
        print(f"kx = {kx}")
        # Train the composite network
        start_time = time.time()
        model.train_bfgs(XYT_c)
        elapsed = time.time() - start_time
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save the trained model
        model.save_NN(pathu, TYPE='U')
        model.save_NN(pathv, TYPE='V')

        # Check the loss for each part
        model.getloss()

        loss_value_u, loss_value_v, loss_f_u, u_mult, v_mult = model.GetFinalLoss(XYT_c)
        
        file = open(path_loss, "w")
        file.write('loss_value_u= ' + str(loss_value_u) + '\n')
        file.write('loss_value_v = ' + str(loss_value_v) + '\n')
        file.write('loss_f_u = ' + str(loss_f_u) + '\n')
        file.write('u_mult = ' + str(u_mult) + '\n')
        file.write('v_mult = ' + str(v_mult) + '\n')
        file.write('elapsed time = ' + str(elapsed) + '\n')
        file.close()
