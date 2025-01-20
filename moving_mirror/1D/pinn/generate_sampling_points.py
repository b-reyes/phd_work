import numpy as np
from scipy.io import savemat
from pyDOE import lhs
np.random.seed(1111)

T_max = 1.5
beta = -2.0
acon = 0.25


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

    XT_c = get_points(time_pnts, beta, nx, nt, acon)

    return XT_c


n_int_sp = 5
n_int_tm = 80
XT_c = construct_interior_sampling_points(T_max, beta, n_int_tm, n_int_sp, acon)

mdic = {"x_pnts": XT_c[:, 0], "t_pnts": XT_c[:, 1]}
savemat("1d_collocation_points.mat", mdic)
