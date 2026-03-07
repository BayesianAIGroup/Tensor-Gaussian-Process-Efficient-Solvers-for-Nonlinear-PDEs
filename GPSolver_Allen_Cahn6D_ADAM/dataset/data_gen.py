import numpy as np
from scipy.stats.qmc import LatinHypercube
import argparse
import math
import itertools

def u_base(args, x1, x2):
    return np.sin(2.0*np.pi*args.a*x1) * np.cos(2.0*np.pi*args.a*x2) + np.sin(2.0*np.pi*x1) * np.cos(2.0*np.pi*x2)

def dduddx_base(args, x1, x2):
    return -2.0 * ((2.0*np.pi)**2) * ((args.a**2) * np.sin(2.0*np.pi*args.a*x1) * np.cos(2.0*np.pi*args.a*x2) + np.sin(2.0*np.pi*x1) * np.cos(2.0*np.pi*x2))


def u_ground_truth(args, X):
    u = 0.0
    for i in range(1, X.shape[-1]):
        u += u_base(args, X[:, i-1], X[:, i])
    # for pair in list(itertools.combinations(range(X.shape[-1]), 2)):
    #     u += u_base(args, X[:, pair[0]], X[:, pair[-1]])
    return u

def pde_force_gen(args, X_train_col):
    if args.n_order > 2:
        X_train_col = np.concatenate((X_train_col, X_train_col[:, 0:1]), axis=-1)
    u_truth = u_ground_truth(args, X_train_col)
    u_der = 0.0
    for i in range(1, X_train_col.shape[-1]):
        u_der += dduddx_base(args, X_train_col[:, i-1], X_train_col[:, i])
    # for pair in list(itertools.combinations(range(X_train_col.shape[-1]), 2)):
    #     u_der += dduddx_base(args, X_train_col[:, pair[0]], X_train_col[:, pair[-1]])

    return u_der + args.gamma * (u_truth**args.m - u_truth)


def collocation_gen(args, x_range):
    sampler = LatinHypercube(args.n_order, seed=args.seed, optimization='random-cd')
    X_train_col = x_range[0] + sampler.random(args.n_train_collocation) * (x_range[-1] - x_range[0])
    F_train_col = pde_force_gen(args, X_train_col)
    return X_train_col, F_train_col[..., None]


def boundary_gen(args, x_range):
    sampler = LatinHypercube(args.n_order-1, seed=args.seed, optimization='random-cd')
    X_train_bound = np.empty(shape=(0, args.n_order))
    n_perBound = math.ceil(args.n_train_boundary / (args.n_order * 2))
    for i in range(args.n_order):
        train_samples = x_range[0] + sampler.random(n_perBound) * (x_range[-1] - x_range[0])
        lower_bound = np.concatenate((train_samples[:, :i], np.ones((n_perBound, 1)) * x_range[0], train_samples[:, i:]), axis=-1)
        upper_bound = np.concatenate((train_samples[:, :i], np.ones((n_perBound, 1)) * x_range[-1], train_samples[:, i:]), axis=-1)
        X_train_bound = np.concatenate((X_train_bound, lower_bound, upper_bound), axis=0)
    if args.n_order > 2:
        Y_train_bound = u_ground_truth(args, np.concatenate((X_train_bound, X_train_bound[:, 0:1]), axis=-1))
    else:
        Y_train_bound = u_ground_truth(args, X_train_bound)
    return X_train_bound, Y_train_bound[..., None]


# def test_gen(args, x_range, alpha_range, w_range):
#     sampler = LatinHypercube(args.n_order, optimization='random-cd')
#     samples = sampler.random(args.n_test)
#     X_test = np.concatenate((x_range[0] + samples[:, 0:2] * (x_range[-1] - x_range[0]), alpha_range[0] + samples[:, 2:4] * (alpha_range[-1] - \
#         alpha_range[0]), w_range[0] + samples[:, 4:] * (w_range[-1] - w_range[0])), axis=-1)
#     Y_test = u_6D(X_test)
#     return X_test, Y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preparation for Allen Cahen high dimensional PDE')
    parser.add_argument('--x-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='position range for the experiment')

    parser.add_argument('--n-test', type=int, default=1000000, metavar='N',
                        help='number of points along 1st dir for test dataset (default: 1000000)')
    parser.add_argument('--n-train-boundary', type=int, default=200, metavar='N',
                        help='number of boundary points to be selected during training per axis(default: 500)')
    parser.add_argument('--n-train-collocation', type=int, default=1000, metavar='N',
                        help='number of collocation points to be picked per training epoch (default: 10000)')


    parser.add_argument('--n-order', type=int, default=3, metavar='N',
                        help='number of order in the equation (default: 8)')
    parser.add_argument('--m', type=int, default=3, metavar='S',
                        help='variable control nonlinearity of PDE (default: 0)')
    parser.add_argument('--gamma', type=int, default=1, metavar='S',
                        help='coefficient control nonlinearity of PDE (default: 0)')
    parser.add_argument('--a', type=int, default=15, metavar='S',
                        help='coefficient control nonlinearity of PDE (default: 0)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    
    parser.add_argument('--data-store-path', type=str, default="./beta", metavar='FILE',
                        help='path to store the dataset')
    
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    x_range = [float(x) for x in args.x_range.split(',')]

    x_train_collocation, F_train_PDE = collocation_gen(args, x_range)
    x_train_boundary, y_train_boundary = boundary_gen(args, x_range)

    # X_test, Y_test = test_gen(args, x_range, alpha_range, w_range)

    Dataset = {'XTrain_Col': x_train_collocation, 'FTrain_PDE': F_train_PDE, 'XTrain_Boundary': x_train_boundary, \
        'YTrain_Boundary': y_train_boundary}
    
    # Dataset = {'XTest': X_test, 'YTest': Y_test}
    
    np.savez(args.data_store_path, **Dataset)

    # np.save("Burgers_TestX.npy", X)
    # np.save("Burgers_TestY.npy", Y)

    # X_Test = np.load("Burgers_TestX.npy")
    # Y_Test = np.load("Burgers_TestY.npy")

    # for i in range(Y_Test.shape[0]):
    #     if math.isnan(Y_Test[i].item()):
    #         print(Y_Test[i])
