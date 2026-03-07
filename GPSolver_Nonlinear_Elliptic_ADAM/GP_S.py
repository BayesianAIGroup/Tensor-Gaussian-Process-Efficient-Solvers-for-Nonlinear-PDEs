import tensorly as tl
import argparse

from dataProviderL import DataGen
from model_running import model_run
from model import GP_TD
from utilities import *

def gp_solver(args):

    data_gen = DataGen(args)
    write_log(args)


    gp = GP_TD(data_gen, args)
    model_run(gp, data_gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Solver for Elliptic-non Zhitong')
    parser.add_argument('--n-x1test', type=int, default=100, metavar='N',
                        help='number of points along 1st dir for test dataset (default: 10)')
    parser.add_argument('--n-x2test', type=int, default=100, metavar='N',
                        help='number of points along 2nd dir for test dataset (default: 10)')

    parser.add_argument('--x-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='x1 range for the experiment')
    
    parser.add_argument('--n-xind', nargs='+', default=[100, 100], type=int,
                        help='num of collocation points per dim ')
    parser.add_argument('--n-xtrain', nargs='+', default=[100, 100], type=int,
                        help='num of collocation points per dim ')

    
    parser.add_argument('--rank', nargs='+', default=[10, 10], type=int,
                        help='rank per each variable')
    parser.add_argument('--kernel-s', nargs='+', default=[5, 5], type=int, 
                        help='kernel selection: 0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: inf')
    parser.add_argument('--jitter', nargs='+', default=[22.5, 22.5], type=float, 
                        help='noise set on each kernel')
    parser.add_argument('--log-lsx', nargs='+', default=[-2.5, -2.5], type=float, 
                        help='land scale on each variable')
    

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-gap', type=int, default=2000, metavar='N',
                        help='learning rate modification gap(default: 2000)')
    parser.add_argument('--lr-decay', type=float, default=0.8, metavar='N',
                        help='learning rate decay per round(default: 0.8)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--early-stop', type=int, default=5000, metavar='N',
                        help='number of epochs traced for update')
    
    parser.add_argument('--method', type=int, default=2, metavar='N',
                        help='tensor method choice-0: TT, 1: Tucker, 2: CP')
    parser.add_argument('--alpha', type=float, default=1e43, metavar='N',
                        help='coefficient1 setting weights for loss(default: 0.5)')
    parser.add_argument('--beta', type=float, default=1e45, metavar='N',
                        help='coefficient1 setting weights for loss(default: 0.5)')
    
    
    parser.add_argument('--n-train-boundary', type=int, default=1000, metavar='N',
                        help='number of boundary points to be selected during training per axis(default: 1000)')
    parser.add_argument('--n-train-collocation', type=int, default=10000, metavar='N',
                        help='number of collocation points to be picked per training epoch (default: 10000)')
    parser.add_argument('--n-testing', type=int, default=10000, metavar='N',
                        help='number of testing points to be selected for validation during training (default: 10000)')

    parser.add_argument('--n-order', type=int, default=2, metavar='N',
                        help='number of order in the equation (default: 8)')
    

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-store-path', type=str, default="./result/test2", metavar='FILE',
                        help='path to store the training record and test result')
    parser.add_argument('--process-store-path', type=str, default="./process_data/process_test.npz", metavar='FILE',
                        help='path to store the training time, loss and relative L2')
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    jax.config.update("jax_enable_x64", True)
    tl.set_backend("jax")

    gp_solver(args)
