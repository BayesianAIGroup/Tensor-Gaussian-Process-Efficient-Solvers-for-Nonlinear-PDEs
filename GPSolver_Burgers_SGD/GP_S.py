import argparse

from dataProviderL import DataGen
from model_running import model_run
from model import GP_TD
from utilities import *

def test_Burgers(args):
    data_gen = DataGen(args)
    write_log(args)
    gp = GP_TD(data_gen, args)

    model_run(gp, data_gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Solver for Burgers Dataset')
    parser.add_argument('--n-xtest', nargs='+', default=[100, 100], type=int,
                        help='num of test points per dimension (default: 100)')
    parser.add_argument('--n-train-boundary', type=int, default=280, metavar='N',
                        help='number of boundary points for training (default: 10000)')
    parser.add_argument('--n-train-collocation', type=int, default=1000, metavar='N',
                        help='number of collocation points for training (default: 1000)')
    parser.add_argument('--x1-range', type=str, default="-1.0,1.0", metavar='TUPLE',
                        help='position range for the experiment')
    parser.add_argument('--x2-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='time range for the experiment')

    parser.add_argument('--n-Gpt', type=int, default=250, metavar='N',
                        help='number of Gaussin Data Points for dataset builder (default: 80)')

    parser.add_argument('--nu', type=float, default=0.001, metavar='N',
                        help='viscousity for Burgers (default: 0.001)')
    parser.add_argument('--n-xtrain', nargs='+', default=[100, 40], type=int,
                        help='num of collocation points per dim ')
    parser.add_argument('--n-xind', nargs='+', default=[100, 40], type=int,
                        help='num of inducing points by each dim ')

    parser.add_argument('--random-sampling', action="store_true",
                        help='choose to use random sampling for boundary and collocation points gen')

    parser.add_argument('--train-batch-collocation', type=int, default=1000000000, metavar='N',
                        help='training batch for collocation points (default: 10000)')
    parser.add_argument('--train-batch-boundary', type=int, default=1000000000, metavar='N',
                        help='training batch for boundary points (default: 10000)')
    parser.add_argument('--rank', nargs='+', default=[10, 10], type=int,
                        help='rank per each variable')
    parser.add_argument('--kernel-s', nargs='+', default=[3, 3], type=int, 
                        help='kernel selection: 0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: inf')
    parser.add_argument('--jitter', nargs='+', default=[1e-10, 1e-10],
                        type=float, help='noise set on each kernel')
    parser.add_argument('--log-lsx', nargs='+', default=[4.0, 4.0], type=float,
                        help='land scale on each variable')
    

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
    
    parser.add_argument('--regulator', action="store_true",
                        help='wheter or not using prior distribution log likelihood as regulator')

    parser.add_argument('--n-order', type=int, default=2, metavar='N',
                        help='number of order in the equation (default: 2)')
    
    parser.add_argument('--device', type=str, default="cuda", metavar='S',
                        help='which device to run ((default: cuda))')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-store-path', type=str, default="./result/test", metavar='FILE',
                        help='path to store the training record and test result')
    parser.add_argument('--process-store-path', type=str, default="./process_data/", metavar='FILE',
                        help='path to store the training time, loss and relative L2')

    parser.add_argument('--n-test', type=int, default=10000, metavar='N',
                        help='number of test points selected for evaluation(default: 500)')
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    jax.config.update("jax_enable_x64", True)
    tl.set_backend("jax")

    test_Burgers(args)
