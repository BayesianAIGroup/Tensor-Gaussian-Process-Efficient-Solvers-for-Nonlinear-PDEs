import argparse

from dataProviderL import DataGen
from model_running import model_run
from model import GP_TD
from utilities import *

def test_Allen_Cahen(args):
    data_gen = DataGen(args)

    if not args.resume_training:
        write_log(args)

    gp = GP_TD(data_gen, args)

    model_run(gp, data_gen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Solver for Allen Cahen High Dim')
    parser.add_argument('--n-xtrain', nargs='+', default=[100, 100, 100, 100, 100, 100], type=int,
                        help='number of points per dim (default: 100)')

    parser.add_argument('--x-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='x1 range for the experiment')
    
    parser.add_argument('--rank', nargs='+', default=[3, 3, 3, 3, 3, 3], type=int,
                        help='rank per each variable')
    parser.add_argument('--kernel-s', nargs='+', default=[8, 8, 8, 8, 8, 8], type=int, 
                        help='kernel selection: 0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: inf')
    parser.add_argument('--rbf-threshold', type=int, default=7, metavar='N',
                        help='threshold to use rbf kernel (default: 7)')
    parser.add_argument('--jitter', nargs='+', default=[25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
                        type=float, help='noise set on each kernel')
    parser.add_argument('--lsx', nargs='+', default=[-4.0, -4.0, -4.0, -4.0, -4.0, -4.0], type=float,
                        help='land scale on each variable')
    parser.add_argument('--amplitude', nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], type=float, 
                        help='amplitude for the kernels(must > 0.0)')

    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-gap', type=int, default=2000, metavar='N',
                        help='learning rate modification gap(default: 2000)')
    parser.add_argument('--lr-decay', type=float, default=0.8, metavar='N',
                        help='learning rate decay per round(default: 0.8)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--early-stop', type=int, default=5000, metavar='N',
                        help='number of epochs traced for update')
    
    parser.add_argument('--method', type=int, default=3, metavar='N',
                        help='tensor method choice-0: TT, 1: Tucker, 2: CP, 3: TR')
    parser.add_argument('--alpha', type=float, default=125, metavar='N',
                        help='coefficient1 setting weights for loss(default: 0.5)')
    parser.add_argument('--beta', type=float, default=1e45, metavar='N',
                        help='coefficient1 setting weights for loss(default: 0.5)')
    parser.add_argument('--m', type=int, default=3, metavar='S',
                        help='variable control nonlinearity of PDE (default: 0)')
    parser.add_argument('--gamma', type=int, default=1, metavar='S',
                        help='coefficient control nonlinearity of PDE (default: 0)')

    
    parser.add_argument('--regulator', action="store_true",
                        help='wheter or not using prior distribution log likelihood as regulator')

    
    parser.add_argument('--n-train-boundary', type=int, default=200, metavar='N',
                        help='number of boundary points to be selected during training per axis(default: 500)')
    parser.add_argument('--n-train-collocation', type=int, default=1000, metavar='N',
                        help='number of collocation points to be picked per training epoch (default: 10000)')
    parser.add_argument('--n-train-batch', type=int, default=6500, metavar='N',
                        help='number of collocation points to be picked per training epoch (default: 10000)')


    parser.add_argument('--n-order', type=int, default=6, metavar='N',
                        help='number of order in the equation (default: 6)')
    
    parser.add_argument('--device', type=str, default="cuda", metavar='S',
                        help='which device to run ((default: cuda))')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-store-path', type=str, default="./result/test", metavar='FILE',
                        help='path to store the training record and test result')
    parser.add_argument('--process-store-path', type=str, default="./process_data/process_test.npz", metavar='FILE',
                        help='path to store the training time, loss and relative L2')
    parser.add_argument('--best-model-dir', type=str, default="./model_store/best_model.pth", metavar='FILE',
                        help='path to store the best model until now')
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoint/check_model.pth", metavar='FILE',
                        help='path to store the most current model until now')
    parser.add_argument('--resume-training', action="store_true",
                        help='wheter or not loading existing model')
    
    
    parser.add_argument('--dataset-load-path', type=str, default="./dataset/Dim6_0/Allen_CahenC48000_B9600.npz", metavar='FILE',
                        help='path to load the dataset')
    parser.add_argument('--test-dataset-load-path', type=str, default="./dataset/Allen_Cahen_Test1000000.npz", metavar='FILE',
                        help='path to load the dataset')

    parser.add_argument('--test-batch', type=int, default=10000, metavar='N',
                        help='testing batch during testing phase (default: 10000)')
    parser.add_argument('--train-batch-collocation', type=int, default=20000, metavar='N',
                        help='training batch for collocation points (default: 10000)')
    parser.add_argument('--train-batch-boundary', type=int, default=1000000000, metavar='N',
                        help='training batch for boundary points (default: 10000)')
    
    parser.add_argument('--n-test', type=int, default=1000, metavar='N',
                        help='number of test points selected for evaluation(default: 500)')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.device == "cuda":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tl.set_backend('pytorch')
    

    test_Allen_Cahen(args)
