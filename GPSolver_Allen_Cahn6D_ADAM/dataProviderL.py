from scipy.stats import qmc
from utilities import *
import math

class DataGen:

    def __init__(self, args):
        self.args = args
        
        self.x_range = [float(x) for x in self.args.x_range.split(',')]
        self.x_point_train = [np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[i]) for i in range(args.n_order)]


        self.x_train_collocation, self.F_train_PDE = None, None
        self.x_train_boundary, self.y_train_boundary = None, None
        
        self.X_test, self.Y_test = None, None
        
        self._load_dataset()
        self.YNorm2_test = np.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = np.sum(np.abs(self.Y_test.reshape(-1)))

        self._adjust_argument()
        
    def _load_dataset(self):
        dataset = np.load(self.args.dataset_load_path)
        self.x_train_collocation, self.F_train_PDE = dataset['XTrain_Col'], dataset['FTrain_PDE']
        self.x_train_boundary, self.y_train_boundary = dataset['XTrain_Boundary'], dataset['YTrain_Boundary']
        dataset = np.load(self.args.test_dataset_load_path)
        self.X_test, self.Y_test = dataset['XTest'], dataset['YTest']
        
    def _adjust_argument(self):
        self.args.n_order = self.X_test.shape[-1]
        self.args.n_train_collocation = self.x_train_collocation.shape[0]
        self.args.n_train_boundary = self.x_train_boundary.shape[0]
        self.args.n_test = self.X_test.shape[0]
        self.args.train_batch_collocation = min(self.args.n_train_collocation, self.args.train_batch_collocation)
        self.args.train_batch_boundary = min(self.args.n_train_boundary, self.args.train_batch_boundary)
        
    """
    def _collocation_gen(self):
        sampler = qmc.LatinHypercube(self.args.n_order, optimization='random-cd')
        X_train_col = self.x_range[0] + sampler.random(self.args.n_train_collocation) * (self.x_range[-1] - self.x_range[0])
        F_train_col = pde_force_gen(X_train_col, self.args.beta)
        
        return X_train_col, F_train_col
    
    def _boundary_gen(self):
        sampler = qmc.LatinHypercube(self.args.n_order-1, optimization='random-cd')
        X_train_bound = np.empty(shape=(0, self.args.n_order))
        for i in range(self.args.n_order):
            train_samples = self.x_range[0] + sampler.random(self.args.n_train_boundary) * (self.x_range[-1] - self.x_range[0])
            lower_bound = np.concatenate((train_samples[:, :i], np.ones((self.args.n_train_boundary, 1)) * self.x_range[0], train_samples[:, i:]), axis=-1)
            upper_bound = np.concatenate((train_samples[:, :i], np.ones((self.args.n_train_boundary, 1)) * self.x_range[-1], train_samples[:, i:]), axis=-1)
            X_train_bound = np.concatenate((X_train_bound, lower_bound, upper_bound), axis=0)
            
        Y_train_bound = u_elliptic_nonLinear(X_train_bound, self.args.beta).reshape(-1)
        return X_train_bound, Y_train_bound
    
    def _test_gen(self):
        sampler = qmc.LatinHypercube(self.args.n_order, optimization='random-cd')
        X_test = self.x_range[0] + sampler.random(self.args.n_test) * (self.x_range[-1] - self.x_range[0])
        Y_test = u_elliptic_nonLinear(X_test, self.args.beta)
        return X_test, Y_test
    """ 
