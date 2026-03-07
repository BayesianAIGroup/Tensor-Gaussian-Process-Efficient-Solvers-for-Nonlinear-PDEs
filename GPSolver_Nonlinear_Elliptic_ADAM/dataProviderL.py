from scipy.stats.qmc import LatinHypercube
from itertools import product
from utilities import *

class DataGen:

    def __init__(self, args):

        self.args = args

        self.x_range = [float(x) for x in self.args.x_range.split(',')]
        self.x1_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xind[0])
        self.x2_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xind[1])

        self.x1_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_x1test)
        self.x2_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_x2test)

        self.x_c, self.y_c = self.train_pde_gen()
        self.x_b = self.train_boundary_gen()
        self.args.n_train_boundary = self.x_b.shape[0]

        # self.sampler_training = qmc.LatinHypercube(d=self.args.n_order)
        # self.l_b_training, self.u_b_training = [0 for _ in range(self.args.n_order)], [self.args.n_x1train, self.args.n_x2train]

        # self.sampler_training_bound = qmc.LatinHypercube(d=self.args.n_order-1)
        # self.l_b_training_bound = [0 for _ in range(self.args.n_order-1)]
        # self.u_b_training_bound_special = [self.args.n_x1train, self.args.n_x2train]

        self.X_test, self.Y_test = self.test_data_gen()
        self.YNorm2_test = jnp.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = jnp.sum(jnp.abs(self.Y_test.reshape(-1)))

        # self.sampler_testing = qmc.LatinHypercube(d=self.args.n_order)
        # self.l_b_testing, self.u_b_testing = [0 for _ in range(self.args.n_order)], [self.args.n_x1test, self.args.n_x2test] + [self.args.n_alpha_test for _ in range(2)] + [self.args.n_w_test for _ in range(4)]
    def test_data_gen(self):
        xx2, xx1 = np.meshgrid(self.x2_point_test, self.x1_point_test)
        X_test = jnp.array(np.concatenate((xx1.reshape(-1, 1), xx2.reshape(-1, 1)), axis=1))
        Y_test = vmap(u_elliptic_zhitong)(X_test[:, 0], X_test[:, 1])
        return X_test, Y_test


    """
    def test_data_gen(self):

        test_pickup_idx = self.sampler_testing.integers(l_bounds=self.l_b_testing, u_bounds=self.u_b_testing, n=self.args.n_testing, endpoint=False)
        X_test = np.concatenate((self.x1_point_test[test_pickup_idx[:, 0]][..., None], self.x2_point_test[test_pickup_idx[:, 1]][..., None], \
            self.alpha_point_test[test_pickup_idx[:, 2:4]], self.w_point_test[test_pickup_idx[:, 4:]]), axis=-1)
        X_test = jnp.array(X_test)
        Y_test = vmap(u_6d)(X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3], X_test[:, 4], X_test[:, 5], X_test[:, 6], X_test[:, 7]).reshape(-1)
        return X_test, Y_test
    """
    def train_pde_gen(self):
        # train_pickup_idx = self.sampler_training.integers(l_bounds=self.l_b_training, u_bounds=self.u_b_training, n=self.args.n_train_collocation, endpoint=False)
        # return np.concatenate((self.x1_point_train[train_pickup_idx[:, 0]][..., None], self.x2_point_train[train_pickup_idx[:, 1]][..., None]), axis=-1)
        sampler = LatinHypercube(self.args.n_order, seed=self.args.seed, optimization='random-cd')
        X_samples = sampler.random(self.args.n_train_collocation)
        x1_sampling = self.x_range[0] + X_samples[:, 0:1] * (self.x_range[-1] - self.x_range[0])
        x2_sampling = self.x_range[0] + X_samples[:, 1:] * (self.x_range[-1] - self.x_range[0])
        X_train_col = np.concatenate((x1_sampling, x2_sampling), axis=-1)
        F_PDE = vmap(pde_force_gen, (0, 0))(x1_sampling, x2_sampling)
        return X_train_col, F_PDE.reshape(-1)


    def train_boundary_gen(self):
        X_lower_bound = np.array(self.x_range[0:1])
        X_upper_bound = np.array(self.x_range[1:])
        X_bound = np.array(self.x_range)
        sampler = LatinHypercube(self.args.n_order-1, seed=self.args.seed, optimization='random-cd')
            
        # sampling y and get x boundary:
        X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
        X_train_bound = np.array(list(product(X_lower_bound, X_samples.reshape(-1))))
        X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
        X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_upper_bound, X_samples.reshape(-1))))), axis=0)
        # X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
        # X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
        # X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            
        # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_upper_bound, X_samples.reshape(-1))))), axis=0)
            
        # sampling x and get y boundary:
        X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
        X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_lower_bound)))), axis=0)
        X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
        X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_upper_bound)))), axis=0)
        return X_train_bound


            # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_bound)))), axis=0)
            # X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
            # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_upper_bound)))), axis=0)


        # X_train_bound = np.empty(shape=(0, self.args.n_order))
        # for i in range(self.args.n_order):
        #     train_pickup_idx_ = self.sampler_training_bound.integers(l_bounds=self.l_b_training_bound, u_bounds=self.u_b_training_bound_special[:i] + \
        #         self.u_b_training_bound_special[(i+1):], n=self.args.n_train_boundary, endpoint=False)
        #     train_bound_idx_ = np.concatenate((train_pickup_idx_[:, :i], np.zeros((self.args.n_train_boundary, 1), dtype=int), train_pickup_idx_[:, i:]), axis=-1)
        #     X_train_bound = np.concatenate((X_train_bound, np.concatenate((self.x1_point_train[train_bound_idx_[:, 0]][..., None], \
        #         self.x2_point_train[train_bound_idx_[:, 1]][..., None]), axis=-1)), axis=0)
        #     if i == 0:
        #         bound_idx = np.ones((self.args.n_train_boundary, 1), dtype=int) * (self.args.n_x1train - 1)
        #     else:
        #         bound_idx = np.ones((self.args.n_train_boundary, 1), dtype=int) * (self.args.n_x2train - 1)
        #     train_bound_idx_ = np.concatenate((train_pickup_idx_[:, :i], bound_idx, train_pickup_idx_[:, i:]), axis=-1)
        #     X_train_bound = np.concatenate((X_train_bound, np.concatenate((self.x1_point_train[train_bound_idx_[:, 0]][..., None], \
        #                                                                    self.x2_point_train[train_bound_idx_[:, 1]][..., None]), axis=-1)), axis=0)   
        
        # X_train_bound = jnp.array(X_train_bound)
        # Y_train_bound = vmap(u_elliptic_zhitong)(X_train_bound[:, 0], X_train_bound[:, 1])
        # return X_train_bound, Y_train_bound





