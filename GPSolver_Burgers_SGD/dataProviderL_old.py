from utilities import *


class DataGen:

    def __init__(self, args):

        self.args = args

        # position mesh
        self.x1_range = [float(x) for x in self.args.x1_range.split(',')]
        self.x1_point_train = np.linspace(self.x1_range[0], self.x1_range[-1], self.args.n_x1train)

        # time mesh
        self.x2_range = [float(x) for x in self.args.x2_range.split(',')]
        self.x2_point_train = np.linspace(self.x2_range[0], self.x2_range[-1], self.args.n_x2train)

        self.x_train_collocation = None
        self.x_train_boundary, self.y_train_boundary = None, None
        self.X_test, self.Y_test = None, None

        self._load_dataset()

        self.YNorm2_test = jnp.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = jnp.sum(jnp.abs(self.Y_test.reshape(-1)))

        self._adjust_argument()
        
    def _load_dataset(self):
        dataset = np.load(self.args.dataset_load_path)
        self.args.random_sampling = dataset['train_sampling']
        self.args.nu = dataset['nu']
        self.x_train_collocation = dataset['XTrain_Col']
        self.x_train_boundary, self.y_train_boundary = dataset['XTrain_Boundary'], dataset['YTrain_Boundary']
        self.X_test, self.Y_test = jnp.array(dataset['TestX']), jnp.array(dataset['TestY'])
        
    def _adjust_argument(self):
        self.args.n_order = self.X_test.shape[-1]
        self.args.n_train_collocation = self.x_train_collocation.shape[0]
        self.args.n_train_boundary = self.x_train_boundary.shape[0]
        self.args.n_test = self.X_test.shape[0]
        self.args.train_batch_collocation = min(self.args.train_batch_collocation, self.args.n_train_collocation)
        self.args.train_batch_boundary = min(self.args.train_batch_boundary, self.args.n_train_boundary)


    """
    def _test_dataset_gen(self):
        [Gauss_pts, weights] = np.polynomial.hermite.hermgauss(self.args.n_Gpt)

        X = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtest)
        # T = np.ones(1).reshape(-1,1)
        T = np.linspace(self.t_range[0], self.t_range[-1], self.args.n_ttest)

        tt, xx = np.meshgrid(T, X)
        X_test = np.concatenate((xx.reshape(-1, 1), tt.reshape(-1, 1)), axis=1)
        Y_test = vmap(u_truth_Burgers, (None, None, 0, 0))(Gauss_pts, weights, X_test[:, 1], X_test[:, 0]).reshape(-1)
        return jnp.array(X_test), Y_test
    
    def _train_dataset_gen(self):

        tt, xx = np.meshgrid(self.t_point_train, self.x_point_train)
        X_train = np.concatenate((xx.reshape(-1, 1), tt.reshape(-1, 1)), axis=1)
        X_b1 = np.concatenate((xx[:, 0].reshape(-1, 1), tt[:, 0].reshape(-1, 1)), axis=1) # when t = 0
        X_b2 = np.concatenate((xx[0, :].reshape(-1, 1), tt[0, :].reshape(-1, 1)), axis=1) # when x = -1
        X_b3 = np.concatenate((xx[-1, :].reshape(-1, 1), tt[-1, :].reshape(-1, 1)), axis=1) # when x = 1

        X_b = np.concatenate((X_b1, X_b2, X_b3), axis=0)
        return jnp.array(X_train), X_b

    """



