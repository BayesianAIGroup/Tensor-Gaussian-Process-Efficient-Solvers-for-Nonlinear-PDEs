from kernels import *
import copy

class GP_TD(nn.Module):
    def __init__(self, data_gen, args):
        super(GP_TD, self).__init__()

        self.args = args
        self.data_gen = data_gen

        self.mu, self.core = self._get_tensor_parameters() # tensor information
        
        self.ls = self._pretreat_landscale()
        self.amplitude = self._pretreat_amplitude()
        self.jitter = self._pretreat_jitter()

        self.Kernels = [Kernel(self.jitter[i], Matern(args, args.kernel_s[i], self.amplitude[i], self.ls[i])) for i in range(args.n_order)]
        self.Kernel_Matrix = [self.Kernels[i].get_kernel_matrix(torch.tensor(data_gen.x_point_train[i], dtype=FLOAT, device=args.device)) for i in range(args.n_order)]

        self.X_train_collocation = None
        self.Force_train_collocation= None
        
        self.X_train_Bound = None
        self.Y_train_Bound = None
        
        self.tensor_einsum_string = self._gen_einsum_string()
        
        

    def _get_tensor_parameters(self):
        mu, core = nn.ParameterList(), None
        if self.args.method == 0:
            pre_rank = 1
            for i in range(len(self.args.rank)):
                mu.append(nn.Parameter(0.1 * torch.randn(pre_rank, self.args.n_xtrain[i], self.args.rank[i], dtype=FLOAT, device=self.args.device), requires_grad=True))
                pre_rank = self.args.rank[i]
        elif self.args.method == 1:
            for i in range(len(self.args.rank)):
                mu.append(nn.Parameter(0.1 * torch.randn(self.args.n_xtrain[i], self.args.rank[i], dtype=FLOAT, device=self.args.device), requires_grad=True))
            core = nn.Parameter(0.1 * torch.randn(*self.args.rank, dtype=FLOAT, device=self.args.device), requires_grad=True)
        elif self.args.method == 2:
            Params_init = np.load("./initial_params/Params_init.npz")
            for i in range(self.args.n_order):
                mu.append(nn.Parameter(torch.tensor(Params_init[str(i)], dtype=FLOAT, device=self.args.device), requires_grad=True))
            # for i in range(len(self.args.rank)):
            #     mu.append(nn.Parameter(0.1 * torch.randn(self.args.n_xtrain[i], self.args.rank[0], dtype=FLOAT, device=self.args.device), requires_grad=True))
            # core = nn.Parameter(0.1 * torch.randn(self.args.rank[0], dtype=FLOAT, device=self.args.device), requires_grad=True)
            core = nn.Parameter(torch.ones(self.args.rank[0], dtype=FLOAT, device=self.args.device), requires_grad=True)
        else:
            Params_init = np.load("./initial_params/Params_TR_init.npz")
            for i in range(self.args.n_order):
                mu.append(nn.Parameter(torch.tensor(Params_init[str(i)], dtype=FLOAT, device=self.args.device), requires_grad=True))
            # pre_rank = self.args.rank[0]
            # for i in range(1, len(self.args.rank)):
            #     mu.append(nn.Parameter(0.1 * torch.randn(pre_rank, self.args.n_xtrain[i-1], self.args.rank[i], dtype=FLOAT, device=self.args.device), requires_grad=True))
            #     pre_rank = self.args.rank[i]
            # mu.append(nn.Parameter(0.1 * torch.randn(pre_rank, self.args.n_xtrain[-1], self.args.rank[0], dtype=FLOAT, device=self.args.device), requires_grad=True))
        return mu, core
    
    def _pretreat_landscale(self):
        # if self.args.landScale_train:
        #     return nn.ParameterList([nn.Parameter(torch.tensor(log_ls, dtype=FLOAT, device=self.args.device), requires_grad=True) for log_ls in self.args.log_lsx])
        # else:
        return [torch.tensor(ls, dtype=FLOAT, device=self.args.device) for ls in self.args.lsx]

    def _pretreat_amplitude(self):
        # if self.args.amplitude_train:
        #     return nn.ParameterList([nn.Parameter(torch.tensor(amp, dtype=FLOAT, device=self.args.device), requires_grad=True) for amp in self.args.amplitude])
        # else:
        return [torch.tensor(amp, dtype=FLOAT, device=self.args.device) for amp in self.args.amplitude]
        
    def _pretreat_jitter(self):
        # if self.args.jitter_train:
        #     return nn.ParameterList([nn.Parameter(torch.tensor(jitt, dtype=FLOAT, device=self.args.device), requires_grad=True) for jitt in self.args.jitter])
        # else:
        return [torch.tensor(jitt, dtype=FLOAT, device=self.args.device) for jitt in self.args.jitter]
    
    def _gen_einsum_string(self):

        if self.args.method == 1:
            s = 'abcdefghijklmnoqrstuvwxyz'
            command = s[:self.args.n_order]
            for i in range(self.args.n_order):
                command += ',' + 'p' + command[i]
            command += '->' + 'p'
        elif self.args.method == 2:
            command = (',br' * self.args.n_order + "->br")[1:]
        else:
            command = None
        return command
    
    def forward_boundary(self):
        pred_bound = self._to_tensor(self.X_train_Bound).unsqueeze(-1)
        boundary_loss = torch.sum(torch.square(pred_bound - self.Y_train_Bound))
        
        return boundary_loss
    
    def forward_col(self):
        pred_y = self._to_tensor(self.X_train_collocation).unsqueeze(-1)

        pred_force = self._pred_grad(pred_y).unsqueeze(-1)
        pde_loss = torch.sum(torch.square(pred_force - self.Force_train_collocation))
        return (self.args.alpha / self.args.beta) * pde_loss
    
    def forward_reg(self):
        return (1.0 / self.args.beta) * self._get_loss_prior_dis()
        
    def _get_loss_prior_dis(self):
        log_prior_list = []
        
        for i in range(self.args.n_order):
            if self.args.method == 0:
                mu = self.mu[i]
            else:
                mu = self.mu[i]
            log_prior_list.append(mu * torch.linalg.solve(self.Kernel_Matrix[i], mu))
        return sum([log_prior.sum() for log_prior in log_prior_list])
    
    def _to_tensor(self, X):
        embed = []
        for i in range(self.args.n_order):
            Cov = self.Kernels[i].get_cross_cov(X[i], torch.tensor(self.data_gen.x_point_train[i], dtype=FLOAT, device=self.args.device))
            if self.args.method == 0:
                mu = self.mu[i]
            else:
                mu = self.mu[i]
            embed.append(Cov @ torch.linalg.solve(self.Kernel_Matrix[i], mu))
        if self.args.method == 0:
            y_pred = embed[0]
            for j in range(1, self.args.n_order):
                y_pred = torch.einsum("abc,cbd->abd", y_pred, embed[j])
        elif self.args.method == 1:
            y_pred = torch.einsum(self.tensor_einsum_string, self.core, *embed)
        elif self.args.method == 2:
            y_pred = torch.sum(torch.einsum(self.tensor_einsum_string, *embed) * self.core.unsqueeze(0), dim=-1)
        else:
            y_pred = embed[0]
            for j in range(1, self.args.n_order):
                y_pred = torch.einsum("abc,cbd->abd", y_pred, embed[j])
            y_pred = torch.einsum("ibi->b", y_pred)
        return y_pred.reshape(-1)

    def _pred_grad(self, pred_y):
        predy_sum = torch.sum(pred_y)

        pred_pde = self.args.gamma * (pred_y**self.args.m - pred_y)
        for i in range(self.args.n_order):
            dudx = torch.autograd.grad(predy_sum, self.X_train_collocation[i], create_graph=True)[0]
            pred_pde += torch.autograd.grad(dudx.sum(), self.X_train_collocation[i], create_graph=True)[0]

        return pred_pde.reshape(-1)
    
    def pred(self, X_test):
        return self._to_tensor([torch.tensor(X_test[:, i:(i+1)], dtype=FLOAT, device=self.args.device) for i in range(self.args.n_order)])
    
    def save_state(self):
        return copy.deepcopy(self.state_dict())

    def load_state(self, state):
        self.load_state_dict(state)
    
