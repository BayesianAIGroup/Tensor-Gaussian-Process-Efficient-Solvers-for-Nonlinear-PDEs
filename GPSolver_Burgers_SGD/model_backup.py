from utilities import *
from kernels import *

from tensorly.cp_tensor import cp_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor


class GP_TD(object):
    def __init__(self, data_gen, args):

        self.args = args
        self.data_gen = data_gen

        self.Kernels = [Kernel(self.args.jitter[i], Matern(k_mapping[self.args.kernel_s[i]])) for i in range(self.args.n_order)]
        self.log_ls_exp = jnp.exp(jnp.array(self.args.log_lsx))

        self.X_train_collocation = None
        
        self.X_train_Bound = None
        self.Y_train_Bound = None

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, key):
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        # X_temp = X[jax.random.choice(key, jnp.arange(X.shape[0]), shape=(self.args.train_batch,), replace=False), ...]

        if self.args.regular:
            K_list = [self.Kernels[0].get_kernel_matrix(self.data_gen.x1_point_train, self.log_ls_exp[0]), \
                      self.Kernels[1].get_kernel_matrix(self.data_gen.x2_point_train, self.log_ls_exp[1])]
            K_inv_mu = [jnp.linalg.solve(K_list[i], params["mu"][str(i)]) for i in range(len(K_list))]
            log_prior_list = [params["mu"][str(i)] * K_inv_mu[i] for i in range(len(K_inv_mu))]

            log_prior_sum = 0.0
            for log_prior in log_prior_list:
                log_prior_sum += (log_prior).sum()

        u, u_dx1, u_ddx1, u_dx2 = self.pred_grad(params, self.X_train_collocation)
        log_ll_eq = jnp.sum(jnp.square(u_dx2.reshape(-1, 1) + u.reshape(-1, 1) * u_dx1.reshape(-1, 1) - self.args.nu * u_ddx1.reshape(-1, 1)))

        u_b = vmap(self.to_tensor, (None, 0, 0))(params, self.X_train_Bound[:, 0], self.X_train_Bound[:, 1])

        log_ll_boundaries = jnp.sum(jnp.square(u_b.reshape(-1) - self.Y_train_Bound.reshape(-1)))
        loss = self.args.alpha * log_ll_eq + self.args.beta * log_ll_boundaries + log_prior_sum
        return loss
    
    @partial(jit, static_argnums=(0,))
    def pred_grad(self, params, X):
        preds = vmap(self.to_tensor, (None, 0, 0))(params, X[:, 0], X[:, 1])
        preds_x1 = vmap(grad(self.to_tensor, 1), (None, 0, 0))(params, X[:, 0], X[:, 1])
        preds_x2 = vmap(grad(self.to_tensor, 2), (None, 0, 0))(params, X[:, 0], X[:, 1])
        preds_xx1 = vmap(grad(grad(self.to_tensor, 1), 1), (None, 0, 0))(params, X[:, 0], X[:, 1])
        return (
            preds.reshape(-1),
            preds_x1.reshape(-1),
            preds_xx1.reshape(-1),
            preds_x2.reshape(-1),
        )
    
    @partial(jit, static_argnums=(0,))
    def to_tensor(self, params, x1, x2):
        K_list = [self.Kernels[0].get_kernel_matrix(self.data_gen.x1_point_train, self.log_ls_exp[0]), \
                  self.Kernels[1].get_kernel_matrix(self.data_gen.x2_point_train, self.log_ls_exp[1])]
        cov_list = [self.Kernels[0].get_cov(x1, self.data_gen.x1_point_train, self.log_ls_exp[0]), \
                    self.Kernels[1].get_cov(x2, self.data_gen.x2_point_train, self.log_ls_exp[1])]

        K_inv_mu = [jnp.linalg.solve(K_list[i], params["mu"][str(i)]) for i in range(len(K_list))]
        embed = [jnp.matmul(cov_list[i], K_inv_mu[i]) for i in range(len(cov_list))]
        if self.args.method == 0:
            pred_y = tt_to_tensor(embed).reshape(())
        elif self.args.method == 1:
            pred_y = tucker_to_tensor([params["core"], embed]).reshape(())
        else:
            pred_y = cp_to_tensor([params["core"], embed]).reshape(())
        return pred_y
    
    @partial(jit, static_argnums=(0,))
    def pred(self, params, X, y):
        preds = vmap(self.to_tensor, (None, 0, 0))(params, X[:, 0], X[:, 1])
        diff = preds.reshape(-1) - y.reshape(-1)
        mae = jnp.mean(jnp.abs(diff))
        relativeL2 = jnp.sqrt(jnp.sum(jnp.square(diff)))/self.data_gen.YNorm2_test
        rmse = jnp.sqrt(jnp.mean(jnp.square(diff)))
        return rmse, mae, relativeL2
