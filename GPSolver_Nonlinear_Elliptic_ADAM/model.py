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


    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, key):
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, key):
        # loss probability prior distribution
        log_prior_list = []
        for i in range(self.args.n_order):
            if i == 0: xtrain_point = self.data_gen.x1_point_train
            else: xtrain_point = self.data_gen.x2_point_train
            K = self.Kernels[i].get_kernel_matrix(xtrain_point, self.args.log_lsx[i])

            log_prior_list.append(params["mu"][str(i)] * jnp.linalg.solve(K, params["mu"][str(i)]))
        log_prior_sum = sum([log_prior.sum() for log_prior in log_prior_list])
        
        # Collocation Points PDE Equality
        u, u_ddx1, u_ddx2 = self.pred_grad(params, jnp.array(self.data_gen.x_c))
        # force = vmap(pde_force_gen)(X_C[:, 0], X_C[:, 1])
        force = jnp.array(self.data_gen.y_c)
        log_ll_eq = jnp.sum(jnp.square(u.reshape(-1, 1)**3 - u_ddx1.reshape(-1, 1) - u_ddx2.reshape(-1, 1) - force.reshape(-1, 1)))
        
        # boundary conditions
        u_b = vmap(self.to_tensor, (None, 0, 0))(params, self.data_gen.x_b[:, 0], self.data_gen.x_b[:, 1])
        log_ll_boundaries = jnp.sum(jnp.square(u_b.reshape(-1)))

        loss = (self.args.alpha / self.args.beta) * log_ll_eq + log_ll_boundaries + (1.0 / self.args.beta) * log_prior_sum
        return loss

    
    @partial(jit, static_argnums=(0,))
    def pred_grad(self, params, X):
        preds = vmap(self.to_tensor, (None, 0, 0))(params, X[:, 0], X[:, 1])
        preds_xx1 = vmap(grad(grad(self.to_tensor, 1), 1), (None, 0, 0))(params, X[:, 0], X[:, 1])
        preds_xx2 = vmap(grad(grad(self.to_tensor, 2), 2), (None, 0, 0))(params, X[:, 0], X[:, 1])
        return (
            preds.reshape(-1),
            preds_xx1.reshape(-1),
            preds_xx2.reshape(-1),
        )
    
    @partial(jit, static_argnums=(0,))
    def to_tensor(self, params, x, y):
        X = [x, y]
        embed = []
        for i in range(self.args.n_order):
            if i == 0: xtrain_point = self.data_gen.x1_point_train
            else: xtrain_point = self.data_gen.x2_point_train
            K = self.Kernels[i].get_kernel_matrix(xtrain_point, self.args.log_lsx[i])
            Cov = self.Kernels[i].get_cov(X[i], xtrain_point, self.args.log_lsx[i])
            embed.append(jnp.matmul(Cov, jnp.linalg.solve(K, params["mu"][str(i)])))
        if self.args.method == 0:
            pred_y = tt_to_tensor(embed).reshape(())
        elif self.args.method == 1:
            pred_y = tucker_to_tensor([params["core"], embed]).reshape(())
        else:
            pred_y = cp_to_tensor([params["core"], embed]).reshape(())
        return pred_y
    
    @partial(jit, static_argnums=(0,))
    def pred(self, params):
        start = time.perf_counter()
        preds = vmap(self.to_tensor, (None, 0, 0))(params, self.data_gen.X_test[:, 0], self.data_gen.X_test[:, 1])
        end = time.perf_counter()
        pred_diff = jnp.subtract(preds.reshape(-1), self.data_gen.Y_test.reshape(-1))

        relativeL1 = jnp.sum(jnp.abs(pred_diff))/self.data_gen.YSUM_ABS
        relativeL2 = jnp.sqrt(jnp.sum(jnp.square(pred_diff)))/self.data_gen.YNorm2_test
        return jnp.sqrt(jnp.square(pred_diff).mean()), relativeL1, relativeL2, end - start
