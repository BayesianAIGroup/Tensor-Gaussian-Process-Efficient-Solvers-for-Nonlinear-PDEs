from utilities import *


class Matern(object):

    def __init__(self, nu):
        self.nu = nu

    @partial(jit, static_argnums=(0,))
    def kappa(self, x1, y1, ls):
        dist = jnp.abs(x1 - y1) / ls
        if self.nu == 0.5:
            K = jnp.exp(-dist)
        elif self.nu == 1.5:
            K = dist * jnp.sqrt(3)
            K = (1.0 + K) * jnp.exp(-K)
        elif self.nu == 2.5:
            K = dist * jnp.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
        elif self.nu == 3.5:
            K = dist * jnp.sqrt(7)
            K = (1.0 + K + K**2 / 5.0*2 + K**3 / 15.0) * jnp.exp(-K)
        elif self.nu == 4.5:
            # K = 3*dist * jnp.sqrt(2)
            # K = (1.0 + K + 3*K**2 / 4.0 + K**3 / 4.0+ K**4 / 24.0) * jnp.exp(-K)
            K = 3*dist
            K = (1.0 + K + (3.0/7.0)*K**2 + (2.0/21)*K**3 + (1.0/105)*K**4)*jnp.exp(-K)
        elif self.nu == jnp.inf:
            K = jnp.exp(-(dist**2) / 2.0)
        return K


class Kernel(object):

    def __init__(self, jitter, K_u):
        self.jitter = jitter
        self.K_u = K_u

    def get_kernel_matrix(self, X, ls):
        num_X = X.shape[0]
        x_p = jnp.tile(X.flatten(), (num_X, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        N = int((X1_p.shape[0]) ** 0.5)
        K_u_u = vmap(self.K_u.kappa, (0, 0, None))(
            X1_p.flatten(), X2_p.flatten(), ls
        ).reshape(N, N)
        # K_u_u = K_u_u + self.jitter * jnp.eye(N)
        K_u_u = K_u_u + jnp.exp(-self.jitter) * jnp.eye(N)
        return K_u_u

    def get_cov(self, X1, X2, ls):
        num_X1 = 1
        num_X2 = X2.shape[0]
        x1_p = jnp.tile(X1.flatten(), (num_X2, 1)).T
        x2_p = jnp.tile(X2.flatten(), (num_X1, 1)).T
        X1_p = x1_p.flatten()
        X2_p = jnp.transpose(x2_p).flatten()
        cov = vmap(self.K_u.kappa, (0, 0, None))(
            X1_p.flatten(), X2_p.flatten(), ls
        ).reshape(num_X1, num_X2)
        return cov
