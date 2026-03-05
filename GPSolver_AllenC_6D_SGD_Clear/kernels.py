from utilities import *


class Matern():

    def __init__(self, args, ks, amp, ls):
        self.args = args
        self.ks = ks
        self.ls = ls

        if self.ks <= self.args.rbf_threshold:
            self.coefficient = amp * (math.factorial(ks) / math.factorial(2 * ks)) * torch.tensor([math.factorial(ks + i) / (math.factorial(i) * \
            math.factorial(ks - i)) for i in range(ks+1)], dtype=FLOAT, device=self.args.device)
    

    def kappa(self, x1, y1):
        x1 = x1.reshape(x1.shape[0], -1)
        y1 = y1.reshape(y1.shape[0], -1)

        dist = torch.cdist(x1, y1, p=2.0) / self.ls
        if self.ks <= self.args.rbf_threshold:
            tempK = np.sqrt(2 * self.ks + 1) * dist
        
            return torch.sum(torch.cat([torch.pow(2 * tempK, self.ks-i).unsqueeze(0) for i in range(self.ks+1)], dim=0) * \
                self.coefficient[..., None, None], dim=0) * torch.exp(-tempK)
        else:
            # return torch.exp(-(dist**2) / 2.0)
            return torch.exp(- 0.5*(x1 - y1.T)**2 / torch.exp(2*self.ls))


class Kernel(object):

    def __init__(self, jitter, K_u):
        self.jitter = jitter
        self.K_u = K_u

    def get_kernel_matrix(self, X):
        num_X = X.shape[0]
        K_u_u = self.K_u.kappa(X, X)
        K_u_u = K_u_u + (torch.exp(-self.jitter)) * torch.eye(num_X, device=self.K_u.args.device)
        return K_u_u

    def get_cross_cov(self, X1, X2):
        K_u_u = self.K_u.kappa(X1, X2)
        return K_u_u