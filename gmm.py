import numpy as np
import time
from numpy.testing import assert_allclose


class GMM():
    def __init__(self, C=3, seed=None):
        self.N = None
        self.Dimension = None
        self.C = C
        self.seed = seed

        if self.seed:
            np.random.seed(self.seed)

    def _initial_param(self):
        N = self.N
        d = self.Dimension
        C = self.C

        p_i = np.random.rand(C)
        self.p_i = p_i / p_i.sum()

        self.mu = np.random.uniform(C)
        self.Q = np.zeros((N, C))
        self.sigma = [np.eye(d) for _ in range(C)]

        self.best_p_i = self.p_i
        self.best_mu = self.mu
        self.best_Q = self.Q
        self.best_sigma = self.sigma
        self.best_elbo = -np.inf

        verbose = "gmm参数初始化完成，{}s"
        print(verbose.format(time.ctime()))

    def boundary(self):
        N = self.N
        C = self.C

        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            x_i = self.X[i]

            for c in range(C):
                pi_c = self.p_i[c]
                z_nk = self.Q[i, c]
                mu_c = self.mu[c]
                sigma_c = self.sigma[c, :, :]

                log_pi_c = np.log(pi_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)
                prob = z_nk *(log_p_x_i + log_pi_c)##这一句似乎只是用做计算的衡量值，无具体的推导来由，使用的模板一个是信息熵

                expec1 += prob
                expec2 += z_nk*np.log(z_nk)

        loss = expec1 - expec2
        return loss

    def fit(self, X, max_iter=100, tol=1e-2, verbose=False):
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]

        self._initial_param()
        prev_vlb = -np.inf

        for it in range(max_iter):
            self._E_step()
            self._M_step()

            vlb = self.boundary()

            converg = it > 0 and np.abs(vlb - prev_vlb) <= tol
            if converg or np.isnan(vlb):
                break

            prev_vlb = vlb

            if vlb > self.best_elbo:
                self.best_elbo = vlb
                self.best_p_i = self.p_i
                self.best_mu = self.mu
                self.best_Q = self.Q
                self.best_sigma = self.sigma
        return 0

    def _E_step(self):
        N = self.N
        C = self.C

        for i in range(N):
            x_i = self.X[i, :]

            val = []
            for c in range(C):
                p_i_c = self.p_i[c]
                mu_c = self.mu[c]
                sigma_c = self.sigma[c]

                log_p_i_c = np.log(p_i_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c. sigma_c)

                val.append(log_p_x_i + log_p_i_c)
            log_val = logsumexp(val)
            q_i = np.exp([num - log_val for num in val])

            self.Q[i, :] = q_i

    def _M_step(self):
        N = self.N
        C = self.C

        Q_sum = np.sum(self.Q, axis=0)

        self.mu = [np.dot(self.Q[:, c], self.X) / Q_sum[c] for c in range(C)]

        for c in range(C):
            mu_c = self.mu[c, :]
            n_c = Q_sum[c]

            outer = np.zeros((self.d, self.d))
            for i in range(N):
                w_i_c = self.Q[i, c]
                xi = self.X[i, :]
                outer += w_i_c * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            self.sigma[c, :, :] = outer

def log_gaussian_pdf(x_i, mu, sigma):
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)


def logsumexp(log_probs, axis=None):
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)

