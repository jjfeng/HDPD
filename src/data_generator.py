import numpy as np
from scipy.stats import multivariate_normal

class DataGenerator:
    """
    Base class: generates uniform X
    """
    scale = 4
    def __init__(self, beta, intercept, x_mean, w_indices, nonlinear):
        self.beta = beta
        self.x_mean = x_mean.reshape((1,-1))
        self.intercept = intercept
        self.num_p = beta.size
        self.w_indices = w_indices
        self.w_mask = np.repeat([False], self.num_p)
        if self.w_indices.size != 0:
            self.w_mask[w_indices] = True
        self.nonlinear = nonlinear
    
    @property
    def num_w(self):
        return int(self.w_mask.sum())

    def _get_prob(self, X):
        transform_X = X
        if self.nonlinear:
            transform_X = np.concatenate([
                X[:,:-2],
                # np.abs(X[:,2:4]),
                np.abs(X[:,-2:])
                ], axis=1)
        logit = np.matmul(transform_X, self.beta.reshape((-1,1))) + self.intercept
        return 1/(1 + np.exp(-logit))
    
    def _get_density_W(self, W):
        if self.w_indices.size != 0:
            return np.logical_and(
                np.all(W >= -self.scale + self.x_mean[self.w_indices], axis=1),
                np.all(W <= self.scale + self.x_mean[self.w_indices], axis=1),
            ).astype(float)
        else:
            return 1
    
    def _get_density_XW(self, XW):
        return np.logical_and(
            np.all(XW >= -self.scale + self.x_mean, axis=1),
            np.all(XW <= self.scale + self.x_mean, axis=1),
        ).astype(float)
    
    def _get_density_XWs(self, XW, subgroup_mask):
        return np.logical_and(
            np.all(XW[:, subgroup_mask] >= -self.scale + self.x_mean[:, subgroup_mask], axis=1),
            np.all(XW[:, subgroup_mask] <= self.scale + self.x_mean[:, subgroup_mask], axis=1),
        ).astype(float)
    
    def _generate_X(self, num_obs):
        return (np.random.rand(num_obs, self.num_p) - 0.5) * self.scale * 2 + self.x_mean
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return (np.random.rand(Xms.shape[0], subgroup.sum()) - 0.5) * self.scale * 2 + self.x_mean[:, subgroup]
    
    def _get_W(self, X):
        return X[:, self.w_mask]
    
    def _get_XminusW(self, X):
        return X[:, ~self.w_mask]
    
    def _concat_XW(self, X, W):
        XW = np.zeros((X.shape[0], self.num_p))
        XW[:, self.w_mask] = W
        XW[:, ~self.w_mask] = X
    
    def _generate_Y(self, X):
        probs = self._get_prob(X)
        y = np.random.binomial(1, probs.flatten(), size=probs.size)
        return y

    def generate(self, num_obs):
        X = self._generate_X(num_obs)
        y = self._generate_Y(X)
        return X, y    

class DataGeneratorMultiNorm(DataGenerator):
    """
    Base class: generates normally distributed X
    """
    scale = 2

    def _get_density_W(self, W):
        return multivariate_normal.pdf(W, mean=self.x_mean[:, self.w_mask].flatten(), cov=self.scale**2)
    
    def _get_density_XW(self, XW):
        return multivariate_normal.pdf(XW, mean=self.x_mean.flatten(), cov=self.scale**2)
    
    def _get_density_XWs(self, XW, subgroup_mask):
        if np.all(~subgroup_mask):
            return 1
        else:
            return multivariate_normal.pdf(XW[:, subgroup_mask], mean=self.x_mean[:, subgroup_mask].flatten(), cov=self.scale**2)
    
    def _generate_X(self, num_obs):
        return np.random.randn(num_obs, self.num_p) * self.scale + self.x_mean
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return np.random.randn(Xms.shape[0], subgroup.sum()) * self.scale + self.x_mean[:, subgroup]

class DataGeneratorSeqNorm(DataGenerator):
    """
    Base class: generates normally distributed X
    """
    w_scale = 1
    def __init__(self, beta, intercept, x_mean, w_indices, scale, nonlinear):
        self.beta = beta
        self.x_mean = x_mean.reshape((1,-1))
        self.intercept = intercept
        self.num_p = beta.size
        self.w_indices = w_indices
        self.w_mask = np.repeat([False], self.num_p)
        if self.w_indices.size != 0:
            self.w_mask[w_indices] = True
        self.nonlinear = nonlinear
        self.scale = scale

    def _generate_X(self, num_obs):
        w = np.random.randn(num_obs, self.num_w) * self.w_scale + self.x_mean[:,:self.num_w]
        x_early_eps = np.random.randn(num_obs, self.num_p-1 - self.num_w) * self.scale
        x_early = x_early_eps + w + self.x_mean[:,self.num_w:-1]
        eps1 = np.random.randn(num_obs, 1) * 0.5 + 0.5 + self.x_mean[:,-1:]
        eps2 = np.random.randn(num_obs, 1) * 0.5 - 0.5 + self.x_mean[:,-1:]
        choice = np.random.choice(2, size=(num_obs, 1), replace=True)
        x_later = -x_early[:,-1:] + eps1 * choice + eps2 * (1 - choice)
        # eps1 = np.random.randn(num_obs, 1) - 1
        # eps2 = np.random.randn(num_obs, 1) + 1
        # choice = x_early[:,-1:] > 0
        # x_later = eps1 * choice + eps2 * (1 - choice)
        return np.concatenate([w, x_early, x_later], axis=1)
    
    def _generate_Xs_Xms(self, subgroup, Xms):
        """
        Generate X_s|X_{-s}
        """
        return np.random.randn(Xms.shape[0], subgroup.sum()) * self.scale + self.x_mean[:, subgroup]

    def _get_density_W(self, W):
        raise NotImplementedError()
    
    def _get_density_XW(self, XW):
        raise NotImplementedError()
    
    def _get_density_XWs(self, XW, subgroup_mask):
        raise NotImplementedError()