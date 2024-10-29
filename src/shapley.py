"""Do shapley computation, black box inference

adapted from vimpy github repo
"""
import logging

import math
import numpy as np
from scipy.stats import norm

## utility functions
def choose(n, k) -> int:
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))

def shapley_influence_function(Z, z_counts, W, v, psi, G, ics):
    """
    Compute influence function for the given predictiveness measure

    @param Z the subsets of the power set with estimates
    @param W the matrix of weights
    @param v the estimated predictivness
    @param psi the estimated Shapley values
    @param G the constrained ls matrix
    @param ics a list of all ics
    """
    ## compute contribution from estimating V
    Z_W = Z.transpose().dot(W)
    A_m = Z_W.dot(Z)
    A_m_inv = np.linalg.pinv(A_m)
    phi_01 = A_m_inv.dot(Z_W).dot(ics)

    ## compute contribution from estimating Q
    qr_decomp = np.linalg.qr(G.transpose(), mode = 'complete')
    U_2 = qr_decomp[0][:, 3:(Z.shape[1])]
    V = U_2.transpose().dot(Z.transpose().dot(W).dot(Z)).dot(U_2)
    phi_02_shared_mat = (-1) * U_2.dot(np.linalg.inv(V))
    phi_02_uniq_vectors = np.array([(Z[z, :].dot(psi) - v[z]) * (U_2.transpose().dot(Z[z, :])) for z in range(Z.shape[0])], dtype = np.float64).transpose()
    phi_02_uniq = phi_02_shared_mat.dot(phi_02_uniq_vectors)
    phi_02 = np.repeat(phi_02_uniq, z_counts, axis=1)

    return {'contrib_v': phi_01, 'contrib_s': phi_02}


def shapley_se(shapley_ics, idx, gamma, na_rm = True):
    """
    Standard error for the desired Shapley value

    @param shapley_ics: all influence function estimates
    @param idx: the index of interest
    @param gamma: the constant for sampling
    @param na_rm: remove NaNs?

    @return the standard error corresponding to the shapley value at idx
    """
    if na_rm:
        var_v = np.nanvar(shapley_ics['contrib_v'][idx, :])
        var_s = np.nanvar(shapley_ics['contrib_s'][idx, :])
    else:
        var_v = np.var(shapley_ics['contrib_v'][idx, :])
        var_s = np.var(shapley_ics['contrib_s'][idx, :])
    logging.info("SHAPLEY SE contrib 1 %f %f", var_v / shapley_ics['contrib_v'].shape[1], var_v)
    logging.info("SHAPLEY SE contrib 2 %f %f", var_s / shapley_ics['contrib_s'].shape[1] / gamma, var_s)
    se = np.sqrt(var_v / shapley_ics['contrib_v'].shape[1] + var_s / shapley_ics['contrib_s'].shape[1] / gamma)
    return se

class ShapleyInference:
    def __init__(self, num_obs: int, num_p: int, subset_inference_func, gamma: float = 1):
        self.gamma = gamma
        self.num_obs = num_obs
        self.num_p = num_p
        self.subset_inference_func = subset_inference_func

        self.ics_ = {}
        self.vimp_ = {}
        self.lambdas_ = {}
        self.ses_ = {}
        self.cis_ = {}
        self.Z_ = []
        self.z_counts_ = []
        self.v_ = {}
        self.v_ics_ = {}
        self.W_ = []
        self.G_ = np.vstack((np.append(1, np.zeros(self.num_p)), np.ones(self.num_p + 1) - np.append(1, np.zeros(self.num_p))))

    def _get_kkt_matrix(self):
        # kkt matrix for constrained wls
        A_W = np.sqrt(self.W_).dot(self.Z_)
        kkt_matrix_11 = 2 * A_W.transpose().dot(A_W)
        kkt_matrix_12 = self.G_.transpose()
        kkt_matrix_21 = self.G_
        kkt_matrix_22 = np.zeros((kkt_matrix_21.shape[0], kkt_matrix_12.shape[1]))
        kkt_matrix = np.vstack((np.hstack((kkt_matrix_11, kkt_matrix_12)), np.hstack((kkt_matrix_21, kkt_matrix_22))))
        return(kkt_matrix)

    def _get_ls_matrix(self, c_n, v):
        A_W = np.sqrt(self.W_).dot(self.Z_)
        v_W = np.sqrt(self.W_).dot(v)
        ls_matrix = np.vstack((2 * A_W.transpose().dot(v_W.reshape((len(v_W), 1))), c_n.reshape((c_n.shape[0], 1))))
        return(ls_matrix)

    ## calculate the point estimates
    def get_point_est(self):
        ## sample subsets, set up Z
        max_subset = np.array(list(range(self.num_p)))
        sampling_weights = np.append(np.append(1, [choose(self.num_p - 2, s - 1) ** (-1) for s in range(1, self.num_p)]), 1)
        subset_sizes = np.random.choice(
            self.num_p + 1, p = sampling_weights / sum(sampling_weights),
            size = int(self.gamma * self.num_obs),
            replace = True)
        S_lst_all = [np.sort(np.random.choice(self.num_p, subset_size, replace = False)) for subset_size in list(subset_sizes)]
        ## only need to continue with the unique subsets S
        Z_lst_all = [np.in1d(max_subset, S).astype(np.float64) for S in S_lst_all]
        Z, z_counts = np.unique(np.array(Z_lst_all), axis = 0, return_counts = True)
        Z_lst = list(Z)
        logging.info("Z_lst (num uniq sets %d, num sampled sets %d) %s", len(Z_lst), subset_sizes.size, Z_lst)
        print("Z_lst", Z_lst, len(Z_lst))
        Z_aug_lst = [np.append(1, z) for z in Z_lst]
        self.Z_ = np.array(Z_aug_lst)
        self.z_counts_ = z_counts
        self.W_ = np.diag(z_counts / np.sum(z_counts))
        kkt_matrix = self._get_kkt_matrix()                    
        ## get point estimates and EIFs for null set
        subgroup_empty = np.zeros(self.num_p, dtype=bool)
        res_empty = self.subset_inference_func(subgroup_empty)

        ## get point estimates and EIFs for remaining non-null groups in S
        self.inference_res_lst = [self.subset_inference_func(z.astype(bool)) for z in Z_lst[1:]]

        self.res_keys = list(res_empty.keys())

        ## set up full lists
        for key in self.res_keys:
            v_lst_all = [res_empty[key].estim] + [res[key].estim for res in self.inference_res_lst]
            self.v_[key] = np.array(v_lst_all)
            ic_lst_all = [res_empty[key].ic] + [res[key].ic for res in self.inference_res_lst]
            self.v_ics_[key] = ic_lst_all
            c_n = np.array([res_empty[key].estim, v_lst_all[-1] - res_empty[key].estim])
            ls_matrix = self._get_ls_matrix(c_n, self.v_[key])
            ls_solution = np.linalg.pinv(kkt_matrix).dot(ls_matrix)
            self.vimp_[key] = ls_solution[0:(self.num_p + 1), :]
            self.lambdas_[key] = ls_solution[(self.num_p + 1):ls_solution.shape[0], :]
        return self.vimp_

    ## calculate standard errors
    def get_ses(self):
        for key in self.res_keys:
            v_ic_array = np.vstack([self.v_ics_[key][0], np.stack(self.v_ics_[key][1:], axis = 0)])
            self.ics_[key] = shapley_influence_function(self.Z_, self.z_counts_, self.W_, self.v_[key], self.vimp_[key], self.G_, v_ic_array)
            self.ses_[key] = np.array([shapley_se(self.ics_[key], idx, self.gamma) for idx in range(self.num_p + 1)])
        return self.ses_

    ## calculate the ci based on the estimate and the standard error
    def get_cis(self, level = 0.95):
        ## get alpha from the level
        a = (1 - level) / 2.
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)
        ## create it
        for key in self.res_keys:
            self.cis_[key] = self.vimp_[key] + np.outer((self.ses_[key]), fac)
        return self.cis_
