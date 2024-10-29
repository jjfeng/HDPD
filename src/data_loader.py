"""Loads data from a file or dataframe
Note that X denotes the conditional covariates which is named Z in the manuscript
"""

import numpy as np

from common import to_safe_prob

class DataLoader:
    def __init__(self, XW: np.ndarray, Y: np.ndarray, w_indices: np.ndarray):
        self.XW = XW
        self.Y = Y
        self.w_indices = w_indices
        self.w_mask = np.repeat([False], XW.shape[1])
        if self.w_indices.size != 0:
            self.w_mask[w_indices] = True
        self.num_n = XW.shape[0]
        self.num_p = XW.shape[1]
    
    def _get_XW(self):
        return self.XW
    
    def _get_X(self):
        return self.XW[:, ~self.w_mask]
    
    def _get_Y(self):
        return self.Y
    
    def _get_W(self):
        return self.XW[:, self.w_mask]

    def _concat_XW(self, X: np.ndarray, W: np.ndarray):
        XW = np.zeros((X.shape[0], self.num_p))
        XW[:, ~self.w_mask] = X
        XW[:, self.w_mask] = W
        return XW
