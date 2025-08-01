
import numpy as np
import torch
from torch.utils.data import Dataset

class PWLData(Dataset):
    """
    n-D inputs -> 1-D piecewise-linear outputs.
    breakpoints_x, breakpoints_y: lists of length = segments+1
    """
    def __init__(self, breakpoints_x, breakpoints_y,
                 n_samples=1000, inject=5, seed=0, in_dim=1):
        super().__init__()
        assert len(breakpoints_x) == len(breakpoints_y), \
            "breakpoints_x and breakpoints_y must match length"
        bx = np.array(breakpoints_x, dtype=np.float32)
        by = np.array(breakpoints_y, dtype=np.float32)
        order = np.argsort(bx)
        bx, by = bx[order], by[order]
        self.breakpoints_x = bx

        # compute slopes & intercepts using original formula:
        # slope_i = (y[i+1] - y[i]) / (x[i+1] - x[i])
        # intercept_i = y[i] - slope_i * x[i]
        self.slopes = (by[1:] - by[:-1]) / (bx[1:] - bx[:-1])
        self.intercepts = by[:-1] - self.slopes * bx[:-1]

        # sample uniform inputs
        rng = np.random.default_rng(seed)
        x0, xN = bx[0], bx[-1]
        X = rng.uniform(x0, xN, size=(n_samples, in_dim)).astype(np.float32)

        # inject interior breakpoints
        n_segments = len(self.slopes)
        if inject > 0 and n_segments > 1:
            internal = bx[1:-1]
            Xi = np.repeat(internal, inject).reshape(-1, 1)
            Xi = np.tile(Xi, (1, in_dim))
            X = np.vstack([X, Xi])

        # always include endpoints
        Xe = np.array([[x0]*in_dim, [xN]*in_dim], dtype=np.float32)
        X = np.vstack([X, Xe])

        # compute outputs from first coordinate
        x_primary = X[:, 0]
        bins = bx[1:-1]
        seg_idx = np.digitize(x_primary, bins, right=False)
        y = self.slopes[seg_idx] * x_primary + self.intercepts[seg_idx]

        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

