# pwl_data2.py
import numpy as np
import torch
from torch.utils.data import Dataset

def get_nodes():
    # node_id,   x,    y
    return np.array([
        [1, 0.0, 0.0],   # lower-left corner
        [2, 1.0, 0.0],   # lower-right corner
        [3, 1.0, 1.0],   # upper-right corner
        [4, 0.0, 1.0],   # upper-left corner
        [5, 0.4, 0.4],   # center point
    ])

def get_triangles():
    # tri_id,  node1, node2, node3
    return np.array([
        [1, 1, 2, 5],
        [2, 2, 3, 5],
        [3, 3, 4, 5],
        [4, 4, 1, 5],
    ])

def f(x, y):
    """Dome‐shaped surface function."""
    return 0.07 * (22 - 8*x**2 - 10*y**2) + 0.14

class TriangularPWLData(Dataset):
    """
    Dataset of points inside each triangle of a fixed mesh,
    with linear interpolation of Z at the triangle’s vertices.

    Signature matches PWLData for drop-in compatibility.
    """
    def __init__(self,
                 breakpoints_x=None,    # ignored
                 breakpoints_y=None,    # ignored
                 n_samples=1000,        # points per triangle
                 inject=None,           # ignored
                 seed=0,                # RNG seed
                 in_dim=None):          # ignored
        super().__init__()

        # === 1) Load mesh ===
        nodes = get_nodes()[:, 1:]                   # (5,2)
        tris  = get_triangles()[:, 1:].astype(int) - 1  # (4,3)

        # === 2) Compute Z at each node ===
        xs, ys = nodes[:,0], nodes[:,1]
        zs = f(xs, ys)  # (5,)

        # === 3) Compute plane‐parameters (a,b,c) per triangle ===
        n_tri = len(tris)
        slopes     = np.zeros((n_tri, 2), dtype=np.float32)  # a,b
        intercepts = np.zeros( n_tri,    dtype=np.float32)  # c
        for i, tri in enumerate(tris):
            p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
            z0, z1, z2 = zs[tri[0]], zs[tri[1]], zs[tri[2]]
            # solve [ [x0,y0,1], [x1,y1,1], [x2,y2,1] ] @ [a,b,c] = [z0,z1,z2]
            A = np.array([
                [p0[0], p0[1], 1.0],
                [p1[0], p1[1], 1.0],
                [p2[0], p2[1], 1.0]
            ], dtype=np.float32)
            b_vec = np.array([z0, z1, z2], dtype=np.float32)
            a, b, c = np.linalg.solve(A, b_vec)
            slopes[i]     = [a, b]
            intercepts[i] = c

        # store for NetPWL initialization
        self.slopes     = slopes        # shape (4,2)
        self.intercepts = intercepts    # shape (4,)

        # === 4) Sample points via barycentric coords ===
        rng = np.random.default_rng(seed)
        all_pts = []
        for tri in tris:
            p0, p1, p2 = nodes[tri]
            z0, z1, z2 = zs[tri]

            u = rng.random(n_samples)
            v = rng.random(n_samples)
            mask = (u + v) > 1
            u[mask] = 1 - u[mask]
            v[mask] = 1 - v[mask]
            w0 = 1 - u - v
            w1 = u
            w2 = v

            xy = (w0[:,None]*p0 +
                  w1[:,None]*p1 +
                  w2[:,None]*p2)           # (n_samples,2)
            z  = w0*z0 + w1*z1 + w2*z2    # (n_samples,)

            all_pts.append(np.hstack([xy, z.reshape(-1,1)]))

        data = np.vstack(all_pts).astype(np.float32)  # (4*n_samples,3)
        self.x = torch.from_numpy(data[:, :2])
        self.y = torch.from_numpy(data[:,  2:]).reshape(-1,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
