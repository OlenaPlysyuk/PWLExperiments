import torch

class NetPWL(torch.nn.Module):
    """
    First layer: initialized to the true PWL planes (1D or 2D),
    optionally fixed, then ReLU → hidden layers → output.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 slopes,             # either shape (n_segments,) or (n_segments, in_dim)
                 intercepts,         # shape (n_segments,)
                 fix_first_layer: bool = True):
        super().__init__()

        # how many “pieces” (1D: segments, 2D: triangles)
        # slopes may be a list/array or a 2-D array
        slopes_arr = torch.tensor(slopes, dtype=torch.float32)
        intercepts_arr = torch.tensor(intercepts, dtype=torch.float32)
        n_segments = slopes_arr.shape[0]

        # First layer: from in_dim → n_segments
        self.first = torch.nn.Linear(in_dim, n_segments)
        with torch.no_grad():
            # build weight matrix and bias vector
            w = torch.zeros(n_segments, in_dim, dtype=torch.float32)
            b = torch.zeros(n_segments,     dtype=torch.float32)

            if slopes_arr.ndim == 1:
                # classic 1-D PWL: only first coordinate has slope
                w[:, 0] = slopes_arr
            else:
                # 2-D PWL: each row i is the [A, B] plane-coeffs for triangle i
                # slopes_arr shape is (n_segments, in_dim)
                w.copy_(slopes_arr)

            # same for intercept
            b.copy_(intercepts_arr)

            self.first.weight.copy_(w)
            self.first.bias.copy_(b)

        # optionally freeze that layer
        if fix_first_layer:
            for p in self.first.parameters():
                p.requires_grad = False

        # build the rest of the network
        layers = [ self.first,
                   torch.nn.ReLU(),
                   torch.nn.Linear(n_segments, hidden_dim),
                   torch.nn.ReLU() ]
        for _ in range(n_hidden_layers - 1):
            layers += [ torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU() ]
        layers.append(torch.nn.Linear(hidden_dim, 1))

        self.layers  = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.layers(x)











'''
import torch

class NetPWL(torch.nn.Module):
    """
    First layer: either fixed to true PWL or trainable linear initialized with true slopes,
    then ReLU → trainable hidden layers → output
    """
    def __init__(self, in_dim, hidden_dim, n_hidden_layers, slopes, intercepts,
                 fix_first_layer=True):
        super().__init__()
        n_segments = len(slopes)

        # first layer initialization
        self.first = torch.nn.Linear(in_dim, n_segments)
        with torch.no_grad():
            w = torch.zeros(n_segments, in_dim)
            b = torch.zeros(n_segments)
            for i, (m, c) in enumerate(zip(slopes, intercepts)):
                w[i, 0] = float(m)
                b[i]    = float(c)
            self.first.weight.copy_(w)
            self.first.bias.copy_(b)
        if fix_first_layer:
            for p in self.first.parameters():
                p.requires_grad = False

        # build head: first layer → ReLU → hidden layers → output
        layers = [self.first, torch.nn.ReLU(),
                  torch.nn.Linear(n_segments, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden_dim, 1))
        self.layers = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.layers(x)
'''
