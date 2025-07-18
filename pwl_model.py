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

