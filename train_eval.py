import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, loaders, optimizer, scheduler, epochs,
                l1_lambda, early_stop_patience, min_delta, outdir):
    train_loader, val_loader, _ = loaders
    history = {'train': [], 'val': []}
    best_loss = float('inf')
    best_state = None
    no_imp = 0

    for epoch in range(1, epochs+1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = model.loss_fn(preds, yb)
            reg = l1_lambda * sum(p.abs().sum() for p in model.parameters())
            (loss + reg).backward()
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
        t_loss /= len(train_loader.dataset)
        history['train'].append(t_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                loss = model.loss_fn(model(xb), yb)
                reg = l1_lambda * sum(p.abs().sum() for p in model.parameters())
                v_loss += (loss + reg).item() * xb.size(0)
        v_loss /= len(val_loader.dataset)
        history['val'].append(v_loss)

        scheduler.step(v_loss)
        if v_loss < best_loss - min_delta:
            best_loss, best_state, no_imp = v_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} train_loss={t_loss:.4e} val_loss={v_loss:.4e}")

    model.load_state_dict(best_state)
    _save_loss_plot(history, len(history['train'])-1, outdir)
    return model, history


def _save_loss_plot(history, early_epoch, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(history['train'], label='train')
    plt.plot(history['val'],   label='val')
    plt.axvline(early_epoch, linestyle='--', label='early_stop')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, 'loss.png')
    plt.savefig(path)
    print(f"Saved loss plot to {path}")


def evaluate_model(model, loader):
    mse, max_err = 0.0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb)
            mse += model.loss_fn(p, yb).item() * xb.size(0)
            max_err = max(max_err, (p - yb).abs().max().item())
    mse /= len(loader.dataset)
    print(f"Test MSE={mse:.4e}, max error={max_err:.4e}")
    return mse, max_err


def plot_1d(ds, outdir):
    bx = ds.breakpoints_x
    xs = np.linspace(bx[0], bx[-1], 500)
    seg_idx = np.digitize(xs, bx[1:-1], right=False)
    ys = ds.slopes[seg_idx] * xs + ds.intercepts[seg_idx]

    Xs = ds.x.numpy()[:, 0]
    Ys = ds.y.numpy().squeeze()

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label='True PWL', linewidth=2)
    plt.scatter(Xs, Ys, s=10, alpha=0.3, label='Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('1D: True PWL + Samples')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, 'plot1d.png')
    plt.savefig(path)
    print(f"Saved 1D plot to {path}")


def plot_2d(ds, outdir):
    X = ds.x.numpy()
    Y = ds.y.numpy().squeeze()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y, s=12, alpha=0.7)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('y')
    ax.set_title('2D Input → 1D PWL Output')
    plt.tight_layout()
    path = os.path.join(outdir, 'plot2d.png')
    plt.savefig(path)
    print(f"Saved 2D plot to {path}")
    # show interactive window for 3D rotation (requires an interactive backend)
    plt.show()


def analyze_activation_patterns(model, ds):
    X_cpu = ds.x
    lin_layers = [l for l in model.layers if isinstance(l, torch.nn.Linear)]
    hidden_layers = lin_layers[:-1]
    W_h = [l.weight for l in hidden_layers]
    b_h = [l.bias for l in hidden_layers]
    all_masks = []

    with torch.no_grad():
        for x in X_cpu:
            x_curr = x.unsqueeze(0)
            masks = []
            for W, b in zip(W_h, b_h):
                Z = x_curr @ W.T + b
                mask = (Z > 0).int().squeeze(0).tolist()
                masks.append(tuple(mask))
                x_curr = torch.relu(Z)
            all_masks.append(tuple(masks))

    patterns = sorted(set(all_masks))
    print(f"Found {len(patterns)} unique activation patterns across {len(hidden_layers)} hidden layers:")
    for pat in patterns:
        count = all_masks.count(pat)
        print(f"Pattern {pat}: count={count}")
        V = None
        for i, mask in enumerate(pat):
            W = W_h[i]
            m_vec = torch.tensor(mask, dtype=torch.float32)
            if i == 0:
                V = torch.diag(m_vec) @ W
            else:
                V = torch.diag(m_vec) @ (W @ V)
            slope = float(V.sum())
            active_idxs = [j for j, bit in enumerate(mask) if bit]
            print(f"  Hidden Layer {i+1}: active indices={active_idxs}, slope={slope:.6f}")
        W_out = lin_layers[-1].weight
        out_vec = (W_out @ V).squeeze(0)
        formatted = ", ".join(f"{s:.6f}" for s in out_vec.tolist())
        print(f"  Output Layer slopes per dim: [{formatted}]")


