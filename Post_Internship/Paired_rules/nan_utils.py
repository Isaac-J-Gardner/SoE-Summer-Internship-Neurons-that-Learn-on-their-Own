"""
nan_utils.py
================
Rule-agnostic infrastructure for the spiking Neuron-Autoencoder (NAN) experiments.

Everything in here is independent of *which* reconstruction learning rule you use
(backprop, or any of the four paired rules), so it can be imported and shared across
all of them. Each function that previously relied on a module-level global
(``N``, ``device``, ``p``, ``num_steps``, ``SAVE_DIR`` ...) now takes that value as an
explicit argument, so there is no hidden state.

Contents
--------
data            : get_mnist_loaders, _normalize
metrics         : spk_effective_rank, alt_effective_rank, recon_loss
readout probe   : LinearReadout, correct, train_linear, test_linear, train_readout
encoder eval    : test_encoder
recording       : record_neuron_stats
plotting        : to_numpy, save_line_plot, save_feature_grid, save_matrix
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")          # headless-safe; remove if you want interactive figures
import matplotlib.pyplot as plt

EIG_FLOOR = 1e-12              # floor used in effective-rank entropy to avoid log(0)


# ---------------------------------------------------------------------------- data
def get_mnist_loaders(data_dir="./data", batch_size=100):
    """Return (train_dataset, test_dataset, train_loader, test_loader) for MNIST."""
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def _normalize(img):
    """Per-image zero-mean / unit-std normalisation over the pixel dimension.

    Expects a (B, 1, 28, 28) batch (the extra leading dims used by the 'top image'
    grid still broadcast for top_k == 1)."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std


# ------------------------------------------------------------------------- metrics
def spk_effective_rank(spk_rec, n_neurons, eig_floor=EIG_FLOOR, jitter=1e-6):
    """Participation-ratio / entropy effective rank of the spike covariance.

    spk_rec : (T, B, H) or (totalT, H, 1) tensor of spikes.  n_neurons == H."""
    N = n_neurons
    spikes = spk_rec.permute(1, 0, 2).reshape(N, -1)      # (H, samples)
    cov = torch.cov(spikes)                               # (H, H)
    cov = cov + jitter * torch.eye(N, device=cov.device, dtype=cov.dtype)

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clip(eigvals, 0.0, None)
    total = eigvals.sum()

    # nothing spiked -> return a graph-connected zero rather than NaN
    if total <= eig_floor:
        return spikes.sum() * 0.0

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = torch.sum(-(p * torch.log(p)))
    return torch.exp(entropy)                             # effective rank


def alt_effective_rank(mat, eig_floor=EIG_FLOOR, jitter=1e-6):
    """NumPy effective rank of a 2-D weight matrix (rows treated as variables).

    Fixed vs. the original: ``np.eye`` takes no ``device``/``dtype=torch`` kwargs,
    and the size is taken from the covariance, not a global N."""
    cov = np.cov(mat)                                     # (rows, rows)
    cov = np.atleast_2d(cov)
    cov = cov + jitter * np.eye(cov.shape[0])

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= eig_floor:
        return 0.0

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = np.sum(-(p * np.log(p)))
    return float(np.exp(entropy))


def activation_effective_rank(A, eig_floor=EIG_FLOOR, jitter=1e-6):
    """Effective rank of the (H, H) covariance of an activation matrix A: (samples, H).

    Low values mean the neurons are redundant / collapsed onto a shared subspace."""
    cov = torch.cov(A.T.float())
    cov = torch.atleast_2d(cov)
    H = cov.shape[0]
    cov = cov + jitter * torch.eye(H, device=cov.device, dtype=cov.dtype)
    ev = torch.clip(torch.linalg.eigvalsh(cov), 0.0, None)
    tot = ev.sum()
    if tot <= eig_floor:
        return 0.0
    p = ev / tot
    p = p[p > eig_floor]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def recon_loss(x_recon, x, activity, gating=True):
    """Gated per-(sample, neuron) mean-squared reconstruction error.

    x_recon  : (B, H, 784) per-neuron reconstruction
    x        : (B, 784)    input
    activity : (B, H)      total spikes per neuron (the gate source)
    """
    target = x.unsqueeze(1).expand_as(x_recon)            # (B, H, 784)
    se = ((target - x_recon) ** 2).mean(dim=2)            # (B, H)
    if gating:
        gate = (activity.detach() > 0).float()            # (B, H)
        return (se * gate).sum() / gate.sum().clamp(min=1)
    return se.mean()


# -------------------------------------------------------------------- readout probe
class LinearReadout(nn.Module):
    """Frozen encoder features (fc1) + trainable linear classifier (fc2).

    fc1 is seeded with the encoder's feed-forward weights and never receives a
    gradient (the forward pass detaches before fc2), so it just applies the encoder."""
    def __init__(self, weight, n_neurons):
        super().__init__()
        self.fc1 = nn.Linear(784, n_neurons)
        self.fc1.weight = nn.Parameter(weight.detach().clone())
        self.fc2 = nn.Linear(n_neurons, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x.detach())
        return x


def correct(output, target):
    predicted_digits = output.argmax(1)
    correct_ones = (predicted_digits == target).type(torch.float)
    return correct_ones.sum().item()


def train_linear(data_loader, model, criterion, optimizer, device):
    model.train()
    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)
    total_loss = 0.0
    total_correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(_normalize(data))
        loss = criterion(output, target)
        total_loss += loss.item()                         # .item(): don't retain graph
        total_correct += correct(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")
    return train_loss


@torch.no_grad()
def test_linear(test_loader, model, criterion, device):
    model.eval()
    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)
    test_loss = 0.0
    total_correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(_normalize(data))
        test_loss += criterion(output, target).item()
        total_correct += correct(output, target)
    test_loss /= num_batches
    accuracy = total_correct / num_items
    print(f"Testset accuracy: {100 * accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return accuracy


def train_readout(net, seed, train_loader, test_loader, device, n_neurons):
    """Train a fresh linear read-out on top of the current encoder features."""
    torch.manual_seed(seed + 100)                         # identical init every probe
    readout = LinearReadout(net.encoder.fc.weight, n_neurons).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)

    prev_loss, curr_loss, i = 1.0, 0.0, 0
    while (i < 3) or (prev_loss - curr_loss > 0.01):
        prev_loss = curr_loss
        curr_loss = train_linear(train_loader, readout, criterion, optimizer, device)
        print(f"Read-out epoch {i + 1}: loss {curr_loss:.4f} "
              f"(diff {prev_loss - curr_loss:.4f})")
        i += 1
    return test_linear(test_loader, readout, criterion, device)


# -------------------------------------------------------------------- encoder eval
@torch.no_grad()
def test_encoder(network, loader, device, n_neurons):
    """Effective rank, mean firing rate, mean threshold, inhibition density."""
    network.eval()
    spikes = []
    for img, _ in loader:
        img = img.to(device)
        spk_rec, x, x_recon, activity = network(_normalize(img))
        spikes.append(torch.cat(spk_rec, dim=0))          # (T*B, H)
    spikes = torch.cat(spikes, dim=0)                     # (sum T*B, H)

    r_eff = spk_effective_rank(spikes.unsqueeze(-1), n_neurons)
    avg_rate = spikes.float().mean()
    avg_thresh = network.encoder.theta.mean()

    inhibition = network.encoder.W_inh
    inhibition_prop = (inhibition > 0).float().mean()
    inhibition_prop = inhibition_prop * n_neurons / (n_neurons - 1)   # undo zero diag
    return r_eff.item(), avg_rate.item(), avg_thresh.item(), inhibition_prop.item()


# ------------------------------------------------------------------------ recording
@torch.no_grad()
def record_neuron_stats(network, dataset, device, batch_size=100, n_neurons=None,
                        top_k=1):
    """Per-image, per-neuron record of delta_i, firings n_i and current C_i.

      deltas[j, i]   = sum_k R_ik e_ik   (back-projected recon error of neuron i)
      firings[j, i]  = total spikes neuron i emitted for image j   (n_i)
      currents[j, i] = fc(x) = sum_k Q_ik x_k + B_i                (C_i)

    Rows follow dataset order (shuffle=False), so row j <-> dataset[j]."""
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    dec0 = network.decoder[0]                             # NeuronDecoder
    R = dec0.dec_weight() if hasattr(dec0, "dec_weight") else dec0.weight  # (H, 784)
    H = R.shape[0]
    n_images = len(dataset)

    deltas = torch.empty(n_images, H)
    firings = torch.empty(n_images, H)
    currents = torch.empty(n_images, H)

    pos = 0
    for img, _ in loader:
        b = img.size(0)
        img = img.to(device)
        x_norm = _normalize(img)

        spk_rec_list, x, x_recon, activity = network(x_norm)
        e = x.unsqueeze(1) - x_recon                      # (b, H, 784) == e_ik
        delta = (e * R.unsqueeze(0)).sum(dim=2)           # (b, H)      == delta_i
        current = network.encoder.fc(x)                   # (b, H)      == C_i

        deltas[pos:pos + b] = delta.cpu()
        firings[pos:pos + b] = activity.cpu()
        currents[pos:pos + b] = current.cpu()
        pos += b

    top_by_current = currents.topk(top_k, dim=0).indices  # (top_k, H)
    top_by_firing = firings.topk(top_k, dim=0).indices    # (top_k, H)

    top_imgs_current = torch.stack([
        torch.stack([dataset[int(top_by_current[k, i])][0] for i in range(H)])
        for k in range(top_k)
    ])                                                    # (top_k, H, 1, 28, 28)
    top_imgs_current = _normalize(top_imgs_current)

    return {
        "deltas": deltas,
        "firings": firings,
        "currents": currents,
        "top_by_current": top_by_current,
        "top_by_firing": top_by_firing,
        "top_imgs_current": top_imgs_current,
    }


# ------------------------------------------------------------------------- plotting
def to_numpy(t):
    return t.detach().cpu().numpy()


def save_line_plot(x, series, title, ylabel, filename, save_dir, xlabel="Epoch"):
    """series: list of (label, y, yerr). label=None -> no legend entry;
    yerr=None -> plain line, else error bars."""
    plt.figure()
    for label, y, yerr in series:
        if yerr is None:
            plt.plot(x, y, label=label)
        else:
            plt.errorbar(x, y, yerr, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if any(label is not None for label, _, _ in series):
        plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def save_feature_grid(mat, title, epoch, save_dir, cols=5, shape=(28, 28)):
    """Show the first (rows*cols) rows of a 2-D weight matrix as images."""
    rows = int(np.ceil(mat.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle(title)
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < mat.shape[0]:
            f = mat[i].reshape(shape)
            m = np.abs(f).max() + 1e-9
            ax.imshow(f, cmap="seismic", vmin=-m, vmax=m)
            ax.set_title(f"neuron {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{title}_{epoch}.png"))
    plt.close(fig)


def save_matrix(mat, title, epoch, filename_prefix, save_dir,
                xlabel="source neuron j", ylabel="target neuron i"):
    """Show a single 2-D matrix (e.g. inhibition weights) with a colour bar."""
    fig, ax = plt.subplots()
    fig.suptitle(title)
    m = np.abs(mat).max() + 1e-9
    im = ax.imshow(mat, cmap="seismic", vmin=-m, vmax=m)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{filename_prefix}_{epoch}.png"))
    plt.close(fig)
