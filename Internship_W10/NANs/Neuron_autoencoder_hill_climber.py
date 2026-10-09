"""
Neuron-Autoencoder trained with a RANDOM HILL-CLIMBER, following the structure of
Larry Bull's "neuron autoencoder networks" (NAN) paper instead of SGD/backprop.

What changed vs the original (SGD) script
------------------------------------------
The paper's learning procedure (no backprop at all):
  * Weights are seeded uniformly in [-1, 1].
  * A "learning cycle" = pick ONE weight at random and add a random amount in [-R, +R].
  * With prob 0.5 the cycle is an AUTOENCODING cycle (a hidden-neuron weight is chosen,
    objective = that neuron's decoder/reconstruction MSE), otherwise a TASK cycle
    (an output-layer weight is chosen, objective = the task loss).
  * The new configuration is KEPT iff its MSE is strictly reduced vs the current one;
    ties are broken at random. Otherwise the change is reverted.
So both the reconstruction training AND the read-out training are now this single
gradient-free hill-climber. All the SGD optimisers, backward() calls, MSELoss /
CrossEntropyLoss optimisation and the separately-SGD-trained probe read-out are gone.

Adaptations forced by MNIST (called out so you can tune / revert them)
----------------------------------------------------------------------
  1. Objective per cycle is evaluated on a FIXED subset of the training set
     (EVAL_SIZE images, drawn once) rather than the whole 60k every cycle. The
     hill-climber only needs current vs candidate scored on the *same* data, and a
     fixed subset keeps that comparison exact while staying fast. Set EVAL_SIZE = None
     to use the full training set (Bull-faithful, but slow).
  2. Bull's task is regression (MSE). MNIST is classification, so TASK_LOSS defaults to
     cross-entropy. Set TASK_LOSS = "mse" for one-hot + sigmoid MSE, closer to the paper.
  3. Bull runs 10,000 cycles on a 1000-point, small-N task. MNIST has 784-dim inputs and
     ~47k weights per net, so it needs many more cycles; scale CYCLES_PER_EPOCH to taste.

Two latent bugs in the original that would crash / mislead the instrumentation are
fixed here and flagged inline: (A) effective_rank was covarying over samples, not
features; (B) alt_effective_rank passed a non-existent `device=` kwarg to np.eye.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
EIG_FLOOR   = 1e-12
N           = 20          # number of hidden neurons (each an autoencoder)
IN_DIM      = 784
N_CLASSES   = 10

R            = 1.0        # perturbation range [-R, +R]            (paper: R = 1.0)
WEIGHT_INIT  = 1.0        # weights seeded uniform in [-WEIGHT_INIT, +WEIGHT_INIT]
P_AUTOENC    = 0.5        # prob a cycle is an autoencoding cycle   (paper: 0.5)
TASK_LOSS    = "ce"       # "ce" (classification) or "mse" (Bull-style one-hot MSE)
EVAL_SIZE    = 2000       # fixed hill-climber eval subset size; None -> full train set

epochs           = 20     # instrumentation checkpoints scaffold (kept from original)
TEST_EVERY       = 4
CYCLES_PER_EPOCH = 10000   # hill-climb cycles run between successive `epoch` values
SEEDS            = range(1)

batch_size = 64
data_dir   = "./data"
SAVE_DIR   = "Internship_W10/NANs/images"

print("Using PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("Using GPU, device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("No GPU found, using CPU instead.")
    device = torch.device("cpu")

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
print("data_dir =", data_dir)
train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=ToTensor())
test_dataset  = datasets.MNIST(data_dir, train=False,             transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


def _normalize(img):
    """Same per-image normalisation used during AE training."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std  = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std


def build_eval_set(dataset, size):
    """A fixed, normalised, flattened subset used to score every hill-climb cycle."""
    n = len(dataset) if size is None else min(size, len(dataset))
    idx = torch.randperm(len(dataset))[:n]
    imgs   = torch.stack([dataset[j][0] for j in idx])          # [n, 1, 28, 28]
    labels = torch.tensor([dataset[j][1] for j in idx])         # [n]
    X = _normalize(imgs).view(n, -1).to(device)                 # [n, IN_DIM]
    y = labels.to(device)
    return X, y


# --------------------------------------------------------------------------------------
# Model  (no autograd; parameters are perturbed directly)
# --------------------------------------------------------------------------------------
class NeuronAutoencoder(nn.Module):
    """
    encoder        : IN_DIM -> N          (hidden-neuron incoming weights + bias)
    decoder_*      : per-neuron autoencoder;  neuron i reconstructs x as h_i*w_i + b_i
    task_*         : the output layer (hidden N -> N_CLASSES), Bull's "task learning" head
    """
    def __init__(self, n_neurons=N, in_dim=IN_DIM, n_classes=N_CLASSES, init_range=WEIGHT_INIT):
        super().__init__()
        self.encoder         = nn.Linear(in_dim, n_neurons)
        self.decoder_weights = nn.Parameter(torch.empty(n_neurons, in_dim))
        self.decoder_bias    = nn.Parameter(torch.empty(n_neurons, in_dim))
        self.task_weights    = nn.Parameter(torch.empty(n_classes, n_neurons))
        self.task_bias       = nn.Parameter(torch.empty(n_classes))

        for p in (self.encoder.weight, self.encoder.bias,
                  self.decoder_weights, self.decoder_bias,
                  self.task_weights, self.task_bias):
            nn.init.uniform_(p, -init_range, init_range)   # seed in [-1, 1]
            p.requires_grad_(False)                         # gradient-free learning

    @torch.no_grad()
    def features(self, x_flat):
        """Hidden activations h = sigmoid(W x + b)  ->  [batch, N]."""
        return torch.sigmoid(x_flat @ self.encoder.weight.t() + self.encoder.bias)

    @torch.no_grad()
    def forward(self, x):
        x = nn.Flatten()(x)
        h = self.features(x)
        decoded = (h.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))         # [batch, N, IN_DIM]
        return decoded, x, h


# --------------------------------------------------------------------------------------
# Objectives evaluated by the hill-climber
# --------------------------------------------------------------------------------------
@torch.no_grad()
def neuron_recon_mse(net, X, i):
    """Reconstruction MSE for the single neuron i experiencing the change."""
    h_i   = torch.sigmoid(X @ net.encoder.weight[i] + net.encoder.bias[i])   # [M]
    recon = h_i.unsqueeze(1) * net.decoder_weights[i].unsqueeze(0) + net.decoder_bias[i].unsqueeze(0)
    return ((X - recon) ** 2).mean()


@torch.no_grad()
def task_loss(net, X, y):
    logits = net.features(X) @ net.task_weights.t() + net.task_bias           # [M, N_CLASSES]
    if TASK_LOSS == "ce":
        return F.cross_entropy(logits, y)
    onehot = F.one_hot(y, logits.size(1)).float()
    return ((torch.sigmoid(logits) - onehot) ** 2).mean()                     # Bull-style MSE


# --------------------------------------------------------------------------------------
# Random hill-climber
# --------------------------------------------------------------------------------------
def sample_autoenc_weight(net, i):
    """Pick one scalar weight of neuron i, uniformly over its enc+decoder weights."""
    d = IN_DIM
    k = random.randrange(3 * d + 1)
    if k < d:            return net.encoder.weight, (i, k)
    k -= d
    if k < 1:            return net.encoder.bias,   (i,)
    k -= 1
    if k < d:            return net.decoder_weights, (i, k)
    k -= d
    return net.decoder_bias, (i, k)


def sample_task_weight(net):
    """Pick one scalar weight of the output layer, uniformly."""
    k = random.randrange(N_CLASSES * N + N_CLASSES)
    if k < N_CLASSES * N:
        return net.task_weights, (k // N, k % N)
    return net.task_bias, (k - N_CLASSES * N,)


@torch.no_grad()
def run_cycles(net, X, y, n_cycles, r=R, p_auto=P_AUTOENC):
    """Run n_cycles Bull-style learning cycles; keep-if-improved, ties random, else revert."""
    n_auto = n_auto_acc = n_task = n_task_acc = 0
    for _ in range(n_cycles):
        if random.random() < p_auto:                         # ----- autoencoding cycle -----
            n_auto += 1
            i = random.randrange(N)                          # neuron chosen for adjustment
            cur = neuron_recon_mse(net, X, i).item()
            t, idx = sample_autoenc_weight(net, i)
            old = t[idx].item()
            t[idx] = old + random.uniform(-r, r)
            cand = neuron_recon_mse(net, X, i).item()
            if cand < cur or (cand == cur and random.random() < 0.5):
                n_auto_acc += 1                              # keep
            else:
                t[idx] = old                                 # revert
        else:                                                # -------- task cycle ----------
            n_task += 1
            cur = task_loss(net, X, y).item()
            t, idx = sample_task_weight(net)
            old = t[idx].item()
            t[idx] = old + random.uniform(-r, r)
            cand = task_loss(net, X, y).item()
            if cand < cur or (cand == cur and random.random() < 0.5):
                n_task_acc += 1                              # keep
            else:
                t[idx] = old                                 # revert
    print(f"  {n_cycles} cycles | autoenc kept {n_auto_acc}/{n_auto} | task kept {n_task_acc}/{n_task}")
    return dict(n_auto=n_auto, n_auto_acc=n_auto_acc, n_task=n_task, n_task_acc=n_task_acc)


# --------------------------------------------------------------------------------------
# Metrics / instrumentation
# --------------------------------------------------------------------------------------
def effective_rank(mat):
    # mat: [samples, features]. FIX: covary over FEATURES (transpose), not samples.
    cov = torch.cov(mat.T)
    eigenvalues = torch.clip(torch.linalg.eigvalsh(cov), 0.0, None)
    total = torch.sum(eigenvalues)
    if total <= EIG_FLOOR:
        return float("nan")
    p = eigenvalues / total
    p = p[p > EIG_FLOOR]
    entropy = torch.sum(-(p * torch.log(p)))
    return torch.exp(entropy)


def alt_effective_rank(mat, eig_floor=EIG_FLOOR, jitter=1e-6):
    mat = np.atleast_2d(mat)
    cov = np.atleast_2d(np.cov(mat))
    cov = cov + jitter * np.eye(cov.shape[0])          # FIX: np.eye has no `device=` kwarg
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = eigvals.sum()
    if total <= eig_floor:
        return float("nan")
    p = eigvals / total
    p = p[p > eig_floor]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


@torch.no_grad()
def test_encoder(net, loader):
    net.eval()
    acts = []
    for img, _ in loader:
        img = img.to(device)
        acts.append(net.features(_normalize(img).view(img.size(0), -1)))
    return effective_rank(torch.cat(acts, dim=0)).item()


@torch.no_grad()
def test_task_head(net, loader):
    net.eval()
    total_correct = total = 0
    for img, target in loader:
        img, target = img.to(device), target.to(device)
        logits = net.features(_normalize(img).view(img.size(0), -1)) @ net.task_weights.t() + net.task_bias
        total_correct += (logits.argmax(1) == target).sum().item()
        total += target.numel()
    acc = total_correct / total
    print(f"  Task-head test accuracy: {100 * acc:.1f}%")
    return acc


# --------------------------------------------------------------------------------------
# Plotting helpers (unchanged)
# --------------------------------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

PARAM_SPECS = [
    ("encoder",      lambda net: net.encoder.weight),
    ("decoder",      lambda net: net.decoder_weights),
    ("decoder_bias", lambda net: net.decoder_bias),
]


def to_numpy(t):
    return t.detach().cpu().numpy()


def save_line_plot(x, series, title, ylabel, filename, xlabel="Epoch"):
    plt.figure()
    for label, y, yerr in series:
        if yerr is None:
            plt.plot(x, y, label=label)
        else:
            plt.errorbar(x, y, yerr, label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if any(label is not None for label, _, _ in series):
        plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename)); plt.close()


def save_feature_grid(mat, title, epoch, rows=int(N / 5), cols=5, shape=(28, 28)):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        f = mat[i].reshape(shape)
        m = np.abs(f).max() + 1e-9
        ax.imshow(f, cmap="seismic", vmin=-m, vmax=m)
        ax.set_title(f"neuron {i}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, f"{title}_{epoch}.png")); plt.close(fig)


# --------------------------------------------------------------------------------------
# Training run
# --------------------------------------------------------------------------------------
for seed in SEEDS:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = NeuronAutoencoder().to(device)
    X_eval, y_eval = build_eval_set(train_dataset, EVAL_SIZE)
    print(net)

    metrics    = defaultdict(list)
    param_hist = {name: defaultdict(list) for name, _ in PARAM_SPECS}

    for e in range(epochs + 1):
        if e % TEST_EVERY == 0:
            print(f"[seed {seed}] checkpoint at epoch {e}")
            metrics["epoch"].append(e)
            metrics["r_eff"].append(test_encoder(net, test_loader))

            for name, getter in PARAM_SPECS:
                mat = to_numpy(getter(net))
                if mat.ndim == 2:
                    save_feature_grid(mat, name, e)
                param_hist[name]["mean"].append(mat.mean())
                param_hist[name]["std"].append(mat.std())
                param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                param_hist[name]["eff_rank"].append(alt_effective_rank(mat))

            metrics["accuracy"].append(test_task_head(net, test_loader))

        if e == epochs:
            break

        run_cycles(net, X_eval, y_eval, CYCLES_PER_EPOCH)   # <-- hill-climber replaces SGD

    x = metrics["epoch"]
    save_line_plot(x, [(None, metrics["accuracy"], None)],
                   "Read-out Test Accuracy across Epochs",
                   "Test Set Accuracy", "Test_accuracy.png")
    save_line_plot(x, [(None, metrics["r_eff"], None)],
                   "Encoder Activation Effective Rank across Epochs",
                   "R_eff", "eff_rank.png")
    save_line_plot(x, [(f"mean |{name}|", param_hist[name]["abs_mean"], None)
                       for name, _ in PARAM_SPECS],
                   "Mean of Absolute Parameters across Epochs", "", "Absolute_Means.png")
    save_line_plot(x, [(f"mean {name}", param_hist[name]["mean"], param_hist[name]["std"])
                       for name, _ in PARAM_SPECS],
                   "Mean (±std) of Signed Parameters across Epochs", "", "Signed_Means.png")
    save_line_plot(x, [(f"{name} r_eff", param_hist[name]["eff_rank"], None)
                       for name, _ in PARAM_SPECS],
                   "Effective Rank of Parameters across Epochs", "R_eff", "params_R_eff.png")