"""
Neuron-Autoencoder (NaN) redundancy experiment
================================================
Compares how activation-space redundancy evolves during reconstruction training
across four input geometries:

    NK           - near-isotropic random binary landscape (Bull's data), regression
    MNIST_raw    - raw MNIST pixels in [0,1], classification
    MNIST_meansub- MNIST with the training mean subtracted
    MNIST_zca    - MNIST ZCA-whitened (fit on train, reused on test)

For each condition, for each seed, an MLP hidden layer of N_HIDDEN neurons is
trained as a Neuron Autoencoder (each hidden neuron is its own 784 -> 1 -> 784
autoencoder). At every epoch (including epoch 0 = random init) we log, on the
held-out test set:

    * effective rank of the activation covariance matrix  (r_eff_cov)
    * effective rank of the activation correlation matrix (r_eff_corr)
    * spectral redundancy from each  (R_spec = 1 - r_eff / N_HIDDEN)
    * reconstruction loss (raw MSE, NOT normalised -- only compare within a condition)
    * redundancy index   : mean |cosine| between encoder rows  (-> 1 == redundancy collapse)
    * mean-collapse index: mean cosine( decoder bias row , input mean )
                           (-> 1 == bias absorbing the dataset mean; NaN when the
                            input mean is ~0, i.e. NK / mean-sub / whitened)

At CHECKPOINT epochs a FRESH linear readout (random init) is trained to
convergence on the FROZEN hidden activations (task gradients never touch the
hidden layer). Probe is fit on train activations and evaluated on test, so the
number reported is a *generalisation* measure:
    * MNIST conditions  -> test classification accuracy
    * NK                -> test regression MSE (reported separately)

Also computed once per condition: effective rank of the INPUT covariance, which
anchors the isotropy story (NK & ZCA high, raw/mean-sub MNIST low).

Everything is averaged across seeds and saved as plots (mean +/- std bands).

Runtime notes
-------------
The dominant cost is the reconstruction training. One seed, one learning rate,
four conditions, 20 epochs is a few minutes on GPU and roughly 10-40 min on CPU.
To sweep learning rates just add entries to LR_LIST -- the whole grid re-runs and
an extra "final r_eff vs lr" summary plot is produced. Start with SEEDS=[0] and a
single LR, then widen both once the pipeline behaves.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  CONFIG  (everything you would want to change lives here)
# --------------------------------------------------------------------------- #
SEEDS      = [0]                 # e.g. [0, 1, 2, 3, 4] for the rigorous run
LR_LIST    = [100.0]              # e.g. [0.01, 0.1, 0.5, 1.29, 3.0, 10.0] to sweep
RECON_EPOCHS = 20
CHECKPOINTS  = [0, 5, 10, 15, 20]  # epochs at which the readout probe is run
N_HIDDEN   = 20
IN_DIM     = 784
BATCH_SIZE = 64
RECON_MOMENTUM = 0.0             # plain SGD, matching the original script

# NK landscape / dataset (kept identical to Bull's fitness semantics)
NK_N        = 784               # = IN_DIM so the architecture is identical across conditions
NK_K        = 2
NK_M_TRAIN  = 60000
NK_M_TEST   = 10000
NK_DATA_SEED = 12345            # fixed so NK data is a fixed dataset (like MNIST); seeds vary init only

# ZCA whitening
ZCA_EPS = 1e-2                  # eigenvalue floor for the whitening transform

# Linear-probe (readout) training -- trained to convergence
PROBE_LR        = 0.1
PROBE_MOMENTUM  = 0.9
PROBE_BATCH     = 256
PROBE_MAX_EPOCHS = 200
PROBE_TOL       = 1e-5          # stop when train loss improves by less than this
PROBE_PATIENCE  = 5

EIG_FLOOR = 1e-12              # clamp for effective-rank eigenvalues
DATA_DIR  = "./data"
RESULTS_DIR = "./results"

CONDITIONS = ["NK", "MNIST_raw", "MNIST_meansub", "MNIST_zca"]
COLORS = {"NK": "#1b9e77", "MNIST_raw": "#d95f02",
          "MNIST_meansub": "#7570b3", "MNIST_zca": "#e7298a"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  REPRODUCIBILITY
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random_gen_seed = int(seed)
    np.random.seed(random_gen_seed)
    torch.manual_seed(random_gen_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_gen_seed)


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def generate_nk_dataset(N, K, M_train, M_test, gen_seed):
    """Vectorised NK landscape + dataset, faithful to Bull's fitness definition.

    Genome bits in {0,1} index the fitness tables; the MLP inputs are the
    +/-1 recoding (genome*2 - 1). Fitness is the per-gene table lookup summed
    over genes and normalised by N -> a scalar regression target.
    """
    rng = np.random.RandomState(gen_seed)
    fitness_table = rng.rand(N, 2 ** (K + 1))          # one table per gene
    powers = 2 ** np.arange(K, -1, -1)                 # MSB is the gene itself

    def make(M):
        genome = rng.randint(2, size=(M, N)).astype(np.int64)  # {0,1}
        total = np.zeros(M, dtype=np.float64)
        for i in range(N):                              # 784 vectorised passes
            neigh = (i + np.arange(K + 1)) % N          # gene i + K partners
            idx = genome[:, neigh] @ powers             # (M,) row into table i
            total += fitness_table[i, idx]
        fitness = (total / N).astype(np.float32)        # (M,)
        inputs = (genome * 2 - 1).astype(np.float32)    # {-1,+1} -> MLP input
        return torch.from_numpy(inputs), torch.from_numpy(fitness)

    Xtr, ytr = make(M_train)
    Xte, yte = make(M_test)
    return Xtr, ytr, Xte, yte


def load_mnist():
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=ToTensor())
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=ToTensor())
    Xtr = tr.data.float().div(255.0).view(-1, IN_DIM)
    Xte = te.data.float().div(255.0).view(-1, IN_DIM)
    ytr = tr.targets.long()
    yte = te.targets.long()
    return Xtr, ytr, Xte, yte


def fit_zca(X, eps):
    """Fit a ZCA whitening transform on X (rows = samples). Returns (mean, W)."""
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)                # symmetric
    evals = torch.clamp(evals, min=0.0)
    W = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
    return mean, W


def prepare_condition(name, mnist_cache):
    """Return a dict with train/test tensors, task metadata, and the input mean."""
    if name == "NK":
        Xtr, ytr, Xte, yte = generate_nk_dataset(
            NK_N, NK_K, NK_M_TRAIN, NK_M_TEST, NK_DATA_SEED)
        return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                    task="reg", out_dim=1, in_mean=Xtr.mean(dim=0))

    Xtr, ytr, Xte, yte = mnist_cache
    if name == "MNIST_raw":
        pass
    elif name == "MNIST_meansub":
        m = Xtr.mean(dim=0, keepdim=True)
        Xtr, Xte = Xtr - m, Xte - m
    elif name == "MNIST_zca":
        mean, W = fit_zca(Xtr, ZCA_EPS)
        Xtr = (Xtr - mean) @ W
        Xte = (Xte - mean) @ W
    else:
        raise ValueError(name)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                task="clf", out_dim=10, in_mean=Xtr.mean(dim=0))


# --------------------------------------------------------------------------- #
#  MODEL
# --------------------------------------------------------------------------- #
class NeuronAutoencoder(nn.Module):
    """Each hidden neuron is its own 784 -> 1 -> 784 autoencoder."""
    def __init__(self, n_neurons=N_HIDDEN, in_dim=IN_DIM):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def activations(self, x):
        return torch.sigmoid(self.encoder(x))            # (B, n_neurons)

    def forward(self, x):
        a = self.activations(x)                           # (B, H)
        decoded = (a.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))       # (B, H, in_dim)
        return decoded, x, a


# --------------------------------------------------------------------------- #
#  METRICS
# --------------------------------------------------------------------------- #
def effective_rank(mat):
    """exp(spectral entropy) of a symmetric PSD matrix. NaN-safe."""
    evals = np.linalg.eigvalsh(mat)                       # real, ascending
    evals = np.clip(evals.real, 0.0, None)
    total = evals.sum()
    if total <= EIG_FLOOR:
        return float("nan")                              # degenerate / all-constant
    p = evals / total
    p = p[p > EIG_FLOOR]                                  # 0*log0 -> 0
    entropy = -(p * np.log(p)).sum()
    return float(math.exp(entropy))


def spectral_redundancy(r_eff, D):
    if math.isnan(r_eff):
        return float("nan")
    return 1.0 - r_eff / D


def safe_corrcoef(A):
    """Correlation matrix of columns of A (rows = samples), NaN-safe."""
    C = np.corrcoef(A.T)
    C = np.nan_to_num(C, nan=0.0)                         # dead (0-variance) neurons
    np.fill_diagonal(C, 1.0)
    return C


def encoder_row_cosine(model):
    """Mean |cosine| between distinct encoder rows -> redundancy index."""
    W = model.encoder.weight.detach().cpu().numpy()       # (H, in_dim)
    norm = np.linalg.norm(W, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    Wn = W / norm
    C = Wn @ Wn.T                                          # (H, H) cosines
    H = C.shape[0]
    off = ~np.eye(H, dtype=bool)
    return float(np.abs(C[off]).mean())


def decoder_bias_mean_cosine(model, in_mean):
    """Mean cosine( decoder bias row , input mean ) -> mean-collapse index.

    NaN when the input mean is ~0 (NK, mean-subtracted, whitened): there is no
    mean to collapse onto, which is itself part of the story.
    """
    mean = in_mean.detach().cpu().numpy().ravel()
    mn = np.linalg.norm(mean)
    if mn < 1e-6:
        return float("nan")
    B = model.decoder_bias.detach().cpu().numpy()          # (H, in_dim)
    bn = np.linalg.norm(B, axis=1)
    ok = bn > 1e-12
    if not ok.any():
        return 0.0
    cos = (B[ok] @ mean) / (bn[ok] * mn)
    return float(cos.mean())


# --------------------------------------------------------------------------- #
#  TRAINING / EVAL HELPERS
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_activations(model, X, batch=4096):
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        out.append(model.activations(X[i:i + batch].to(device)).cpu())
    return torch.cat(out, 0).numpy()


@torch.no_grad()
def recon_loss_eval(model, X, batch=1024):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device)
        decoded, _, _ = model(xb)
        target = xb.unsqueeze(1).expand_as(decoded)
        total += ((decoded - target) ** 2).mean().item() * xb.shape[0]
        n += xb.shape[0]
    return total / n


def train_recon_one_epoch(model, X, optimizer, generator):
    model.train()
    perm = torch.randperm(X.shape[0], generator=generator)
    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[perm[i:i + BATCH_SIZE]].to(device)
        decoded, _, _ = model(xb)
        target = xb.unsqueeze(1).expand_as(decoded)
        loss = ((decoded - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_probe(feat_tr, y_tr, feat_te, y_te, task, out_dim):
    """Fresh linear readout, SGD to convergence on frozen features.
    Fit on train, evaluate on test. Returns accuracy (clf) or MSE (reg)."""
    Xtr = torch.from_numpy(feat_tr).to(device)
    Xte = torch.from_numpy(feat_te).to(device)
    ytr = y_tr.to(device)
    yte = y_te.to(device)

    probe = nn.Linear(Xtr.shape[1], out_dim).to(device)   # random init each call
    opt = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    loss_fn = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()

    prev, wait = float("inf"), 0
    gen = torch.Generator(device="cpu")
    for _ in range(PROBE_MAX_EPOCHS):
        probe.train()
        perm = torch.randperm(Xtr.shape[0], generator=gen)
        running = 0.0
        for i in range(0, Xtr.shape[0], PROBE_BATCH):
            idx = perm[i:i + PROBE_BATCH]
            out = probe(Xtr[idx])
            if task == "clf":
                loss = loss_fn(out, ytr[idx])
            else:
                loss = loss_fn(out.squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * idx.numel()
        running /= Xtr.shape[0]
        if prev - running < PROBE_TOL:                    # convergence check
            wait += 1
            if wait >= PROBE_PATIENCE:
                break
        else:
            wait = 0
        prev = running

    probe.eval()
    with torch.no_grad():
        out = probe(Xte)
        if task == "clf":
            return (out.argmax(1) == yte).float().mean().item()   # accuracy
        return ((out.squeeze(1) - yte) ** 2).mean().item()        # MSE


# --------------------------------------------------------------------------- #
#  ONE RUN  (condition x lr x seed)
# --------------------------------------------------------------------------- #
def measure_epoch(model, data):
    """All per-epoch activation-space metrics on the test set."""
    A = get_activations(model, data["Xte"])              # (n_test, H)
    cov = np.cov(A.T)
    corr = safe_corrcoef(A)
    r_cov = effective_rank(cov)
    r_corr = effective_rank(corr)
    return dict(
        reff_cov=r_cov,
        reff_corr=r_corr,
        rspec_cov=spectral_redundancy(r_cov, N_HIDDEN),
        rspec_corr=spectral_redundancy(r_corr, N_HIDDEN),
        recon=recon_loss_eval(model, data["Xte"]),
        enc_cos=encoder_row_cosine(model),
        bias_mean_cos=decoder_bias_mean_cosine(model, data["in_mean"]),
    )


def probe_at_checkpoint(model, data):
    feat_tr = get_activations(model, data["Xtr"])
    feat_te = get_activations(model, data["Xte"])
    return train_probe(feat_tr, data["ytr"], feat_te, data["yte"],
                       data["task"], data["out_dim"])


def run_one(condition, lr, seed, mnist_cache):
    set_seed(seed)
    data = prepare_condition(condition, mnist_cache)

    # input covariance effective rank (isotropy anchor) -- once
    input_reff = effective_rank(np.cov(data["Xtr"].numpy().T))

    model = NeuronAutoencoder().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=RECON_MOMENTUM)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    keys = ["reff_cov", "reff_corr", "rspec_cov", "rspec_corr",
            "recon", "enc_cos", "bias_mean_cos"]
    epoch_log = {k: [] for k in keys}
    probe_log = {}

    for epoch in range(RECON_EPOCHS + 1):               # 0..RECON_EPOCHS
        if epoch > 0:
            train_recon_one_epoch(model, data["Xtr"], optimizer, gen)
        m = measure_epoch(model, data)
        for k in keys:
            epoch_log[k].append(m[k])
        if epoch in CHECKPOINTS:
            probe_log[epoch] = probe_at_checkpoint(model, data)
        print(f"    [{condition} lr={lr} seed={seed}] epoch {epoch:2d}  "
              f"r_eff_cov={m['reff_cov']:.2f}  recon={m['recon']:.4g}"
              + (f"  probe={probe_log[epoch]:.4g}" if epoch in CHECKPOINTS else ""))

    return dict(epoch_log={k: np.array(v) for k, v in epoch_log.items()},
                probe=np.array([probe_log[c] for c in CHECKPOINTS]),
                input_reff=input_reff, task=data["task"])


# --------------------------------------------------------------------------- #
#  AGGREGATION + PLOTTING
# --------------------------------------------------------------------------- #
def aggregate(per_seed_runs):
    """per_seed_runs: list (over seeds) of run_one dicts for one (condition,lr)."""
    stack = lambda key: np.stack([r["epoch_log"][key] for r in per_seed_runs], 0)
    agg = {}
    for k in per_seed_runs[0]["epoch_log"]:
        arr = stack(k)
        agg[k] = (np.nanmean(arr, 0), np.nanstd(arr, 0))
    probe = np.stack([r["probe"] for r in per_seed_runs], 0)
    agg["probe"] = (np.nanmean(probe, 0), np.nanstd(probe, 0))
    agg["input_reff"] = float(np.nanmean([r["input_reff"] for r in per_seed_runs]))
    agg["task"] = per_seed_runs[0]["task"]
    return agg


def _band(ax, x, mean, std, color, label):
    ax.plot(x, mean, color=color, label=label, lw=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def plot_epoch_metric(results_lr, key, ylabel, title, fname, conditions=None):
    conditions = conditions or CONDITIONS
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in conditions:
        if c not in results_lr:
            continue
        mean, std = results_lr[c][key]
        if np.all(np.isnan(mean)):
            continue
        _band(ax, x, mean, std, COLORS[c], c)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_recon(results_lr, fname):
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CONDITIONS:
        mean, std = results_lr[c]["recon"]
        _band(ax, x, mean, std, COLORS[c], c)
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction epoch")
    ax.set_ylabel("test reconstruction MSE (raw, log scale)")
    ax.set_title("Reconstruction loss (compare WITHIN a condition only)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_probe_mnist(results_lr, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ["MNIST_raw", "MNIST_meansub", "MNIST_zca"]:
        mean, std = results_lr[c]["probe"]
        ax.errorbar(CHECKPOINTS, mean, yerr=std, color=COLORS[c],
                    label=c, lw=2, marker="o", capsize=3)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Linear readout accuracy (MNIST conditions)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_probe_nk(results_lr, fname):
    mean, std = results_lr["NK"]["probe"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(CHECKPOINTS, mean, yerr=std, color=COLORS["NK"],
                label="NK", lw=2, marker="o", capsize=3)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel("test regression MSE")
    ax.set_title("Linear readout MSE (NK, regression)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_input_reff(results_lr, fname):
    conds = [c for c in CONDITIONS if c in results_lr]
    vals = [results_lr[c]["input_reff"] for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(conds, vals, color=[COLORS[c] for c in conds])
    ax.axhline(N_HIDDEN, ls="--", color="gray", label=f"hidden width = {N_HIDDEN}")
    ax.set_ylabel("effective rank of INPUT covariance")
    ax.set_title("Input isotropy anchor (higher = more isotropic)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def make_plots_for_lr(results_lr, lr):
    tag = f"lr{lr:g}".replace(".", "p")
    d = RESULTS_DIR
    plot_epoch_metric(results_lr, "reff_cov", "effective rank (cov)",
                      f"Activation effective rank (covariance), lr={lr:g}",
                      f"{d}/reff_cov_{tag}.png")
    plot_epoch_metric(results_lr, "reff_corr", "effective rank (corr)",
                      f"Activation effective rank (correlation), lr={lr:g}",
                      f"{d}/reff_corr_{tag}.png")
    plot_epoch_metric(results_lr, "rspec_cov", "spectral redundancy (cov)",
                      f"Spectral redundancy (covariance), lr={lr:g}",
                      f"{d}/rspec_cov_{tag}.png")
    plot_epoch_metric(results_lr, "rspec_corr", "spectral redundancy (corr)",
                      f"Spectral redundancy (correlation), lr={lr:g}",
                      f"{d}/rspec_corr_{tag}.png")
    plot_recon(results_lr, f"{d}/recon_{tag}.png")
    plot_probe_mnist(results_lr, f"{d}/probe_mnist_{tag}.png")
    plot_probe_nk(results_lr, f"{d}/probe_nk_{tag}.png")
    plot_epoch_metric(results_lr, "enc_cos", "mean |cosine| of encoder rows",
                      f"Redundancy index (encoder-row alignment), lr={lr:g}",
                      f"{d}/redundancy_index_{tag}.png")
    plot_epoch_metric(results_lr, "bias_mean_cos", "cos(decoder bias, input mean)",
                      f"Mean-collapse index, lr={lr:g}",
                      f"{d}/meancollapse_index_{tag}.png",
                      conditions=["MNIST_raw"])   # only raw MNIST has a non-zero mean
    plot_input_reff(results_lr, f"{d}/input_reff_{tag}.png")


def plot_lr_summary(all_results):
    """Final-epoch r_eff (cov) vs learning rate, per condition."""
    if len(LR_LIST) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CONDITIONS:
        finals = [all_results[lr][c]["reff_cov"][0][-1] for lr in LR_LIST]
        ax.plot(LR_LIST, finals, marker="o", color=COLORS[c], label=c, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate"); ax.set_ylabel("final effective rank (cov)")
    ax.set_title("Final activation effective rank vs learning rate")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/summary_reff_vs_lr.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"device = {device}")
    print("loading MNIST ...")
    mnist_cache = load_mnist()

    all_results = {}                                     # all_results[lr][cond] = agg
    for lr in LR_LIST:
        print(f"\n=== learning rate {lr:g} ===")
        results_lr = {}
        for condition in CONDITIONS:
            print(f"  -- condition {condition} --")
            per_seed = [run_one(condition, lr, s, mnist_cache) for s in SEEDS]
            results_lr[condition] = aggregate(per_seed)
        all_results[lr] = results_lr
        make_plots_for_lr(results_lr, lr)

    plot_lr_summary(all_results)
    print(f"\nDone. Plots saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()