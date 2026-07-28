"""
Neuron-Autoencoder (NaN) lateral-inhibition sweep
=================================================
Same measurement framework as the earlier redundancy experiment, but the swept
axis is no longer the *input geometry* (NK / raw / mean-sub / ZCA). Here the data
is ALWAYS raw MNIST, and the swept variable is INHIB_SCALER: the strength of a
spectral-redundancy loss added to reconstruction as a lateral-inhibition /
diversity pressure.

Training objective (per batch):

    loss = reconstruction_MSE  +  inhib_scaler * R_spec(batch activations)

where R_spec = 1 - r_eff / N is the spectral redundancy of the *neuron*
covariance across the batch (r_eff = exp(spectral entropy)). Minimising R_spec
maximises effective rank, i.e. pushes the hidden neurons to span more independent
directions -> the encoder rows diversify. inhib_scaler = 0 recovers the plain
reconstruction baseline.

For each inhib_scaler, for each seed, an MLP hidden layer of N_HIDDEN neurons is
trained as a Neuron Autoencoder (each hidden neuron is its own 784 -> 1 -> 784
autoencoder). At every epoch (including epoch 0 = random init) we log, on the
held-out test set, the same metrics as before:

    * effective rank of the activation covariance / correlation matrix
    * spectral redundancy from each  (R_spec = 1 - r_eff / N_HIDDEN)
    * reconstruction loss (raw MSE)
    * redundancy index   : mean |cosine| between encoder rows
    * mean-collapse index: cos(decoder bias row, input mean)  (raw MNIST has a
                           strong non-zero mean, so this stays meaningful)

At CHECKPOINT epochs a FRESH linear readout is trained to convergence on the
FROZEN hidden activations (task gradients never touch the hidden layer) and its
test *classification accuracy* is recorded (MNIST is a classification task, so
there is no regression-MSE probe here).

The learning-rate sweep (LR_LIST) is unchanged from the previous experiment.

Removed vs the previous experiment (no longer meaningful here):
    * NK condition + regression-MSE probe + standard-MLP baseline
    * mean-subtracted / ZCA-whitened MNIST conditions and the ZCA machinery
    * the per-condition input-effective-rank bar chart (input is identical for
      every series now, so it is just computed and printed once)

Added:
    * a differentiable R_spec training loss (the lateral inhibition)
    * a "diversity pressure vs task performance" summary: final R_spec and final
      probe accuracy vs inhib_scaler
    * optional encoder-filter grids per (lr, inhib_scaler) to eyeball the
      diversification qualitatively
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
SEEDS        = [0]               # e.g. [0, 1, 2, 3, 4] for the rigorous run
LR_LIST      = [100.0]           # same sweep as before, e.g. [0.1, 1.0, 10.0, 100.0]
INHIB_SCALERS = [0.0, 0.5, 1.0, 2.0]   # the swept axis: lateral-inhibition strength
RECON_EPOCHS = 20
CHECKPOINTS  = [0, 5, 10, 15, 20]      # epochs at which the readout probe is run
N_HIDDEN     = 20
IN_DIM       = 784
BATCH_SIZE   = 64
RECON_MOMENTUM = 0.0             # plain SGD, matching the original script
INHIB_MIN_BATCH = 2              # cov needs >=2 samples; skip the inhib term below this

# Qualitative check: save encoder-filter grids for one seed per (lr, inhib_scaler)
SAVE_FILTERS      = True
FILTER_SEED_INDEX = -1           # which entry of SEEDS to visualise (default: last)

# Linear-probe (readout) training -- trained to convergence
PROBE_LR        = 0.1
PROBE_MOMENTUM  = 0.9
PROBE_BATCH     = 256
PROBE_MAX_EPOCHS = 200
PROBE_TOL       = 1e-5           # stop when train loss improves by less than this
PROBE_PATIENCE  = 5

EIG_FLOOR   = 1e-12              # clamp for effective-rank eigenvalues
DATA_DIR    = "./data"
RESULTS_DIR = "./results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  COLOURS  (one per inhib_scaler, light -> dark as inhibition increases)
# --------------------------------------------------------------------------- #
def make_colors(scalers):
    cmap = plt.get_cmap("viridis")
    lo, hi = min(scalers), max(scalers)
    if hi == lo:
        return {scalers[0]: cmap(0.5)}
    return {s: cmap(0.1 + 0.8 * (s - lo) / (hi - lo)) for s in scalers}


COLORS = make_colors(INHIB_SCALERS)


def series_label(s):
    return f"inhib={s:g}"


# --------------------------------------------------------------------------- #
#  REPRODUCIBILITY
# --------------------------------------------------------------------------- #
def set_seed(seed):
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- #
#  DATA  (raw MNIST only)
# --------------------------------------------------------------------------- #
def load_mnist_raw():
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=ToTensor())
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=ToTensor())
    Xtr = tr.data.float().div(255.0).view(-1, IN_DIM)
    Xte = te.data.float().div(255.0).view(-1, IN_DIM)
    ytr = tr.targets.long()
    yte = te.targets.long()
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
#  METRICS  (measurement-side; numpy, no gradient)
# --------------------------------------------------------------------------- #
def effective_rank(mat):
    """exp(spectral entropy) of a symmetric PSD matrix. NaN-safe."""
    evals = np.linalg.eigvalsh(mat)
    evals = np.clip(evals.real, 0.0, None)
    total = evals.sum()
    if total <= EIG_FLOOR:
        return float("nan")
    p = evals / total
    p = p[p > EIG_FLOOR]
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
    C = Wn @ Wn.T
    H = C.shape[0]
    off = ~np.eye(H, dtype=bool)
    return float(np.abs(C[off]).mean())


def decoder_bias_mean_cosine(model, in_mean):
    """Mean cosine( decoder bias row , input mean ) -> mean-collapse index."""
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
#  LATERAL-INHIBITION LOSS  (training-side; differentiable, torch)
# --------------------------------------------------------------------------- #
def spectral_redundancy_loss(activations):
    """Differentiable R_spec = 1 - r_eff/N on the batch NEURON-covariance.

    activations: (B, H). Returns a scalar tensor, or None if the batch is too
    small / degenerate to define a covariance. Minimising this term spreads the
    neurons across more independent directions (the diversity pressure).
    """
    if activations.shape[0] < INHIB_MIN_BATCH:
        return None
    cov = torch.cov(activations.T)                         # (H, H) across the batch
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    total = evals.sum()
    if total <= EIG_FLOOR:
        return None
    p = evals / total
    p = p[p > EIG_FLOOR]
    entropy = -(p * torch.log(p)).sum()
    r_eff = torch.exp(entropy)
    return 1.0 - r_eff / N_HIDDEN


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


def train_recon_one_epoch(model, X, optimizer, generator, inhib_scaler):
    """One reconstruction epoch with the optional inhibition term. Returns the
    epoch-average (recon_loss, inhibition_R_spec) on the TRAIN batches."""
    model.train()
    perm = torch.randperm(X.shape[0], generator=generator)
    run_recon, run_inhib, nb = 0.0, 0.0, 0
    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[perm[i:i + BATCH_SIZE]].to(device)
        decoded, feats, acts = model(xb)
        target = feats.unsqueeze(1).expand_as(decoded)
        recon = ((decoded - target) ** 2).mean()

        loss = recon
        inhib_val = 0.0
        if inhib_scaler != 0.0:
            r_spec = spectral_redundancy_loss(acts)
            if r_spec is not None:
                loss = recon + inhib_scaler * r_spec
                inhib_val = float(r_spec.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_recon += float(recon.detach())
        run_inhib += inhib_val
        nb += 1
    nb = max(nb, 1)
    return run_recon / nb, run_inhib / nb


def train_probe(feat_tr, y_tr, feat_te, y_te, out_dim):
    """Fresh linear readout, SGD to convergence on frozen features. Fit on train,
    evaluate on test. Returns test classification accuracy."""
    Xtr = torch.from_numpy(feat_tr).to(device)
    Xte = torch.from_numpy(feat_te).to(device)
    ytr = y_tr.to(device)
    yte = y_te.to(device)

    probe = nn.Linear(Xtr.shape[1], out_dim).to(device)   # random init each call
    opt = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=PROBE_MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()

    prev, wait = float("inf"), 0
    gen = torch.Generator(device="cpu")
    for _ in range(PROBE_MAX_EPOCHS):
        probe.train()
        perm = torch.randperm(Xtr.shape[0], generator=gen)
        running = 0.0
        for i in range(0, Xtr.shape[0], PROBE_BATCH):
            idx = perm[i:i + PROBE_BATCH]
            out = probe(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * idx.numel()
        running /= Xtr.shape[0]
        if prev - running < PROBE_TOL:
            wait += 1
            if wait >= PROBE_PATIENCE:
                break
        else:
            wait = 0
        prev = running

    probe.eval()
    with torch.no_grad():
        out = probe(Xte)
        return (out.argmax(1) == yte).float().mean().item()   # accuracy


# --------------------------------------------------------------------------- #
#  MEASUREMENT + ONE RUN  (inhib_scaler x lr x seed)
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
    return train_probe(feat_tr, data["ytr"], feat_te, data["yte"], data["out_dim"])


def save_encoder_filters(model, fname, title):
    """Grid of the 20 encoder rows as 28x28 filters (symmetric seismic colormap)."""
    W = model.encoder.weight.detach().cpu().numpy()       # (H, 784)
    H = W.shape[0]
    ncols = 5
    nrows = int(math.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    for i, ax in enumerate(np.array(axes).flat):
        if i < H:
            filt = W[i].reshape(28, 28)
            m = np.abs(filt).max() or 1.0
            ax.imshow(filt, cmap="seismic", vmin=-m, vmax=m)
            ax.set_title(f"n{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def run_one(inhib_scaler, lr, seed, data, input_reff):
    set_seed(seed)

    model = NeuronAutoencoder().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=RECON_MOMENTUM)
    gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))

    keys = ["reff_cov", "reff_corr", "rspec_cov", "rspec_corr",
            "recon", "enc_cos", "bias_mean_cos"]
    epoch_log = {k: [] for k in keys}
    probe_log = {}

    for epoch in range(RECON_EPOCHS + 1):               # 0..RECON_EPOCHS
        inhib_avg = float("nan")
        if epoch > 0:
            _, inhib_avg = train_recon_one_epoch(
                model, data["Xtr"], optimizer, gen, inhib_scaler)
        m = measure_epoch(model, data)
        for k in keys:
            epoch_log[k].append(m[k])
        if epoch in CHECKPOINTS:
            probe_log[epoch] = probe_at_checkpoint(model, data)
        print(f"    [inhib={inhib_scaler:g} lr={lr:g} seed={seed}] epoch {epoch:2d}  "
              f"rspec_cov={m['rspec_cov']:.3f}  recon={m['recon']:.4g}"
              + (f"  train_Rspec={inhib_avg:.3f}" if epoch > 0 else "")
              + (f"  probe_acc={probe_log[epoch]:.4g}" if epoch in CHECKPOINTS else ""))

    # Optional qualitative check: encoder filters for the chosen seed.
    if SAVE_FILTERS and seed == SEEDS[FILTER_SEED_INDEX]:
        tag = f"lr{lr:g}_inhib{inhib_scaler:g}".replace(".", "p")
        save_encoder_filters(
            model, f"{RESULTS_DIR}/filters_{tag}.png",
            f"Encoder filters  (lr={lr:g}, inhib={inhib_scaler:g}, seed={seed})")

    return dict(epoch_log={k: np.array(v) for k, v in epoch_log.items()},
                probe=np.array([probe_log[c] for c in CHECKPOINTS]),
                input_reff=input_reff, task=data["task"])


# --------------------------------------------------------------------------- #
#  AGGREGATION + PLOTTING
# --------------------------------------------------------------------------- #
def aggregate(per_seed_runs):
    """per_seed_runs: list (over seeds) of run_one dicts for one (inhib,lr)."""
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


def plot_epoch_metric(results_lr, key, ylabel, title, fname):
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in INHIB_SCALERS:
        if s not in results_lr:
            continue
        mean, std = results_lr[s][key]
        if np.all(np.isnan(mean)):
            continue
        _band(ax, x, mean, std, COLORS[s], series_label(s))
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_recon(results_lr, fname):
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in INHIB_SCALERS:
        if s not in results_lr:
            continue
        mean, std = results_lr[s]["recon"]
        _band(ax, x, mean, std, COLORS[s], series_label(s))
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction epoch")
    ax.set_ylabel("test reconstruction MSE (raw, log scale)")
    ax.set_title("Reconstruction loss vs inhibition strength")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_probe_accuracy(results_lr, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in INHIB_SCALERS:
        if s not in results_lr:
            continue
        mean, std = results_lr[s]["probe"]
        ax.errorbar(CHECKPOINTS, mean, yerr=std, color=COLORS[s],
                    label=series_label(s), lw=2, marker="o", capsize=3)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Linear readout accuracy (raw MNIST) vs inhibition strength")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_redundancy_vs_probe(results_lr, rkey, xlabel, title, fname):
    """Probe test accuracy (y) against spectral redundancy (x), paired at the
    checkpoint epochs. Points connected in checkpoint order trace the training
    trajectory; error bars are cross-seed std in each axis."""
    ckpt = np.array(CHECKPOINTS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in INHIB_SCALERS:
        if s not in results_lr:
            continue
        rmean, rstd = results_lr[s][rkey]
        pmean, pstd = results_lr[s]["probe"]
        x, xe = rmean[ckpt], rstd[ckpt]
        ax.errorbar(x, pmean, xerr=xe, yerr=pstd, color=COLORS[s],
                    label=series_label(s), lw=1.5, marker="o", capsize=3, alpha=0.9)
        for xi, yi, e in zip(x, pmean, CHECKPOINTS):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(f"e{e}", (xi, yi), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[s])
    ax.set_xlabel(xlabel); ax.set_ylabel("linear-probe test accuracy")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_inhib_summary(results_lr, lr, fname):
    """The money plot: final spectral redundancy (diversity pressure achieved)
    and final probe accuracy (task performance) vs inhibition strength."""
    scalers = [s for s in INHIB_SCALERS if s in results_lr]
    r_mean = [results_lr[s]["rspec_cov"][0][-1] for s in scalers]
    r_std  = [results_lr[s]["rspec_cov"][1][-1] for s in scalers]
    a_mean = [results_lr[s]["probe"][0][-1] for s in scalers]
    a_std  = [results_lr[s]["probe"][1][-1] for s in scalers]

    c_red, c_acc = "#d95f02", "#1b9e77"
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.errorbar(scalers, r_mean, yerr=r_std, color=c_red, marker="o", lw=2,
                 capsize=3, label="final spectral redundancy (cov)")
    ax1.set_xlabel("inhibition scaler")
    ax1.set_ylabel("final spectral redundancy (cov)", color=c_red)
    ax1.tick_params(axis="y", labelcolor=c_red)

    ax2 = ax1.twinx()
    ax2.errorbar(scalers, a_mean, yerr=a_std, color=c_acc, marker="s", lw=2,
                 capsize=3, label="final probe accuracy")
    ax2.set_ylabel("final probe test accuracy", color=c_acc)
    ax2.tick_params(axis="y", labelcolor=c_acc)

    ax1.set_title(f"Diversity pressure vs task performance, lr={lr:g}")
    ax1.grid(alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
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
    plot_probe_accuracy(results_lr, f"{d}/probe_acc_{tag}.png")
    plot_epoch_metric(results_lr, "enc_cos", "mean |cosine| of encoder rows",
                      f"Redundancy index (encoder-row alignment), lr={lr:g}",
                      f"{d}/redundancy_index_{tag}.png")
    plot_epoch_metric(results_lr, "bias_mean_cos", "cos(decoder bias, input mean)",
                      f"Mean-collapse index, lr={lr:g}",
                      f"{d}/meancollapse_index_{tag}.png")
    plot_redundancy_vs_probe(
        results_lr, "rspec_cov",
        "spectral redundancy (cov)  =  1 - r_eff/N",
        f"Probe accuracy vs spectral redundancy (cov), lr={lr:g}",
        f"{d}/redund_vs_probe_cov_{tag}.png")
    plot_redundancy_vs_probe(
        results_lr, "rspec_corr",
        "spectral redundancy (corr)  =  1 - r_eff/N",
        f"Probe accuracy vs spectral redundancy (corr), lr={lr:g}",
        f"{d}/redund_vs_probe_corr_{tag}.png")
    plot_inhib_summary(results_lr, lr, f"{d}/inhib_summary_{tag}.png")


def plot_lr_summary(all_results):
    """Final-epoch r_eff (cov) vs learning rate, per inhib_scaler."""
    if len(LR_LIST) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in INHIB_SCALERS:
        finals = [all_results[lr][s]["reff_cov"][0][-1] for lr in LR_LIST]
        ax.plot(LR_LIST, finals, marker="o", color=COLORS[s],
                label=series_label(s), lw=2)
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
    print("loading raw MNIST ...")
    data = load_mnist_raw()

    # Input isotropy anchor -- identical for every series, so computed once.
    input_reff = effective_rank(np.cov(data["Xtr"].numpy().T))
    print(f"raw MNIST input-covariance effective rank = {input_reff:.2f} "
          f"(hidden width N = {N_HIDDEN})")

    all_results = {}                                     # all_results[lr][inhib] = agg
    for lr in LR_LIST:
        print(f"\n=== learning rate {lr:g} ===")
        results_lr = {}
        for s in INHIB_SCALERS:
            print(f"  -- inhib_scaler {s:g} --")
            per_seed = [run_one(s, lr, seed, data, input_reff) for seed in SEEDS]
            results_lr[s] = aggregate(per_seed)
        all_results[lr] = results_lr
        make_plots_for_lr(results_lr, lr)

    plot_lr_summary(all_results)
    print(f"\nDone. Plots saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
