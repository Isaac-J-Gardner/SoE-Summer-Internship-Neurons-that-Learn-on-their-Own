"""
Neuron-Autoencoder (NaN) diversity-mechanism experiment
=======================================================
Follow-up to the redundancy experiment. Same measurement harness and structure
(per-epoch activation-space redundancy + a fresh linear readout trained to
convergence at milestone epochs, swept over learning rates), but here every
condition is RAW MNIST and the only thing that changes is the diversity
mechanism bolted onto the hidden layer:

    MNIST_baseline - plain NaN (no diversity pressure; the collapse reference)
    MNIST_ortho    - soft-orthogonality penalty on encoder rows (the DIVERSITY ORACLE:
                     a global, backprop-dependent regulariser -- tells us the ceiling,
                     not a mechanism meant to transfer to spiking)
    MNIST_kwta     - k-Winners-Take-All competition with homeostatic boosting
                     (local, activity-based, the candidate that transfers to SNNs)

The baseline is a reference line so the rescue (or lack of it) is legible; set
INCLUDE_BASELINE = False to drop it.

Per-epoch, on the held-out test set:
    * effective rank of activation covariance / correlation (r_eff_cov, r_eff_corr)
    * spectral redundancy from each (R_spec = 1 - r_eff / N_HIDDEN)
    * reconstruction loss  -- NOTE the definition differs by condition:
        baseline/ortho : mean over ALL neurons of ||decoded - x||^2
        kwta           : mean over WINNING (sample,neuron) pairs only
      so only compare within a condition.
    * redundancy index   : mean |cosine| between encoder rows (weight-space; -> 1 at collapse)
    * mean-collapse index: cos(decoder bias row, input mean) (all conditions share the
                           raw-MNIST mean, so this is meaningful for all three here)
    * active fraction    : fraction of hidden neurons that fire on >=1% of test inputs
                           (kWTA health / dead-unit check; ~1.0 for baseline & ortho)

At CHECKPOINT epochs a fresh random linear readout is trained to convergence on the
FROZEN hidden representation actually used by each mechanism (sigmoid activations for
baseline/ortho, the sparse top-k code for kWTA), fit on train and evaluated on test.
All conditions are classification, so readout performance is test accuracy on one plot.

Runtime: same order as the first experiment. Start with SEEDS=[0] and one LR; widen
LR_LIST to sweep (an extra "final r_eff vs lr" summary plot appears automatically).

Key NEW knobs (flagged, will need a little tuning): LAMBDA_ORTHO, KWTA_K, KWTA_BOOST.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
SEEDS      = [0]                 # e.g. [0, 1, 2, 3, 4] for the rigorous run
LR_LIST    = [0.01, 0.1, 1.0, 10.0, 100.0]              # e.g. [0.01, 0.1, 0.5, 1.29, 3.0, 10.0] to sweep
RECON_EPOCHS = 20
CHECKPOINTS  = [0, 5, 10, 15, 20]
N_HIDDEN   = 20
IN_DIM     = 784
BATCH_SIZE = 64
RECON_MOMENTUM = 0.0

INCLUDE_BASELINE = True          # plain raw-MNIST reference line; False to drop it

# --- diversity-mechanism knobs (the new hyperparameters) ---
LAMBDA_ORTHO = 0.05              # strength of the encoder-row cos^2 penalty (TUNE ME)
KWTA_K       = 5                 # winners per input, out of N_HIDDEN (sparsity = K/N_HIDDEN)
# Homeostasis: an adaptive per-neuron bias on the PRE-activation logits (intrinsic
# plasticity / adaptive threshold). It grows for under-firing neurons until they win,
# so it reliably revives dead units and mirrors the adaptive-threshold homeostasis
# you'll use in the spiking phase. Set KWTA_BOOST = 0.0 to disable (dead-unit risk).
KWTA_BOOST   = 0.02             # homeostatic adaptation rate (TUNE ME)
KWTA_WINRATE_EMA = 0.02         # EMA rate for tracking per-neuron win frequency

# Linear-probe (readout) -- trained to convergence
PROBE_LR, PROBE_MOMENTUM = 0.1, 0.9
PROBE_BATCH, PROBE_MAX_EPOCHS = 256, 200
PROBE_TOL, PROBE_PATIENCE = 1e-5, 5

EIG_FLOOR = 1e-12
DATA_DIR  = "./data"
RESULTS_DIR = "./results_diversity"

INCLUDE = (["MNIST_baseline"] if INCLUDE_BASELINE else [])
CONDITIONS = INCLUDE + ["MNIST_ortho", "MNIST_kwta"]
MODE_OF = {"MNIST_baseline": "baseline", "MNIST_ortho": "ortho", "MNIST_kwta": "kwta"}
COLORS = {"MNIST_baseline": "#7f7f7f", "MNIST_ortho": "#d95f02", "MNIST_kwta": "#1b9e77"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# --------------------------------------------------------------------------- #
#  DATA  (raw MNIST for every condition)
# --------------------------------------------------------------------------- #
def load_mnist():
    from torchvision import datasets
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr = tr.data.float().div(255.0).view(-1, IN_DIM)
    Xte = te.data.float().div(255.0).view(-1, IN_DIM)
    return Xtr, tr.targets.long(), Xte, te.targets.long()


def prepare(mnist_cache):
    Xtr, ytr, Xte, yte = mnist_cache
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                task="clf", out_dim=10, in_mean=Xtr.mean(dim=0))


# --------------------------------------------------------------------------- #
#  MODEL
# --------------------------------------------------------------------------- #
class NeuronAutoencoder(nn.Module):
    """Each hidden neuron is its own 784 -> 1 -> 784 autoencoder, with a
    selectable diversity mechanism on the hidden representation."""
    def __init__(self, mode="baseline", n_neurons=N_HIDDEN, in_dim=IN_DIM,
                 k=KWTA_K, boost=KWTA_BOOST, winrate_ema=KWTA_WINRATE_EMA):
        super().__init__()
        self.mode = mode
        self.encoder = nn.Linear(in_dim, n_neurons)
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))
        # k-WTA state
        self.k = k
        self.boost = boost
        self.winrate_ema = winrate_ema
        self.target_rate = k / n_neurons
        self.register_buffer("win_rate", torch.full((n_neurons,), k / n_neurons))
        # adaptive per-neuron excitability (learned threshold); part of the model,
        # so it is used at eval too -- it is not reset between train and test
        self.register_buffer("adaptive_bias", torch.zeros(n_neurons))

    def _kwta_mask(self, logits, training):
        # winners chosen on logits shifted by the homeostatic adaptive bias
        scores = logits + self.adaptive_bias.unsqueeze(0)
        _, topi = torch.topk(scores, self.k, dim=1)
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        if training:
            with torch.no_grad():
                self.win_rate.mul_(1 - self.winrate_ema).add_(
                    self.winrate_ema * mask.mean(dim=0))
                # raise excitability of under-firing neurons, lower it for over-firing
                self.adaptive_bias.add_(self.boost * (self.target_rate - self.win_rate))
        return mask

    def activations(self, x, training):
        logits = self.encoder(x)
        a = torch.sigmoid(logits)
        if self.mode == "kwta":
            a = a * self._kwta_mask(logits, training)     # gated sigmoid; losers -> 0
        return a                                          # (B, H)

    def forward(self, x, training):
        a = self.activations(x, training)
        decoded = (a.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))       # (B, H, in_dim)
        return decoded, x, a


# --------------------------------------------------------------------------- #
#  METRICS  (unchanged from the first experiment)
# --------------------------------------------------------------------------- #
def effective_rank(mat):
    evals = np.clip(np.linalg.eigvalsh(mat).real, 0.0, None)
    total = evals.sum()
    if total <= EIG_FLOOR:
        return float("nan")
    p = evals / total
    p = p[p > EIG_FLOOR]
    return float(math.exp(-(p * np.log(p)).sum()))


def spectral_redundancy(r_eff, D):
    return float("nan") if math.isnan(r_eff) else 1.0 - r_eff / D


def safe_corrcoef(A):
    with np.errstate(invalid="ignore", divide="ignore"):   # dead neurons -> 0 variance
        C = np.nan_to_num(np.corrcoef(A.T), nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def encoder_row_cosine(model):
    W = model.encoder.weight.detach().cpu().numpy()
    norm = np.linalg.norm(W, axis=1, keepdims=True); norm[norm < 1e-12] = 1e-12
    Wn = W / norm
    C = Wn @ Wn.T
    off = ~np.eye(C.shape[0], dtype=bool)
    return float(np.abs(C[off]).mean())


def decoder_bias_mean_cosine(model, in_mean):
    mean = in_mean.detach().cpu().numpy().ravel()
    mn = np.linalg.norm(mean)
    if mn < 1e-6:
        return float("nan")
    B = model.decoder_bias.detach().cpu().numpy()
    bn = np.linalg.norm(B, axis=1); ok = bn > 1e-12
    if not ok.any():
        return 0.0
    return float(((B[ok] @ mean) / (bn[ok] * mn)).mean())


# --------------------------------------------------------------------------- #
#  TRAIN / EVAL
# --------------------------------------------------------------------------- #
def train_loss(model, xb):
    decoded, _, a = model(xb, training=True)
    se = (decoded - xb.unsqueeze(1)) ** 2                 # (B, H, D)
    if model.mode == "kwta":
        per_neuron = se.mean(dim=2)                       # (B, H)
        winners = (a > 0).float()
        return (per_neuron * winners).sum() / winners.sum().clamp(min=1.0)
    recon = se.mean()
    if model.mode == "ortho":
        Wn = F.normalize(model.encoder.weight, dim=1)
        G = Wn @ Wn.T
        off = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
        return recon + LAMBDA_ORTHO * (G[off] ** 2).mean()
    return recon


def train_recon_one_epoch(model, X, optimizer, generator):
    model.train()
    perm = torch.randperm(X.shape[0], generator=generator)
    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[perm[i:i + BATCH_SIZE]].to(device)
        loss = train_loss(model, xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()


@torch.no_grad()
def get_activations(model, X, batch=4096):
    model.eval()
    out = [model.activations(X[i:i + batch].to(device), training=False).cpu()
           for i in range(0, X.shape[0], batch)]
    return torch.cat(out, 0).numpy()


@torch.no_grad()
def recon_loss_eval(model, X, batch=1024):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device)
        decoded, _, a = model(xb, training=False)
        se = (decoded - xb.unsqueeze(1)) ** 2
        if model.mode == "kwta":
            winners = (a > 0).float()
            l = (se.mean(dim=2) * winners).sum() / winners.sum().clamp(min=1.0)
        else:
            l = se.mean()
        total += l.item() * xb.shape[0]; n += xb.shape[0]
    return total / n


def train_probe(feat_tr, y_tr, feat_te, y_te, task, out_dim):
    Xtr = torch.from_numpy(feat_tr).to(device); Xte = torch.from_numpy(feat_te).to(device)
    ytr = y_tr.to(device); yte = y_te.to(device)
    probe = nn.Linear(Xtr.shape[1], out_dim).to(device)
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
            loss = loss_fn(out, ytr[idx]) if task == "clf" else loss_fn(out.squeeze(1), ytr[idx])
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
        if task == "clf":
            return (out.argmax(1) == yte).float().mean().item()
        return ((out.squeeze(1) - yte) ** 2).mean().item()


# --------------------------------------------------------------------------- #
#  ONE RUN
# --------------------------------------------------------------------------- #
def measure_epoch(model, data):
    A = get_activations(model, data["Xte"])
    r_cov = effective_rank(np.cov(A.T))
    r_corr = effective_rank(safe_corrcoef(A))
    active_frac = float(((A > 0).mean(axis=0) >= 0.01).mean())
    return dict(
        reff_cov=r_cov, reff_corr=r_corr,
        rspec_cov=spectral_redundancy(r_cov, N_HIDDEN),
        rspec_corr=spectral_redundancy(r_corr, N_HIDDEN),
        recon=recon_loss_eval(model, data["Xte"]),
        enc_cos=encoder_row_cosine(model),
        bias_mean_cos=decoder_bias_mean_cosine(model, data["in_mean"]),
        active_frac=active_frac,
    )


def probe_at_checkpoint(model, data):
    ftr = get_activations(model, data["Xtr"]); fte = get_activations(model, data["Xte"])
    return train_probe(ftr, data["ytr"], fte, data["yte"], data["task"], data["out_dim"])


def run_one(condition, lr, seed, mnist_cache):
    set_seed(seed)
    data = prepare(mnist_cache)
    model = NeuronAutoencoder(mode=MODE_OF[condition]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=RECON_MOMENTUM)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    keys = ["reff_cov", "reff_corr", "rspec_cov", "rspec_corr",
            "recon", "enc_cos", "bias_mean_cos", "active_frac"]
    epoch_log = {k: [] for k in keys}
    probe_log = {}
    for epoch in range(RECON_EPOCHS + 1):
        if epoch > 0:
            train_recon_one_epoch(model, data["Xtr"], optimizer, gen)
        m = measure_epoch(model, data)
        for k in keys:
            epoch_log[k].append(m[k])
        if epoch in CHECKPOINTS:
            probe_log[epoch] = probe_at_checkpoint(model, data)
        print(f"    [{condition} lr={lr} seed={seed}] ep{epoch:2d}  "
              f"r_eff_cov={m['reff_cov']:.2f}  enc_cos={m['enc_cos']:.3f}  "
              f"act={m['active_frac']:.2f}  recon={m['recon']:.4g}"
              + (f"  acc={probe_log[epoch]:.3f}" if epoch in CHECKPOINTS else ""))
    return dict(epoch_log={k: np.array(v) for k, v in epoch_log.items()},
                probe=np.array([probe_log[c] for c in CHECKPOINTS]),
                task=data["task"])


# --------------------------------------------------------------------------- #
#  AGGREGATE + PLOT
# --------------------------------------------------------------------------- #
def aggregate(runs):
    agg = {}
    for k in runs[0]["epoch_log"]:
        arr = np.stack([r["epoch_log"][k] for r in runs], 0)
        agg[k] = (np.nanmean(arr, 0), np.nanstd(arr, 0))
    p = np.stack([r["probe"] for r in runs], 0)
    agg["probe"] = (np.nanmean(p, 0), np.nanstd(p, 0))
    return agg


def _band(ax, x, mean, std, color, label):
    ax.plot(x, mean, color=color, label=label, lw=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def plot_epoch_metric(res, key, ylabel, title, fname):
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CONDITIONS:
        mean, std = res[c][key]
        if np.all(np.isnan(mean)):
            continue
        _band(ax, x, mean, std, COLORS[c], c)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_recon(res, fname):
    x = np.arange(RECON_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CONDITIONS:
        _band(ax, x, *res[c]["recon"], COLORS[c], c)
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction epoch")
    ax.set_ylabel("test reconstruction MSE (log; def. differs for kWTA)")
    ax.set_title("Reconstruction loss (compare WITHIN a condition only)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def plot_probe(res, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CONDITIONS:
        mean, std = res[c]["probe"]
        ax.errorbar(CHECKPOINTS, mean, yerr=std, color=COLORS[c],
                    label=c, lw=2, marker="o", capsize=3)
    ax.set_xlabel("reconstruction epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Linear readout accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def make_plots_for_lr(res, lr):
    tag = f"lr{lr:g}".replace(".", "p"); d = RESULTS_DIR
    plot_epoch_metric(res, "reff_cov", "effective rank (cov)",
                      f"Activation effective rank (covariance), lr={lr:g}",
                      f"{d}/reff_cov_{tag}.png")
    plot_epoch_metric(res, "reff_corr", "effective rank (corr)",
                      f"Activation effective rank (correlation), lr={lr:g}",
                      f"{d}/reff_corr_{tag}.png")
    plot_epoch_metric(res, "rspec_cov", "spectral redundancy (cov)",
                      f"Spectral redundancy (covariance), lr={lr:g}",
                      f"{d}/rspec_cov_{tag}.png")
    plot_epoch_metric(res, "rspec_corr", "spectral redundancy (corr)",
                      f"Spectral redundancy (correlation), lr={lr:g}",
                      f"{d}/rspec_corr_{tag}.png")
    plot_recon(res, f"{d}/recon_{tag}.png")
    plot_probe(res, f"{d}/probe_acc_{tag}.png")
    plot_epoch_metric(res, "enc_cos", "mean |cosine| of encoder rows",
                      f"Redundancy index (encoder-row alignment), lr={lr:g}",
                      f"{d}/redundancy_index_{tag}.png")
    plot_epoch_metric(res, "bias_mean_cos", "cos(decoder bias, input mean)",
                      f"Mean-collapse index, lr={lr:g}",
                      f"{d}/meancollapse_index_{tag}.png")
    plot_epoch_metric(res, "active_frac", "fraction of neurons firing (>=1% inputs)",
                      f"Active-neuron fraction (kWTA dead-unit check), lr={lr:g}",
                      f"{d}/active_fraction_{tag}.png")


def plot_lr_summary(all_results):
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
    ref = effective_rank(np.cov(mnist_cache[0].numpy().T))
    print(f"raw-MNIST input covariance effective rank = {ref:.1f} "
          f"(shared by all conditions; hidden width = {N_HIDDEN})")

    all_results = {}
    for lr in LR_LIST:
        print(f"\n=== learning rate {lr:g} ===")
        res = {}
        for condition in CONDITIONS:
            print(f"  -- {condition} --")
            res[condition] = aggregate(
                [run_one(condition, lr, s, mnist_cache) for s in SEEDS])
        all_results[lr] = res
        make_plots_for_lr(res, lr)
    plot_lr_summary(all_results)
    print(f"\nDone. Plots saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()