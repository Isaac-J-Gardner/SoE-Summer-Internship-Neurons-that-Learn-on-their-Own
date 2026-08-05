"""
nan_experiments.py
==================================================================
Neuron Autoencoder Network (NaN / "NAN" in Bull, 2023) experiments.

This script fills in the dataset x learning-rule grid so the two changes that
happened when the work moved from Bull's NK study to the MNIST study can be
disentangled:

                     |  random hill climber      |  SGD
    -----------------+---------------------------+--------------------------
      NK  model      |  run_nk_hillclimber()     |  run_nk_sgd()
      MNIST          |  run_mnist_hillclimber()  |  (your existing base code)

Bull's NaN architecture (identical in every cell, only the *learning rule* and
the *dataset* change):

  * inputs  ->  H hidden neurons (sigmoid).                     [the encoder]
  * each hidden neuron n reconstructs the WHOLE input from its
    own scalar activation a_n:   recon_n = a_n * w_dec[n] + b_dec[n]   [decoder]
  * a readout maps the H hidden activations to the task output.  [the readout]

The defining feature: the hidden representation is shaped ONLY by reconstruction
(the autoencoding objective), and the readout is trained ONLY by the task. The
readout never pushes gradient back into the hidden layer. In SGD this is enforced
with a detach; in the hill climber it is enforced structurally, because an
"autoencoding cycle" only touches hidden-neuron weights (evaluated on recon MSE)
and a "task cycle" only touches readout weights (evaluated on task loss) -- exactly
as described in Bull (2023).

Reference settings from Bull (2023) "Toward Neuromic Computing: Neurons as
Autoencoders":  H = 10, R = 1.0, 10,000 learning iterations, averaged over 20
runs, train/test sets of 1000 NK examples, 0<=K<=15, 20<=N<=1000, binary genome
with 0 mapped to -1.0.

Run examples
------------
    python Internship_W4-5\RHC_SGD_NK_MNIST.py --exp nk_hc            # Bull's original cell
    python Internship_W4-5\RHC_SGD_NK_MNIST.py --exp nk_sgd
    python Internship_W4-5\RHC_SGD_NK_MNIST.py --exp mnist_hc
    python Internship_W4-5\RHC_SGD_NK_MNIST.py --exp all --seeds 5
    python Internship_W4-5\RHC_SGD_NK_MNIST.py --exp nk_hc --N 20 --K 5 --iters 10000 --seeds 20

Every runner accepts `seeds=` (or --seeds) to average over independent runs.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib is optional (only used for the summary plot); guard the import so
# the compute path still runs on a headless box without it.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 1. The NaN architecture (shared by every experiment)
# ======================================================================
class NaN(nn.Module):
    """Neuron Autoencoder Network.

    task = 'regression'      -> single sigmoid output  (NK fitness in (0,1))
    task = 'classification'  -> n_outputs logits       (MNIST digits)
    """

    def __init__(self, n_inputs, n_hidden, n_outputs, task="regression"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.task = task

        self.encoder = nn.Linear(n_inputs, n_hidden)          # input -> hidden
        # per-neuron decoder: reconstruct all n_inputs from one scalar activation
        self.dec_w = nn.Parameter(torch.empty(n_hidden, n_inputs))
        self.dec_b = nn.Parameter(torch.empty(n_hidden, n_inputs))
        self.readout = nn.Linear(n_hidden, n_outputs)         # hidden -> task

        self.reset_small()   # SGD-style init by default; hill climber overrides

    # ---- initialisations -------------------------------------------------
    def reset_small(self):
        """SGD-friendly init (close to the MNIST base script)."""
        nn.init.kaiming_uniform_(self.encoder.weight, a=5 ** 0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.dec_w, std=0.01)
        nn.init.zeros_(self.dec_b)
        nn.init.kaiming_uniform_(self.readout.weight, a=5 ** 0.5)
        nn.init.zeros_(self.readout.bias)

    def reset_uniform_pm1(self):
        """Bull's init: every weight seeded uniformly in [-1, 1]."""
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-1.0, 1.0)

    # ---- forward ---------------------------------------------------------
    def forward(self, x):
        x = x.flatten(1)                                   # [B, N]
        hidden = torch.sigmoid(self.encoder(x))            # [B, H]
        # each neuron reconstructs the full input from its own scalar activation
        recon = hidden.unsqueeze(2) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)  # [B,H,N]
        out = self.readout(hidden.detach())                # task readout, detached
        if self.task == "regression":
            out = torch.sigmoid(out)                       # NK fitness lives in (0,1)
        return out, recon, x


def recon_loss_all(recon, x):
    """Mean reconstruction MSE over batch, neurons and inputs."""
    target = x.unsqueeze(1).expand_as(recon)               # [B,H,N]
    return F.mse_loss(recon, target)


# ======================================================================
# 2. Data
# ======================================================================
def make_nk_dataset(N, K, n_train, n_test, seed):
    """Build a train/test NK regression dataset on ONE fixed landscape.

    Faithful to the supplied NK_model.py: fitness table has 2**(K+1) rows per
    gene, the gene itself is the most-significant bit, neighbours wrap around,
    fitness = mean over genes of the looked-up contributions. Genome bits {0,1}
    are fed to the MLP as {-1,+1}. Returns torch tensors on DEVICE.
    """
    rng = np.random.default_rng(seed)
    fitness_table = rng.random((N, 2 ** (K + 1)))          # fixed landscape
    powers = 2 ** np.arange(K, -1, -1)                     # [K+1]
    neigh = (np.arange(N)[:, None] + np.arange(K + 1)[None, :]) % N   # [N,K+1]

    def sample(P):
        G = rng.integers(0, 2, size=(P, N))                # {0,1} genomes  [P,N]
        idx = (G[:, neigh] * powers).sum(axis=2)           # row per gene   [P,N]
        contrib = fitness_table[np.arange(N)[None, :], idx]  # [P,N]
        y = contrib.mean(axis=1)                           # fitness in [0,1]
        X = G.astype(np.float32) * 2.0 - 1.0               # {-1,+1} inputs
        return X, y.astype(np.float32)

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    to = lambda a: torch.from_numpy(a).to(DEVICE)
    return (to(Xtr), to(ytr).unsqueeze(1),
            to(Xte), to(yte).unsqueeze(1))


def make_mnist_dataset(n_train, n_test, center=False, seed=0):
    """Load MNIST as flat tensors. n_train/n_test cap the number of examples
    (the hill climber evaluates the FULL set every iteration, so keep it modest,
    ~1000-2000, matching Bull's train/test size). center=True subtracts the
    training-set pixel mean (the preprocessing you explored in the reports)."""
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    tr = datasets.MNIST("./data", train=True, download=True, transform=ToTensor())
    te = datasets.MNIST("./data", train=False, download=True, transform=ToTensor())

    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(tr), generator=g)[:n_train]
    te_idx = torch.randperm(len(te), generator=g)[:n_test]

    def stack(ds, idx):
        X = torch.stack([ds[i][0].view(-1) for i in idx])
        y = torch.tensor([ds[i][1] for i in idx])
        return X, y

    Xtr, ytr = stack(tr, tr_idx)
    Xte, yte = stack(te, te_idx)
    if center:
        mu = Xtr.mean(0, keepdim=True)
        Xtr, Xte = Xtr - mu, Xte - mu
    return Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)


# ======================================================================
# 3. Learning rule A -- random hill climber (Bull, 2023)
# ======================================================================
@torch.no_grad()
def train_hill_climber(model, Xtr, ytr, Xte, yte, task,
                       n_iters=10000, R=1.0, checkpoints=50, verbose=False):
    """One faithful hill-climber run.

    Every iteration is, with prob 0.5, an AUTOENCODING cycle (perturb one weight
    of one hidden neuron, keep it iff that neuron's reconstruction MSE over the
    training set does not get worse) or a TASK cycle (perturb one readout weight,
    keep it iff the task loss over the training set does not get worse). Ties are
    broken by keeping the new value with prob 0.5, as in Bull.

    Hidden activations are cached and only the touched neuron's column is
    recomputed, so each iteration is cheap even over the whole training set.
    """
    model.to(DEVICE).eval()
    model.reset_uniform_pm1()                              # Bull's [-1,1] seeding

    N, H = model.n_inputs, model.n_hidden
    Wenc, benc = model.encoder.weight, model.encoder.bias  # [H,N], [H]
    Wdec, bdec = model.dec_w, model.dec_b                  # [H,N], [H,N]
    Wout, bout = model.readout.weight, model.readout.bias  # [O,H], [O]

    # ---- cached forward pieces ----
    pre = Xtr @ Wenc.t() + benc                            # [P,H] pre-activation
    hidden = torch.sigmoid(pre)                            # [P,H]

    def neuron_recon_mse(n):
        # recon for neuron n: a_n * w_dec[n] + b_dec[n]  vs  Xtr
        r = hidden[:, n:n + 1] * Wdec[n] + bdec[n]         # [P,N]
        return ((r - Xtr) ** 2).mean()

    def task_loss():
        logits = hidden @ Wout.t() + bout                  # [P,O]
        if task == "regression":
            return F.mse_loss(torch.sigmoid(logits), ytr)
        return F.cross_entropy(logits, ytr)

    # current per-neuron reconstruction baselines (kept in sync on accepted moves)
    cur_recon = torch.stack([neuron_recon_mse(n) for n in range(H)])   # [H]
    # NOTE: the task baseline is recomputed fresh inside each task cycle, because
    # autoencoding cycles drift the hidden representation the readout reads from,
    # so any cached task loss would go stale.

    hist = {"iter": [], "test_task": [], "test_recon": []}
    ckpt_every = max(1, n_iters // checkpoints)
    n_out_w = Wout.numel()
    frac_w = n_out_w / (n_out_w + bout.numel())

    def keep(new, old):
        # accept if strictly better, or break ties at random
        if new < old:
            return True
        if new == old:
            return torch.rand(1).item() < 0.5
        return False

    for it in range(1, n_iters + 1):
        if torch.rand(1).item() < 0.5:
            # ---------- AUTOENCODING CYCLE ----------
            n = np.random.randint(H)
            # which of neuron n's params to nudge:
            #   0 encoder weight, 1 encoder bias, 2 decoder weight, 3 decoder bias
            kind = np.random.randint(4)
            delta = (torch.rand(1, device=DEVICE).item() * 2 - 1) * R

            if kind == 0:                                   # encoder weight -> changes a_n
                j = np.random.randint(N)
                old_val = Wenc[n, j].item(); Wenc[n, j] += delta
                new_pre = pre[:, n] + delta * Xtr[:, j]
                new_hid = torch.sigmoid(new_pre)
                r = new_hid.unsqueeze(1) * Wdec[n] + bdec[n]
                new_mse = ((r - Xtr) ** 2).mean()
                if keep(new_mse, cur_recon[n]):
                    pre[:, n] = new_pre; hidden[:, n] = new_hid; cur_recon[n] = new_mse
                else:
                    Wenc[n, j] = old_val
            elif kind == 1:                                 # encoder bias -> changes a_n
                old_val = benc[n].item(); benc[n] += delta
                new_pre = pre[:, n] + delta
                new_hid = torch.sigmoid(new_pre)
                r = new_hid.unsqueeze(1) * Wdec[n] + bdec[n]
                new_mse = ((r - Xtr) ** 2).mean()
                if keep(new_mse, cur_recon[n]):
                    pre[:, n] = new_pre; hidden[:, n] = new_hid; cur_recon[n] = new_mse
                else:
                    benc[n] = old_val
            elif kind == 2:                                 # decoder weight (a_n unchanged)
                j = np.random.randint(N)
                old_val = Wdec[n, j].item(); Wdec[n, j] += delta
                new_mse = neuron_recon_mse(n)
                if keep(new_mse, cur_recon[n]):
                    cur_recon[n] = new_mse
                else:
                    Wdec[n, j] = old_val
            else:                                           # decoder bias (a_n unchanged)
                j = np.random.randint(N)
                old_val = bdec[n, j].item(); bdec[n, j] += delta
                new_mse = neuron_recon_mse(n)
                if keep(new_mse, cur_recon[n]):
                    cur_recon[n] = new_mse
                else:
                    bdec[n, j] = old_val
        else:
            # ---------- TASK CYCLE ----------
            base_task = task_loss()                         # fresh: encoder may have drifted
            delta = (torch.rand(1, device=DEVICE).item() * 2 - 1) * R
            if np.random.rand() < frac_w:
                o = np.random.randint(Wout.shape[0]); h = np.random.randint(Wout.shape[1])
                old_val = Wout[o, h].item(); Wout[o, h] += delta
                if not keep(task_loss(), base_task):
                    Wout[o, h] = old_val
            else:
                o = np.random.randint(bout.shape[0])
                old_val = bout[o].item(); bout[o] += delta
                if not keep(task_loss(), base_task):
                    bout[o] = old_val

        # ---- checkpoint on the TEST set ----
        if it % ckpt_every == 0 or it == n_iters:
            tt, tr_ = evaluate(model, Xte, yte, task)
            hist["iter"].append(it)
            hist["test_task"].append(tt)
            hist["test_recon"].append(tr_)
            if verbose:
                extra = f"acc {tt:.3f}" if task == "classification" else f"mse {tt:.4f}"
                print(f"  iter {it:>6}: test {extra}, test recon {tr_:.4f}")

    return hist


# ======================================================================
# 4. Learning rule B -- SGD (alternating recon/task cycles, Bull-style)
# ======================================================================
def train_sgd(model, Xtr, ytr, Xte, yte, task,
              epochs=20, batch_size=64, task_lr=0.01, recon_lr=100.0,
              verbose=False):
    """SGD training that mirrors Bull's 50/50 alternation, per batch.

    Two optimizers keep the objectives cleanly separated (which is what the
    detached readout already guarantees): the recon optimizer owns the
    encoder+decoder, the task optimizer owns the readout. Defaults follow the
    MNIST base script (task_lr=0.01, high recon_lr). For NK, recon_lr around
    1-10 is a saner starting point -- tune per dataset.
    """
    model.to(DEVICE).train()
    model.reset_small()

    recon_params = list(model.encoder.parameters()) + [model.dec_w, model.dec_b]
    task_params = list(model.readout.parameters())
    opt_recon = torch.optim.SGD(recon_params, lr=recon_lr)
    opt_task = torch.optim.SGD(task_params, lr=task_lr)

    task_criterion = (nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss())

    P = Xtr.shape[0]
    hist = {"iter": [], "test_task": [], "test_recon": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(P, device=DEVICE)
        for s in range(0, P, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            out, recon, xflat = model(xb)
            if np.random.rand() < 0.5:                      # task cycle
                loss = task_criterion(out, yb)
                opt_task.zero_grad(); loss.backward(); opt_task.step()
            else:                                           # autoencoding cycle
                loss = recon_loss_all(recon, xflat)
                opt_recon.zero_grad(); loss.backward(); opt_recon.step()

        tt, tr_ = evaluate(model, Xte, yte, task)
        hist["iter"].append(ep)
        hist["test_task"].append(tt)
        hist["test_recon"].append(tr_)
        if verbose:
            extra = f"acc {tt:.3f}" if task == "classification" else f"mse {tt:.4f}"
            print(f"  epoch {ep:>3}: test {extra}, test recon {tr_:.4f}")

    return hist


# ======================================================================
# 5. Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, Xte, yte, task):
    """Return (primary_test_metric, test_recon_mse).
    primary metric = accuracy for classification, MSE for regression."""
    model.eval()
    out, recon, xflat = model(Xte)
    recon_mse = recon_loss_all(recon, xflat).item()
    if task == "classification":
        acc = (out.argmax(1) == yte).float().mean().item()
        return acc, recon_mse
    return F.mse_loss(out, yte).item(), recon_mse


# ======================================================================
# 6. Seed-averaging harness
# ======================================================================
def run_seeds(build_data, build_model, train_fn, task, seeds, label):
    """Run `seeds` independent repeats, aligning checkpoints and averaging."""
    print(f"\n=== {label}  ({seeds} seed{'s' if seeds != 1 else ''}) ===")
    all_task, all_recon, iters, finals = [], [], None, []
    t0 = time.time()
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        Xtr, ytr, Xte, yte = build_data(s)
        model = build_model()
        hist = train_fn(model, Xtr, ytr, Xte, yte, task)
        iters = hist["iter"]
        all_task.append(hist["test_task"])
        all_recon.append(hist["test_recon"])
        finals.append(hist["test_task"][-1])
        unit = "acc" if task == "classification" else "mse"
        print(f"  seed {s}: final test {unit} = {hist['test_task'][-1]:.4f}")

    all_task = np.array(all_task); all_recon = np.array(all_recon)
    unit = "accuracy" if task == "classification" else "MSE"
    print(f"  -> final test {unit}: {np.mean(finals):.4f} +/- {np.std(finals):.4f}"
          f"   ({time.time()-t0:.1f}s)")
    return {"label": label, "task": task, "iter": iters,
            "mean_task": all_task.mean(0), "std_task": all_task.std(0),
            "mean_recon": all_recon.mean(0), "std_recon": all_recon.std(0),
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals))}


# ======================================================================
# 7. The three requested experiments
# ======================================================================
def run_nk_hillclimber(N=20, K=5, H=10, R=1.0, iters=10000,
                       n_train=1000, n_test=1000, seeds=20):
    """Cell 1: Bull's original setup -- NaN on NK with a random hill climber."""
    build_data = lambda s: make_nk_dataset(N, K, n_train, n_test, seed=1000 + s)
    build_model = lambda: NaN(N, H, 1, task="regression")
    train_fn = lambda m, a, b, c, d, t: train_hill_climber(
        m, a, b, c, d, t, n_iters=iters, R=R)
    return run_seeds(build_data, build_model, train_fn, "regression",
                     seeds, f"NK + hill climber (N={N}, K={K}, H={H})")


def run_nk_sgd(N=20, K=5, H=10, epochs=50, batch_size=64,
               task_lr=0.01, recon_lr=5.0, n_train=1000, n_test=1000, seeds=20):
    """Cell 2: NaN on NK with SGD (Bull's architecture, SGD swapped in)."""
    build_data = lambda s: make_nk_dataset(N, K, n_train, n_test, seed=1000 + s)
    build_model = lambda: NaN(N, H, 1, task="regression")
    train_fn = lambda m, a, b, c, d, t: train_sgd(
        m, a, b, c, d, t, epochs=epochs, batch_size=batch_size,
        task_lr=task_lr, recon_lr=recon_lr)
    return run_seeds(build_data, build_model, train_fn, "regression",
                     seeds, f"NK + SGD (N={N}, K={K}, H={H})")


def run_mnist_hillclimber(H=20, R=1.0, iters=10000, n_train=1000, n_test=1000,
                          center=False, seeds=5):
    """Cell 3: NaN on MNIST with a random hill climber (Bull's rule, new data).
    H defaults to 20 to line up with your existing MNIST+SGD base script."""
    build_data = lambda s: make_mnist_dataset(n_train, n_test, center=center, seed=s)
    build_model = lambda: NaN(28 * 28, H, 10, task="classification")
    train_fn = lambda m, a, b, c, d, t: train_hill_climber(
        m, a, b, c, d, t, n_iters=iters, R=R)
    return run_seeds(build_data, build_model, train_fn, "classification",
                     seeds, f"MNIST + hill climber (H={H})")


# Bonus (the 4th cell) -- your existing base, exposed here for completeness.
def run_mnist_sgd(H=20, epochs=20, batch_size=64, task_lr=0.01, recon_lr=100.0,
                  n_train=60000, n_test=10000, center=False, seeds=1):
    build_data = lambda s: make_mnist_dataset(n_train, n_test, center=center, seed=s)
    build_model = lambda: NaN(28 * 28, H, 10, task="classification")
    train_fn = lambda m, a, b, c, d, t: train_sgd(
        m, a, b, c, d, t, epochs=epochs, batch_size=batch_size,
        task_lr=task_lr, recon_lr=recon_lr)
    return run_seeds(build_data, build_model, train_fn, "classification",
                     seeds, f"MNIST + SGD (H={H})")


# ======================================================================
# 8. Plotting
# ======================================================================
def plot_results(results, path="nan_results.png"):
    if not _HAVE_MPL or not results:
        return
    fig, ax = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    for i, r in enumerate(results):
        a = ax[0][i]
        m, sd, x = r["mean_task"], r["std_task"], r["iter"]
        a.plot(x, m, lw=2)
        a.fill_between(x, m - sd, m + sd, alpha=0.25)
        a.set_title(r["label"], fontsize=9)
        a.set_xlabel("learning iterations" if "hill" in r["label"] else "epochs")
        a.set_ylabel("test accuracy" if r["task"] == "classification" else "test MSE")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nSaved curves -> {path}")


# ======================================================================
# 9. CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="all",
                   choices=["nk_hc", "nk_sgd", "mnist_hc", "mnist_sgd", "all"])
    p.add_argument("--seeds", type=int, default=None, help="repeats to average over")
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--H", type=int, default=None, help="hidden neurons (defaults per exp)")
    p.add_argument("--R", type=float, default=1.0, help="hill-climber perturbation range")
    p.add_argument("--iters", type=int, default=10000, help="hill-climber iterations")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--task_lr", type=float, default=0.01)
    p.add_argument("--recon_lr", type=float, default=None)
    p.add_argument("--center", action="store_true", help="MNIST: subtract pixel mean")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    results = []
    sd = lambda default: default if args.seeds is None else args.seeds

    if args.exp in ("nk_hc", "all"):
        results.append(run_nk_hillclimber(
            N=args.N, K=args.K, H=(args.H or 10), R=args.R, iters=args.iters,
            n_train=args.n_train, n_test=args.n_test, seeds=sd(20)))
    if args.exp in ("nk_sgd", "all"):
        results.append(run_nk_sgd(
            N=args.N, K=args.K, H=(args.H or 10), epochs=args.epochs,
            task_lr=args.task_lr, recon_lr=(args.recon_lr or 5.0),
            n_train=args.n_train, n_test=args.n_test, seeds=sd(20)))
    if args.exp in ("mnist_hc", "all"):
        results.append(run_mnist_hillclimber(
            H=(args.H or 20), R=args.R, iters=args.iters,
            n_train=args.n_train, n_test=args.n_test, center=args.center, seeds=sd(5)))
    if args.exp == "mnist_sgd":
        results.append(run_mnist_sgd(
            H=(args.H or 20), epochs=args.epochs, task_lr=args.task_lr,
            recon_lr=(args.recon_lr or 100.0), center=args.center, seeds=sd(1)))

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()
