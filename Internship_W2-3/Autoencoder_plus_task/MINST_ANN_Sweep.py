"""
Sweep the reconstruction-loss scaler (lambda) for a shared-encoder model that
jointly does MNIST classification (readout) and input reconstruction (decoder).

Key idea being tested
---------------------
Total loss:      L = L_task + lambda * L_recon
Under *plain* SGD the update is
    theta <- theta - eta*(g_task + lambda*g_recon)
          = theta - eta*g_task - (eta*lambda)*g_recon
so `lambda` acts as an effective learning-rate multiplier on the reconstruction
gradient at every parameter it reaches (decoder + shared encoder), while the
task gradient keeps LR = eta and the readout never sees a recon gradient.

CAVEAT: this "lambda == effective LR" equivalence holds ONLY for vanilla SGD
(no momentum, no weight decay). Under Adam/RMSProp the scale cancels in the
per-parameter normalisation and lambda barely does anything. Keep SGD here.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR      = "./data"
OUT_DIR       = "sweep"
BATCH_SIZE    = 64
EPOCHS        = 25
LR            = 0.1
LATENT_DIM    = 20
INPUT_DIM     = 28 * 28            # 784
NUM_CLASSES   = 10
INIT_SEED     = 0                  # fixed so weights AND batch order are comparable across scalers

# The reconstruction MSE is averaged over 784 pixels (~O(0.1)) while the task
# cross-entropy starts near ln(10) ~ 2.3, so meaningful scalers are large.
RECON_SCALERS = np.logspace(-2, 2.5, 20)   # 0.01 ... ~316, 10 points

MNIST_PIXEL_MEAN = 0.1307          # mean |pixel| for ToTensor()-scaled MNIST ([0, 1])


def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("No GPU found, using CPU.")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SharedEncoderMLP(nn.Module):
    """One shared encoder feeding both a linear decoder (reconstruction) and a
    linear readout (classification)."""

    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)   # recon-only params
        self.readout = nn.Linear(latent_dim, num_classes) # task-only params

    def forward(self, x):
        flat_input = self.flatten(x)          # (B, 784) -- also the recon target
        latent = torch.relu(self.encoder(flat_input))
        decoded = self.decoder(latent)        # (B, 784)
        logits = self.readout(latent)         # (B, 10)
        return decoded, flat_input, logits


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(loader, model, task_criterion, recon_criterion, recon_scaler,
              optimizer=None, device="cpu"):
    """One pass over `loader`. Training if `optimizer` is given, else evaluation.
    Returns (mean_task_loss, mean_recon_loss, accuracy). Reported recon loss is
    the *unscaled* MSE so it's comparable across scalers."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_task, total_recon, total_correct = 0.0, 0.0, 0
    n_batches = len(loader)
    n_items = len(loader.dataset)

    torch.set_grad_enabled(training)
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        decoded, flat_input, logits = model(data)
        task_loss = task_criterion(logits, target)
        recon_loss = recon_criterion(decoded, flat_input)
        loss = task_loss + recon_scaler * recon_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_task += task_loss.item()
        total_recon += recon_loss.item()
        total_correct += (logits.argmax(1) == target).sum().item()
    torch.set_grad_enabled(True)

    return total_task / n_batches, total_recon / n_batches, total_correct / n_items


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def save_weight_grid(weight, path, title_prefix, img_shape=(28, 28)):
    """Grid of per-neuron filters. Adapts to however many filters exist."""
    n = weight.shape[0]
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        if i < n:
            filt = weight[i].reshape(img_shape)
            lim = np.abs(filt).max()
            ax.imshow(filt, cmap="seismic", vmin=-lim, vmax=lim)
            ax.set_title(f"{title_prefix} {i}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_bias(bias, path, img_shape=(28, 28)):
    fig, ax = plt.subplots(figsize=(5, 5))
    filt = bias.reshape(img_shape)
    lim = np.abs(filt).max()
    im = ax.imshow(filt, cmap="seismic", vmin=-lim, vmax=lim)
    ax.set_title("decoder bias (shared)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_curve(y, path, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(y) + 1), y, "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_vs_scaler(scalers, series, path, ylabel, title, logy=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for y, style, label in series:
        ax.plot(scalers, y, style, label=label)
    if hline is not None:
        ax.axhline(hline[0], color="grey", ls="--", lw=1, label=hline[1])
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Recon scaler (lambda)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if len(series) > 1 or hline is not None:
        ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader = DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True, transform=ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=ToTensor()),
        batch_size=BATCH_SIZE, shuffle=False)

    task_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()

    # Per-scaler final metrics (train + test) and parameter magnitudes.
    results = {k: [] for k in
               ("train_task", "train_recon", "train_acc",
                "test_task", "test_recon", "test_acc",
                "enc_mag", "dec_mag", "bias_mag")}

    for recon_scaler in RECON_SCALERS:
        name = f"scaler_{recon_scaler:.3g}"
        run_dir = os.path.join(OUT_DIR, name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== {name}  (recon_scaler={recon_scaler:.4g}) ===")

        # Same init + same shuffle order for every scaler -> directly comparable.
        torch.manual_seed(INIT_SEED)
        model = SharedEncoderMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)

        task_hist, recon_hist, acc_hist = [], [], []
        for epoch in range(EPOCHS):
            tr_task, tr_recon, tr_acc = run_epoch(
                train_loader, model, task_criterion, recon_criterion,
                recon_scaler, optimizer=optimizer, device=device)
            task_hist.append(tr_task)
            recon_hist.append(tr_recon)
            acc_hist.append(tr_acc)
            print(f"  epoch {epoch + 1:2d}/{EPOCHS}  "
                  f"task={tr_task:.4f} recon={tr_recon:.4f} acc={tr_acc:.2%}")

        # Held-out evaluation (this was missing in the original).
        te_task, te_recon, te_acc = run_epoch(
            test_loader, model, task_criterion, recon_criterion,
            recon_scaler, optimizer=None, device=device)
        print(f"  [test] task={te_task:.4f} recon={te_recon:.4f} acc={te_acc:.2%}")

        # Parameter snapshots. decoder.weight is (out=784, in=20); transpose so
        # each row is one latent neuron's fan-out, reshaped to an image.
        W_enc = model.encoder.weight.detach().cpu().numpy()          # (20, 784)
        W_dec = model.decoder.weight.detach().cpu().numpy().T        # (20, 784)
        b_dec = model.decoder.bias.detach().cpu().numpy()            # (784,)

        save_weight_grid(W_enc, os.path.join(run_dir, "encoder_weight.png"), "neuron")
        save_weight_grid(W_dec, os.path.join(run_dir, "decoder_weight.png"), "neuron")
        save_bias(b_dec, os.path.join(run_dir, "decoder_bias.png"))
        save_curve(task_hist,  os.path.join(run_dir, "task_loss.png"),  "task loss (CE)",  "Task loss")
        save_curve(recon_hist, os.path.join(run_dir, "recon_loss.png"), "recon loss (MSE)", "Recon loss")
        save_curve(acc_hist,   os.path.join(run_dir, "accuracy.png"),   "train accuracy",   "Accuracy")

        results["train_task"].append(task_hist[-1])
        results["train_recon"].append(recon_hist[-1])
        results["train_acc"].append(acc_hist[-1])
        results["test_task"].append(te_task)
        results["test_recon"].append(te_recon)
        results["test_acc"].append(te_acc)
        results["enc_mag"].append(float(np.mean(np.abs(W_enc))))
        results["dec_mag"].append(float(np.mean(np.abs(W_dec))))
        results["bias_mag"].append(float(np.mean(np.abs(b_dec))))

    results = {k: np.array(v) for k, v in results.items()}

    # --- Summary plots across scalers ---
    save_vs_scaler(RECON_SCALERS,
                   [(results["train_task"], "o-", "train"),
                    (results["test_task"],  "s-", "test")],
                   os.path.join(OUT_DIR, "task_loss_vs_scaler.png"),
                   "task loss (CE)", "Task loss vs recon scaler")
    save_vs_scaler(RECON_SCALERS,
                   [(results["train_recon"], "o-", "train"),
                    (results["test_recon"],  "s-", "test")],
                   os.path.join(OUT_DIR, "recon_loss_vs_scaler.png"),
                   "recon loss (MSE)", "Recon loss vs recon scaler")
    save_vs_scaler(RECON_SCALERS,
                   [(results["train_acc"], "o-", "train"),
                    (results["test_acc"],  "s-", "test")],
                   os.path.join(OUT_DIR, "accuracy_vs_scaler.png"),
                   "accuracy", "Accuracy vs recon scaler")

    # Where the signal lives: as lambda grows the recon path dominates; watch
    # whether the encoder/decoder weights or the decoder bias carry it.
    save_vs_scaler(RECON_SCALERS,
                   [(results["enc_mag"],  "o-", "mean |encoder weight|"),
                    (results["dec_mag"],  "s-", "mean |decoder weight|"),
                    (results["bias_mag"], "^-", "mean |decoder bias|")],
                   os.path.join(OUT_DIR, "magnitude_vs_scaler.png"),
                   "mean |value|", "Where the signal lives",
                   logy=True, hline=(MNIST_PIXEL_MEAN, "MNIST pixel mean (~0.131)"))

    # --- Text summary ---
    print("\nSweep complete. Summary:")
    header = ("scaler", "tr_task", "te_task", "tr_acc", "te_acc",
              "|W_enc|", "|W_dec|", "|bias|")
    print("  " + "".join(f"{h:>10}" for h in header))
    for i, s in enumerate(RECON_SCALERS):
        print("  " + "".join(f"{v:>10.4g}" for v in (
            s, results["train_task"][i], results["test_task"][i],
            results["train_acc"][i], results["test_acc"][i],
            results["enc_mag"][i], results["dec_mag"][i], results["bias_mag"][i])))


if __name__ == "__main__":
    main()