"""
Spiking Neuron-Autoencoder (NaN) on MNIST, with WTA lateral inhibition and a
per-neuron homeostatic ADAPTIVE THRESHOLD.

What changed relative to the standard (non-spiking) NaN is grouped in CONFIG
below. In short, three new mechanisms were added:
  1. spiking dynamics  -> num_steps, beta, base_thresh, spike_grad
  2. WTA inhibition     -> inhib_strength  (fixed negative recurrent weights)
  3. adaptive threshold -> theta_plus, theta_decay  (homeostasis, NOT backprop)
plus two new modules: WTASpikingEncoder (manual LIF) and a Leaky "readout" in
the decoder that only integrates (never spikes).
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import utils
from snntorch import surrogate

# ============================================================================
# CONFIG  --  everything the spiking version introduces lives here
# ============================================================================
# ---- Standard NaN params (unchanged from the non-spiking network) ----------
n_in        = 784        # input dimension (28*28)
n_hidden    = 20         # hidden / autoencoding neurons
batch_size  = 64
lr          = 1.0        # SGD learning rate (still worth sweeping, as before)
epochs      = 1

# ---- NEW: spiking dynamics -------------------------------------------------
num_steps   = 20          # timesteps simulated per image (try 10-20 to give WTA
                         #   more time to settle before drawing conclusions)
beta        = 0.5        # LIF membrane decay per step (0 = no memory, 1 = perfect integrator)
base_thresh = 1.0        # base spiking threshold (lower -> more spikes)
spike_grad  = surrogate.atan(alpha=2.0)   # surrogate gradient for the non-diff spike

# ---- NEW: WTA lateral inhibition ------------------------------------------
inhib_strength = 1.0     # each neuron's spike subtracts this from EVERY other
                         #   neuron's drive on the next step. You had 2.0, which
                         #   drove single-winner collapse; re-sweep now that the
                         #   adaptive threshold is actually active.

# ---- NEW: adaptive (homeostatic) threshold --------------------------------
theta_plus  = 0.05       # threshold rise, proportional to a neuron's firing.
                         #   THIS is the knob to raise if one neuron still hogs.
theta_decay = 0.9        # per-step decay of the adaptive offset back toward base
# ============================================================================


class NeuronDecoder(nn.Module):
    """Per-neuron decoder: each hidden neuron owns its own 784-d decoder that
    reconstructs the FULL input from its single scalar (Bull's NaN structure)."""
    def __init__(self, n_neurons=n_hidden, in_dim=n_in):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                      # h: (B, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   # (B, n_neurons, in_dim)


class WTASpikingEncoder(nn.Module):
    """20 LIF neurons with all-to-all lateral inhibition (WTA) and a per-neuron
    homeostatic adaptive threshold.

    Implemented as a MANUAL LIF (rather than snn.Leaky) because the threshold has
    to (a) be per-neuron and (b) change every timestep -- neither is convenient
    through snn.Leaky's scalar threshold.
    """
    def __init__(self, n_in, n_hidden, beta, base_thresh,
                 inhib_strength, theta_plus, theta_decay, spike_grad):
        super().__init__()
        self.n_hidden    = n_hidden
        self.beta        = beta
        self.base_thresh = base_thresh
        self.theta_plus  = theta_plus
        self.theta_decay = theta_decay
        self.spike_grad  = spike_grad

        self.fc = nn.Linear(n_in, n_hidden)     # encoder weights (input -> hidden)

        # Fixed lateral-inhibition matrix: -inhib off-diagonal, 0 on the diagonal
        # (no self-inhibition). Not learned.
        W = -inhib_strength * (torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden))
        self.register_buffer("W_inh", W)

        # Adaptive-threshold offset per neuron. Registered as a BUFFER so it
        # (a) moves with .to(device), (b) is saved/loaded, and (c) is OUTSIDE the
        # autograd graph -- homeostasis is not learned by the task gradient.
        # It PERSISTS across batches: that cross-input memory is what stops the
        # same neuron winning for every image.
        self.register_buffer("theta", torch.zeros(n_hidden))

    def forward(self, x, num_steps):
        drive = self.fc(x)                       # (B, H) feedforward drive, constant across steps
        mem = torch.zeros_like(drive)
        spk = torch.zeros_like(drive)            # last step's spikes (0 at t=0 -> no inhibition yet)
        spk_rec, mem_rec = [], []

        for _ in range(num_steps):
            # feedforward drive + inhibition from whoever fired on the previous step
            cur = drive + spk @ self.W_inh       # W_inh is symmetric, so no .t() needed
            mem = self.beta * mem + cur          # leaky integration

            # per-neuron effective threshold; .theta is a detached buffer
            eff_thresh = self.base_thresh + self.theta          # (H,), broadcasts over batch
            spk = self.spike_grad(mem - eff_thresh)             # surrogate spike (0/1)
            mem = mem - spk * eff_thresh                        # subtract-reset

            # ---- homeostatic threshold update (no gradient) -------------------
            # A neuron that just fired raises its own threshold; all thresholds
            # decay back toward base each step. Over many inputs this equalises
            # firing rates. Only update while training.
            if self.training:
                with torch.no_grad():
                    self.theta.mul_(self.theta_decay).add_(
                        self.theta_plus * spk.detach().mean(0))   # mean over batch

            spk_rec.append(spk)
            mem_rec.append(mem)

        return spk_rec, mem_rec                   # each: list[T] of (B, H)


class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WTASpikingEncoder(
            n_in, n_hidden, beta, base_thresh,
            inhib_strength, theta_plus, theta_decay, spike_grad)

        # Decoder = per-neuron decoder -> a Leaky set to NEVER spike (huge
        # threshold), so it acts as a leaky integrator / analog readout whose
        # membrane we read out at the final step.
        self.decoder = nn.Sequential(
            NeuronDecoder(n_hidden, n_in),
            snn.Leaky(beta=beta, spike_grad=spike_grad,
                      init_hidden=True, output=True, threshold=20000),
        )

    def forward(self, x):
        utils.reset(self.decoder)                 # reset decoder LIF membrane (encoder is manual)
        x = x.view(x.size(0), -1)                 # (B, 784)

        # ---- encode ----
        spk_rec, _ = self.encoder(x, num_steps=num_steps)   # list[T] of (B, H)
        spk_rec = torch.stack(spk_rec, dim=2)               # (B, H, T)

        # ---- decode: integrate each neuron's reconstruction over time ----
        mem_rec = []
        for step in range(num_steps):
            _, mem = self.decoder(spk_rec[..., step])       # (B, H, 784)
            mem_rec.append(mem)
        out = mem_rec[-1]                                   # membrane at final step: (B, H, 784)
        return x, out


def train(network, trainloader, opti, epoch):
    network.train()
    for batch_idx, (real_img, labels) in enumerate(trainloader):
        opti.zero_grad()
        real_img = real_img.to(device)

        x, x_recon = network(real_img)                      # x: (B,784), x_recon: (B,H,784)
        # each neuron reconstructs the full image -> compare against x broadcast over H
        loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))

        loss_val.backward()
        opti.step()

        if batch_idx % 50 == 0:
            print(f'Train[{epoch}/{epochs}][{batch_idx}/{len(trainloader)}] '
                  f'Loss: {loss_val.item():.5f}')
    return loss_val


def test(network, testloader, epoch):
    network.eval()
    with torch.no_grad():
        losses = []
        for real_img, labels in testloader:
            real_img = real_img.to(device)
            x, x_recon = network(real_img)
            losses.append(F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon)).item())
    print(f'Test[{epoch}/{epochs}] Loss: {np.mean(losses):.5f}')
    return np.mean(losses)


# ---- device ----------------------------------------------------------------
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('No GPU found, using CPU.')

# ---- data ------------------------------------------------------------------
data_dir = './data'
train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transforms.ToTensor())
test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ---- build + train ---------------------------------------------------------
net = SAE().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for e in range(epochs):
    train(net, train_loader, optimizer, e)
    test(net, test_loader, e)

# ---- visualise encoder / decoder weights + decoder bias --------------------
W  = net.encoder.fc.weight.detach().cpu().numpy()      # (20, 784)  encoder
W2 = net.decoder[0].weight.detach().cpu().numpy()      # (20, 784)  decoder weights
W3 = net.decoder[0].bias.detach().cpu().numpy()        # (20, 784)  decoder bias

for title, mat in [("encoder weights", W), ("decoder weights", W2), ("decoder bias", W3)]:
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        filt = mat[i].reshape(28, 28)
        m = np.abs(filt).max() + 1e-9
        ax.imshow(filt, cmap='seismic', vmin=-m, vmax=m)
        ax.set_title(f'neuron {i}')
        ax.axis('off')
    plt.tight_layout()
plt.show()