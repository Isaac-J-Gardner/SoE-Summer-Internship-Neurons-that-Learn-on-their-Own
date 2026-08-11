import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import utils
from snntorch import surrogate

torch.manual_seed(0)

# ============================================================================
# CONFIG  --  everything below the "standard" block is NEW vs the plain NaN
# ============================================================================
# ---- standard ----
batch_size  = 64
lr          = 0.1       # NOTE: you had 10. With gating the loss is normalised by
                         #   the number of ACTIVE neuron-sample pairs, so re-sweep;
                         #   lr=10 SGD will almost certainly diverge here.
epochs      = 1

# ---- spiking ----
beta        = 0.5        # membrane decay
num_steps   = 5          # timesteps per image
thresh      = 1.0        # base spiking threshold

# ---- WTA lateral inhibition (your mechanism: negative recurrent weights) ----
inhib_strength = 1     # subtractive inhibition; each spike lowers others' drive

# ---- NEW: adaptive (homeostatic) threshold ----
theta_plus  = 0.5    # a neuron that fires raises its own threshold by this
theta_decay = 0.9        # thresholds decay back toward base each step
                         #   -> too small: one neuron hogs (dead units)
                         #   -> too large: threshold overrides the match (scrambles)

# ---- NEW: competitive gating ----
GATING = True            # if True, only neurons that fired learn on a given input
# ============================================================================

spike_grad = surrogate.atan(alpha=2.0)


class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        # h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   # (batch, n_neurons, in_dim)


class WTASpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=20, beta=0.9, inhib_strength=1.0,
                 theta_plus=0.05, theta_decay=0.9):
        super().__init__()
        self.n_hidden    = n_hidden
        self.theta_plus  = theta_plus
        self.theta_decay = theta_decay
        self.fc  = nn.Linear(n_in, n_hidden)
        self.hetero_beta = torch.rand(n_hidden)*0.5 + 0.4
        self.lif = snn.Leaky(beta=self.hetero_beta, spike_grad=spike_grad, threshold=thresh)

        # every neuron inhibits every other; none inhibits itself (fixed, not learned)
        W = -inhib_strength * (torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden))
        self.register_buffer("W_inh", W)

        # NEW: per-neuron adaptive threshold. Buffer -> moves with .to(device),
        # not on the autograd graph, and PERSISTS across batches (homeostasis).
        self.register_buffer("theta", torch.zeros(n_hidden))

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):
            # feedforward drive + lateral inhibition (from last step's spikes)
            #   - adaptive threshold, applied here as a per-neuron negative current.
            #   (snn.Leaky's threshold is scalar, so injecting -theta is the clean
            #    way to give each neuron its own effective threshold. Raising the
            #    threshold and subtracting this current are equivalent up to the leak.)
            cur = self.fc(x) + spk @ self.W_inh.t() - self.theta
            spk, mem = self.lif(cur, mem)

            # NEW: homeostatic update (no gradient). Firing raises own threshold;
            # all thresholds decay each step. Over many inputs this equalises rates.
            if self.training:
                with torch.no_grad():
                    self.theta.mul_(self.theta_decay).add_(self.theta_plus * spk.detach().mean(0))

            spk_rec.append(spk)
            mem_rec.append(mem)
        return spk_rec, mem_rec


class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WTASpikingEncoder(784, 20, beta=beta, inhib_strength=inhib_strength,
                                         theta_plus=theta_plus, theta_decay=theta_decay)
        self.decoder = nn.Sequential(
            NeuronDecoder(20, 784),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                      output=True, threshold=20000)   # never spikes: integrator readout
        )

    def forward(self, x):
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        x = x.view(x.size(0), -1)                    # (B,784)

        # encode
        spk_rec, _ = self.encode(x)
        spk_rec = torch.stack(spk_rec, dim=2)        # (B, H, T)

        # NEW: per-(sample, neuron) activity = total spikes over time -> the gate
        activity = spk_rec.sum(dim=2)                # (B, H)

        # decode: integrate each neuron's reconstruction over time
        spk_mem2 = []
        for step in range(num_steps):
            _, x_mem_recon = self.decode(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]   # membrane at last step (B,H,784)
        return x, out, activity

    def encode(self, x):
        return self.encoder(x, num_steps=num_steps)

    def decode(self, x):
        return self.decoder(x)


def recon_loss(x_recon, x, activity):
    """Per-neuron reconstruction MSE, optionally GATED by activity so that only
    neurons which fired for a given input contribute (and therefore receive
    gradient) on that input."""
    target = x.unsqueeze(1).expand_as(x_recon)            # (B,H,784)
    se = ((x_recon - target) ** 2).mean(dim=2)            # (B,H) per neuron-sample
    if GATING:
        gate = (activity.detach() > 0).float()            # (B,H) 1 if neuron fired
        return (se * gate).sum() / gate.sum().clamp(min=1)
    return se.mean()


def train(network, loader, opti, epoch):
    network.train()
    for i, (img, _) in enumerate(loader):
        opti.zero_grad()
        img = img.to(device)
        x, x_recon, activity = network(img)
        loss = recon_loss(x_recon, x, activity)
        loss.backward()
        opti.step()
        if i % 50 == 0:
            frac_dead = (network.encoder.theta.numel()
                         - (activity.sum(0) > 0).sum().item()) / network.encoder.theta.numel()
            print(f'Train[{epoch}/{epochs}][{i}/{len(loader)}] Loss: {loss.item():.5f} '
                  f'| batch dead-neuron frac: {frac_dead:.2f}')
    return loss


@torch.no_grad()
def test(network, loader, epoch):
    network.eval()
    losses = []
    for img, _ in loader:
        img = img.to(device)
        x, x_recon, activity = network(img)
        losses.append(recon_loss(x_recon, x, activity).item())
    print(f'Test[{epoch}/{epochs}] Loss: {np.mean(losses):.5f}')
    return np.mean(losses)


# ---- setup -----------------------------------------------------------------
print('PyTorch:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| GATING:', GATING)

tfm = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('./data', train=True,  download=True, transform=tfm),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tfm),
                          batch_size=batch_size, shuffle=False)

net = SAE().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for e in range(epochs):
    train(net, train_loader, optimizer, e)
    test(net, test_loader, e)

# ---- visualise -------------------------------------------------------------
W  = net.encoder.fc.weight.detach().cpu().numpy()
W2 = net.decoder[0].weight.detach().cpu().numpy()
W3 = net.decoder[0].bias.detach().cpu().numpy()
for title, mat in [("encoder", W), ("decoder", W2), ("decoder bias", W3)]:
    fig, axes = plt.subplots(4, 5, figsize=(10, 8)); fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        f = mat[i].reshape(28, 28); m = np.abs(f).max() + 1e-9
        ax.imshow(f, cmap='seismic', vmin=-m, vmax=m); ax.set_title(f'neuron {i}'); ax.axis('off')
    plt.tight_layout()
    plt.savefig(fname = f"Spiking_Neural_Networks/Images/SNN_NaN_WTA/gating/{title}_{lr}_{inhib_strength}_{theta_decay}_{theta_plus}.png")

