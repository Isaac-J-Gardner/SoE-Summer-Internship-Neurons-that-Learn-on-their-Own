import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import utils
from snntorch import surrogate



# ============================================================================
# CONFIG  --  everything below the "standard" block is NEW vs the plain NaN
# ============================================================================
# ---- standard ----
batch_size  = 64
lr          = 1       # NOTE: you had 10. With gating the loss is normalised by
                         #   the number of ACTIVE neuron-sample pairs, so re-sweep;
                         #   lr=10 SGD will almost certainly diverge here.
epochs      = 3

# ---- spiking ----
beta        = 0.5        # membrane decay
num_steps   = 20          # timesteps per image
thresh      = 1.0        # base spiking threshold

inhib_weight = 0.01
inhib_leak = 0.98
Lag_steps = 1 #the window that is checked for simultaneous firing, if 1, checks 1 before and 1 after, a window of size 3 time steps.

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
    def __init__(self, n_in=784, n_hidden=20, beta=0.9):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=thresh)

        W = torch.zeros(n_hidden, n_hidden)
        self.register_buffer("W_inh", W)

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):

            cur = self.fc(x) + spk @ self.W_inh.t()
            spk, mem = self.lif(cur, mem)

            spk_rec.append(spk)
            mem_rec.append(mem)

        return spk_rec, mem_rec

    @torch.no_grad()
    def update_inhibition(self, spk_rec, inhib_weight, max_lag=5, decay=0.5, leak=1e-3):
        S = torch.stack(spk_rec, dim=0)                 # [T, B, N] — time kept separate
        T, B, _ = S.shape
        Sc = S - S.mean(dim=(0, 1), keepdim=True)       # centre per-neuron over (t, b)

        # zero-lag term (already symmetric)
        C = torch.einsum('tbi,tbj->ij', Sc, Sc) / (T * B)

        # lagged terms, symmetrised so i-leads-j and j-leads-i both count
        for tau in range(1, min(max_lag, T - 1) + 1):
            w = decay ** tau
            n = (T - tau) * B
            C_tau = torch.einsum('tbi,tbj->ij', Sc[:T - tau], Sc[tau:]) / n
            C = C + w * (C_tau + C_tau.t())

        C.fill_diagonal_(0)
        self.W_inh.mul_(1 - leak).add_(-inhib_weight * C)
        #self.W_inh.clamp_(max=0.0)                       # drop for two-sided



class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WTASpikingEncoder(784, 20, beta=beta)
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
        spk_rec_list, _ = self.encode(x)
        spk_rec = torch.stack(spk_rec_list, dim=2)        # (B, H, T)

        # NEW: per-(sample, neuron) activity = total spikes over time -> the gate
        activity = spk_rec.sum(dim=2)                # (B, H)

        # decode: integrate each neuron's reconstruction over time
        spk_mem2 = []
        for step in range(num_steps):
            _, x_mem_recon = self.decode(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]   # membrane at last step (B,H,784)
        return spk_rec_list, x, out, activity

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
    return se.mean()


def train(network, loader, opti, epoch):
    network.train()
    for i, (img, _) in enumerate(loader):
        opti.zero_grad()
        img = img.to(device)
        spk_rec, x, x_recon, activity = network(img)
        loss = recon_loss(x_recon, x, activity)
        loss.backward()
        opti.step()
        network.encoder.update_inhibition(spk_rec, inhib_weight=inhib_weight)
        if i % 50 == 0:
            print(f'Train[{epoch}/{epochs}][{i}/{len(loader)}] Loss: {loss.item():.5f} ')


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


torch.manual_seed(0)

net = SAE().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for e in range(epochs):
    train(net, train_loader, optimizer, e)

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
    plt.show()
    plt.close()

W4 = net.encoder.W_inh.detach().cpu().numpy()
fig, ax = plt.subplots()
fig.suptitle("Inhibition Weights")
m = np.abs(W4).max() + 1e-9
im = ax.imshow(W4, cmap='seismic', vmin=-m, vmax=m)
fig.colorbar(im, ax=ax)
ax.set_xlabel("source neuron j")
ax.set_ylabel("target neuron i")
plt.tight_layout()
plt.show()
plt.close(fig)

