import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate

import numpy as np


# ----------------------------------------------------------------------------- 
# Per-neuron decoder: each hidden neuron has its OWN 784-dim decoder (weight+bias).
# The `gate` argument lets us stop the weight gradient for neurons that did not win,
# WITHOUT changing the forward value and WITHOUT touching the bias gradient.
# -----------------------------------------------------------------------------
class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h, gate=None):             # h: (batch, n_neurons)
        weighted = h.unsqueeze(-1) * self.weight              # (batch, n_neurons, in_dim)
        if gate is not None:
            g = gate.unsqueeze(-1)                            # (batch, n_neurons, 1), in {0,1}
            # winners: keep gradient to weight; losers: forward value identical but
            # gradient to `weight` (and to the encoder spikes) is detached.
            weighted = g * weighted + (1.0 - g) * weighted.detach()
        return weighted + self.bias                          # bias grad ALWAYS flows -> all biases train


# -----------------------------------------------------------------------------
# WTA encoder with two homeostatic mechanisms on top of the lateral inhibition:
#   1. Fast spike-frequency adaptation `a`  (per sample & neuron, resets each pass)
#        -> the current winner fatigues within the 5-step window so someone else
#           can take over on later steps -> richer codes inside one forward pass.
#   2. Slow homeostatic threshold `theta`   (per neuron, persists across batches)
#        -> neurons that fire above the population average get a higher standing
#           threshold; under-active neurons get a lower (even negative) one.
#           This is what stops the "rich get richer" collapse over training.
# Both are added to the input current as negative (self-inhibitory) terms, which is
# equivalent to raising that neuron's threshold. Both are biologically plausible
# (SFA + intrinsic plasticity), which is the "diversity pressure via inhibition"
# the reports were after.
# -----------------------------------------------------------------------------
class AdaptiveWTASpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=20, beta=0.9, thresh=1.0,
                 spike_grad=None, inhib_strength=1.0,
                 adapt_inc=1.0, adapt_decay=0.8,          # fast SFA
                 theta_lr=0.1, theta_decay=1e-4, theta_clamp=5.0,  # slow homeostasis
                 max_norm=1.0):                           # synaptic scaling (None to disable)
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, threshold=thresh, spike_grad=spike_grad)

        self.adapt_inc   = adapt_inc
        self.adapt_decay = adapt_decay
        self.theta_lr    = theta_lr
        self.theta_decay = theta_decay
        self.theta_clamp = theta_clamp
        self.max_norm    = max_norm

        # every neuron inhibits every other; none inhibits itself (fixed, not learned)
        W = -inhib_strength * (torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden))
        self.register_buffer("W_inh", W)
        # slow per-neuron adaptive threshold, persists across batches (not a Parameter)
        self.register_buffer("theta", torch.zeros(n_hidden))

    @torch.no_grad()
    def renormalize(self):
        """Cap each neuron's input-weight L2 norm so its feedforward drive can't run
        away (synaptic scaling). Call once per optimizer step. This is what lets the
        bounded adaptive threshold actually suppress an over-active neuron."""
        if self.max_norm is None:
            return
        w = self.fc.weight                                   # (n_hidden, n_in)
        norms = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
        w.mul_((self.max_norm / norms).clamp(max=1.0))       # only shrink rows above max_norm

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        a   = torch.zeros(x.shape[0], self.n_hidden, device=x.device)  # fast adaptation state
        spk_rec, mem_rec = [], []

        drive = self.fc(x)                        # feedforward drive (constant over the window)
        for _ in range(num_steps):
            # feedforward + lateral inhibition + fast adaptation + slow homeostatic threshold
            cur = drive + spk @ self.W_inh.t() - a - self.theta
            spk, mem = self.lif(cur, mem)
            a = self.adapt_decay * a + self.adapt_inc * spk   # fatigue whoever just fired
            spk_rec.append(spk)
            mem_rec.append(mem)

        # --- slow homeostatic update: nudge thresholds to equalise firing rates ---
        if self.training:
            with torch.no_grad():
                rate = torch.stack(spk_rec, dim=0).mean(dim=(0, 1))   # (n_hidden,) mean over time+batch
                target = rate.mean()                                  # population average firing rate
                self.theta += self.theta_lr * (rate - target)         # over-active -> higher threshold
                self.theta.mul_(1.0 - self.theta_decay)               # gentle leak toward 0
                self.theta.clamp_(-self.theta_clamp, self.theta_clamp)

        return spk_rec, mem_rec   # each: list length T of (batch, n_hidden)


# -----------------------------------------------------------------------------
# Spiking Neuron-Autoencoder (SAE)
# -----------------------------------------------------------------------------
class SAE(nn.Module):
    def __init__(self, n_in=784, n_hidden=20, num_steps=5, beta=0.5, thresh=1.0,
                 spike_grad=None, inhib_strength=2.0, k_winners=1):
        super().__init__()
        self.num_steps = num_steps
        self.n_hidden  = n_hidden
        self.k_winners = k_winners            # k-WTA: how many neurons may learn per image (1 = hard WTA)

        self.encoder = AdaptiveWTASpikingEncoder(
            n_in, n_hidden, beta=beta, thresh=thresh,
            spike_grad=spike_grad, inhib_strength=inhib_strength,
        )
        # decoder = per-neuron projection (gateable) followed by a leaky integrator read out.
        # NOTE: membrane is managed manually (init_hidden=False) and re-initialised every
        # forward pass. Relying on init_hidden=True + utils.reset retains the autograd graph
        # across batches in recent snnTorch, which crashes the 2nd backward.
        self.dec     = NeuronDecoder(n_hidden, n_in)
        self.dec_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=20000)  # huge -> never spikes

    def _winner_gate(self, counts):
        """counts: (batch, n_hidden) spike totals -> (batch, n_hidden) 0/1 top-k mask."""
        if self.k_winners >= self.n_hidden:
            return torch.ones_like(counts)
        idx  = counts.topk(self.k_winners, dim=1).indices
        gate = torch.zeros_like(counts)
        gate.scatter_(1, idx, 1.0)
        return gate

    def forward(self, x, return_gate=False):
        x = x.view(x.size(0), -1)               # [B,1,28,28] -> [B,784]

        # encode
        spk_rec, _ = self.encoder(x, self.num_steps)
        spk_rec = torch.stack(spk_rec, dim=2)   # (batch, n_hidden, T)

        # decide winners over the whole window (detached: the mask carries no gradient)
        counts = spk_rec.sum(dim=2).detach()    # (batch, n_hidden)
        gate   = self._winner_gate(counts)      # (batch, n_hidden)

        # decode: accumulate membrane over time, read out the final step
        mem = self.dec_lif.init_leaky()         # fresh membrane each forward pass
        mem_last = None
        for step in range(self.num_steps):
            proj = self.dec(spk_rec[..., step], gate)          # (batch, n_hidden, 784)
            _, mem = self.dec_lif(proj, mem)
            mem_last = mem
        out = mem_last                          # membrane potential at t = -1

        if return_gate:
            return x, out, gate
        return x, out


# -----------------------------------------------------------------------------
# Train / test
# -----------------------------------------------------------------------------
def train(network, trainloader, opti, epoch, device, max_epoch):
    network.train()
    win_counts = torch.zeros(network.n_hidden)
    for batch_idx, (real_img, labels) in enumerate(trainloader):
        opti.zero_grad()
        real_img = real_img.to(device)

        x, x_recon, gate = network(real_img, return_gate=True)
        # each neuron's reconstruction is compared to the input (per-neuron autoencoder)
        loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))

        loss_val.backward()
        opti.step()
        network.encoder.renormalize()          # synaptic scaling: keep drive bounded

        win_counts += gate.sum(dim=0).detach().cpu()
        if batch_idx % 100 == 0:
            print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item():.5f}')
    print('  win counts / neuron:', win_counts.int().tolist())
    return loss_val


def test(network, testloader, epoch, device, max_epoch):
    network.eval()
    win_counts = torch.zeros(network.n_hidden)
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(testloader):
            real_img = real_img.to(device)
            x, x_recon, gate = network(real_img, return_gate=True)
            loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))
            win_counts += gate.sum(dim=0).cpu()
    print(f'Test[{epoch}/{max_epoch}] Loss: {loss_val.item():.5f}')
    print('  test win counts / neuron:', win_counts.int().tolist())
    return loss_val, win_counts


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    from torchvision import datasets, transforms
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU found, using CPU instead.')
        device = torch.device('cpu')

    batch_size = 64
    data_dir = './data'

    train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transforms.ToTensor())
    test_dataset  = datasets.MNIST(data_dir, train=False,               transform=transforms.ToTensor())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # SNN parameters
    spike_grad = surrogate.atan(alpha=2.0)
    beta       = 0.5
    num_steps  = 20
    thresh     = 1.0
    epochs     = 1
    max_epoch  = epochs

    net = SAE(n_in=784, n_hidden=20, num_steps=num_steps, beta=beta, thresh=thresh,
              spike_grad=spike_grad, inhib_strength=2.0, k_winners=1).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=10)

    for e in range(epochs):
        train_loss = train(net, train_loader, optimizer, e, device, max_epoch)
        test_loss, win_counts = test(net, test_loader, e, device, max_epoch)

    # ------- visualise learned filters -------
    W  = net.encoder.fc.weight.detach().cpu().numpy()   # (20, 784) encoder weights
    W2 = net.dec.weight.detach().cpu().numpy()          # (20, 784) decoder weights
    W3 = net.dec.bias.detach().cpu().numpy()            # (20, 784) decoder biases

    for title, WW in [('encoder weights', W), ('decoder weights', W2), ('decoder biases', W3)]:
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        fig.suptitle(title)
        for i, ax in enumerate(axes.flat):
            filt = WW[i].reshape(28, 28)
            m = np.abs(filt).max() + 1e-9
            ax.imshow(filt, cmap='seismic', vmin=-m, vmax=m)
            ax.set_title(f'n{i} ({int(win_counts[i])})', fontsize=8)
            ax.axis('off')
        plt.tight_layout()
    plt.show()