import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import utils as utls

import snntorch as snn
from snntorch import utils
from snntorch import surrogate

import numpy as np

# ---------------------------------------------------------------------------
# HETEROGENEITY CONFIG  (maps to the four conditions in Perez-Nieves et al. 2021)
#   HET_INIT = False, LEARN_BETA = False -> homog init,   standard train (your original)
#   HET_INIT = True,  LEARN_BETA = False -> heterog init, standard train
#   HET_INIT = False, LEARN_BETA = True  -> homog init,   heterog train (one shared learned beta)
#   HET_INIT = True,  LEARN_BETA = True  -> heterog init, heterog train (their best condition)
# ---------------------------------------------------------------------------
HET_INIT   = True    # per-neuron beta drawn from a distribution vs a single shared value
LEARN_BETA = True    # train the betas (heterogeneous training) vs hold them fixed
HET_DECODER = False  # also make the decoder integrator's rate heterogeneous (optional/secondary)


def init_hetero_betas(n_neurons, dt=0.5, tau_mean=20.0, tau_min=1.5, tau_max=100.0):
    """Sample per-neuron membrane decay factors beta = exp(-dt/tau).

    Mirrors the paper: tau ~ Gamma(k=3, scale=tau_mean/3), then convert to a decay
    factor and clip tau to [3*dt, 100 ms] (=> beta roughly in [0.717, 0.995]).

    NOTE: with tau_mean=20 and dt=0.5 these betas sit near ~0.97 (slow, long memory),
    which is a different dynamical regime from your original beta=0.5 (fast). If you
    want heterogeneity centred on your current regime instead, just replace the body
    with e.g.:  return torch.empty(n_neurons).uniform_(0.3, 0.9)
    """
    tau = torch.distributions.Gamma(concentration=3.0, rate=3.0 / tau_mean).sample((n_neurons,))
    tau = tau.clamp(tau_min, tau_max)
    return torch.exp(-dt / tau)


def make_beta(n_neurons):
    """Return the beta argument for an snn.Leaky layer given the config."""
    if HET_INIT:
        return init_hetero_betas(n_neurons)          # tensor, shape (n_neurons,)
    return beta                                       # original scalar (defined below)


def clamp_betas(network, lo=0.01, hi=0.995):
    """Reproduce the paper's clipping of the decay factor after each optimiser step.
    snntorch does NOT clamp a learnable beta, so without this it can leave (0,1) and
    the membrane potential diverges."""
    with torch.no_grad():
        for m in network.modules():
            if isinstance(m, snn.Leaky) and isinstance(m.beta, torch.nn.Parameter):
                m.beta.clamp_(lo, hi)


class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        # h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   # (batch, n_neurons, in_dim)


# SAE class
class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder -- this is the layer analogous to the paper's heterogeneous hidden layer.
        # beta is now (optionally) a per-neuron tensor and (optionally) learnable.
        self.encoder = nn.Sequential(
            nn.Linear(784, 20),
            snn.Leaky(beta=make_beta(20),
                      learn_beta=LEARN_BETA,
                      spike_grad=spike_grad,
                      init_hidden=True, output=True, threshold=thresh),
        )

        # Decoder -- threshold set huge so it never spikes (pure leaky integrator readout,
        # analogous to the paper's readout layer with Uth = inf). Kept homogeneous by
        # default. If HET_DECODER, give each latent-decoder its own integration rate;
        # the decoder membrane is (batch, 20, 784), so the per-neuron beta needs shape (20, 1).
        if HET_DECODER:
            dec_beta = init_hetero_betas(20).unsqueeze(-1)   # (20, 1) -> broadcasts over 784
            dec_learn = LEARN_BETA
        else:
            dec_beta = beta
            dec_learn = False

        self.decoder = nn.Sequential(
            NeuronDecoder(20, 784),
            snn.Leaky(beta=dec_beta,
                      learn_beta=dec_learn,
                      spike_grad=spike_grad,
                      init_hidden=True, output=True, threshold=20000),  # large so membrane trains
        )

    def forward(self, x):
        utils.reset(self.encoder)   # need to reset the hidden states of LIF
        utils.reset(self.decoder)

        x = x.view(x.size(0), -1)   # [64, 1, 28, 28] -> [64, 784]

        # encode
        spk_mem = []; spk_rec = []; encoded_x = []
        for step in range(num_steps):                 # for t in time
            spk_x, mem_x = self.encode(x)             # output spike trains and membrane states
            spk_rec.append(spk_x)
            spk_mem.append(mem_x)
        spk_rec = torch.stack(spk_rec, dim=2)
        spk_mem = torch.stack(spk_mem, dim=2)

        # decode
        spk_mem2 = []; spk_rec2 = []; decoded_x = []
        for step in range(num_steps):                 # for t in time
            x_recon, x_mem_recon = self.decode(spk_rec[..., step])
            spk_rec2.append(x_recon)
            spk_mem2.append(x_mem_recon)
        spk_rec2 = torch.stack(spk_rec2, dim=3)
        spk_mem2 = torch.stack(spk_mem2, dim=3)
        out = spk_mem2[:, :, :, -1]   # membrane potential of the output neuron at t = -1
        return x, out

    def encode(self, x):
        spk_latent_x, mem_latent_x = self.encoder(x)
        return spk_latent_x, mem_latent_x

    def decode(self, x):
        spk_x2, mem_x2 = self.decoder(x)
        return spk_x2, mem_x2


# Training
def train(network, trainloader, opti, epoch):
    network = network.train()
    train_loss_hist = []
    for batch_idx, (real_img, labels) in enumerate(trainloader):
        opti.zero_grad()
        real_img = real_img.to(device)
        labels = labels.to(device)

        # Pass data into network, return reconstruction from membrane potential at t = -1
        x, x_recon = network(real_img)

        # Calculate loss
        loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))  # x: [64, 784]

        print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item()}')

        loss_val.backward()
        opti.step()
        clamp_betas(network)   # keep learnable decay factors in a stable range (paper's clip)
    return loss_val


# Testing
def test(network, testloader, opti, epoch):
    network = network.eval()
    test_loss_hist = []
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(testloader):
            real_img = real_img.to(device)
            labels = labels.to(device)
            x, x_recon = network(real_img)

            loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))  # x: [64, 784]
            print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  Loss: {loss_val.item()}')
    return loss_val


# setup GPU
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

batch_size = 64

data_dir = './data'
print('data_dir =', data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# SNN parameters
spike_grad = surrogate.atan(alpha=2.0)  # alternate surrogate gradient: fast_sigmoid(slope=25)
beta = 0.5      # decay rate of neurons (used as the scalar/homogeneous value)
num_steps = 5   # time
thresh = 1      # spiking threshold (lower = more spikes are let through)
epochs = 1      # number of epochs
max_epoch = epochs

# Define Network and optimizer
net = SAE()
net = net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Run training and testing
for e in range(epochs):
    train_loss = train(net, train_loader, optimizer, e)
    test_loss = test(net, test_loader, optimizer, e)

# --- inspect the learned time-constant heterogeneity ---------------------------
enc_leaky = net.encoder[1]
betas_learned = enc_leaky.beta.detach().cpu().numpy().reshape(-1)
print('Encoder betas (per-neuron decay factors):', np.round(betas_learned, 4))
# convert back to approximate membrane time constants for comparison with the paper
dt = 0.5
taus = -dt / np.log(np.clip(betas_learned, 1e-6, 0.999999))
print('Approx per-neuron tau_m (ms):', np.round(taus, 2))

plt.figure(figsize=(5, 3))
plt.hist(taus, bins=15)
plt.xlabel('membrane time constant (ms)')
plt.ylabel('neuron count')
plt.title('Learned time-constant distribution')
plt.tight_layout()
plt.show()

# --- your original weight visualisations ---------------------------------------
W  = net.encoder[0].weight.detach().cpu().numpy()   # (20, 784)
W2 = net.decoder[0].weight.detach().cpu().numpy()
W3 = net.decoder[0].bias.detach().cpu().numpy()

for title, mat in [('encoder', W), ('decoder', W2), ('decoder bias', W3)]:
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        filt = mat[i].reshape(28, 28)
        ax.imshow(filt, cmap='seismic',
                  vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric, centred at 0
        ax.set_title(f'{title} {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()