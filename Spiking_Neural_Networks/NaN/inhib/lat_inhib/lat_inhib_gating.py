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

N_NEURONS = 20
EIG_FLOOR = 1e-12
GATING = True

class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=N_NEURONS, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        # h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   # (batch, n_neurons, in_dim)


# ------------------------------------------------------------------
# Spike-based lateral inhibition.
#
# spk_rec : (batch, n_neurons, num_steps)  -- the encoder spike trains.
#
# We reshape to (N, batch*num_steps) so each (sample, timestep) is one
# observation, then take the covariance ACROSS neurons -- identical in spirit
# to torch.cov(activations.T) in the dense version. Because each column is a
# single (sample, timestep), neurons that spike in the SAME step on the SAME
# input become correlated, which collapses the covariance spectrum, drops the
# effective rank, and raises the redundancy penalty (1 - r_eff / N).
# Neurons firing at different times come out anti-correlated -> higher rank ->
# lower penalty, i.e. desynchronizing pressure.
# ------------------------------------------------------------------
def lat_inhib_loss(spk_rec, n_neurons=N_NEURONS, eig_floor=EIG_FLOOR, jitter=1e-6):
    N = n_neurons
    spikes = spk_rec.permute(1, 0, 2).reshape(N, -1)      # (N, batch*num_steps)

    cov = torch.cov(spikes)                               # (N, N)
    # jitter keeps eigvalsh backward well-behaved on the degenerate spectra
    # you get from sparse / dead spike trains (repeated & zero eigenvalues).
    cov = cov + jitter * torch.eye(N, device=cov.device, dtype=cov.dtype)

    eigvals = torch.linalg.eigvalsh(cov)                  # symmetric -> real, differentiable
    eigvals = torch.clip(eigvals, 0.0, None)
    total = eigvals.sum()

    # If nothing spiked in the whole batch, return a graph-connected zero
    # (rather than nan) so the training step still works.
    if total <= eig_floor:
        return spikes.sum() * 0.0

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = torch.sum(-(p * torch.log(p)))
    r_eff = torch.exp(entropy)                            # effective rank
    return 1.0 - r_eff / N                                # spectral redundancy


# SAE class
class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(784, N_NEURONS),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh),
                          )

        self.decoder = nn.Sequential(NeuronDecoder(N_NEURONS, 784),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000)  # make large so membrane can be trained
                          )

    def forward(self, x):
        utils.reset(self.encoder)  # need to reset the hidden states of LIF
        utils.reset(self.decoder)

        x = x.view(x.size(0), -1)   # [64, 1, 28, 28] -> [64, 784]

        # encode
        spk_mem = []; spk_rec = []; encoded_x = []
        for step in range(num_steps):  # for t in time
            spk_x, mem_x = self.encode(x)  # Output spike trains and neuron membrane states
            spk_rec.append(spk_x)
            spk_mem.append(mem_x)
        spk_rec = torch.stack(spk_rec, dim=2)   # (batch, N, num_steps)
        spk_mem = torch.stack(spk_mem, dim=2)

        activity = spk_rec.sum(dim=2)

        # decode
        spk_mem2 = []; spk_rec2 = []; decoded_x = []
        for step in range(num_steps):  # for t in time
            x_recon, x_mem_recon = self.decode(spk_rec[..., step])
            spk_rec2.append(x_recon)
            spk_mem2.append(x_mem_recon)
        spk_rec2 = torch.stack(spk_rec2, dim=3)
        spk_mem2 = torch.stack(spk_mem2, dim=3)
        out = spk_mem2[:, :, :, -1]  # return the membrane potential of the output neuron at t = -1 (last t)
        return x, out, spk_rec, activity       # <-- also return encoder spike trains for the inhibition loss

    def encode(self, x):
        spk_latent_x, mem_latent_x = self.encoder(x)
        return spk_latent_x, mem_latent_x

    def decode(self, x):
        spk_x2, mem_x2 = self.decoder(x)
        return spk_x2, mem_x2

def calc_recon_loss(x_recon, x, activity):
    """Per-neuron reconstruction MSE, optionally GATED by activity so that only
    neurons which fired for a given input contribute (and therefore receive
    gradient) on that input."""
    target = x.unsqueeze(1).expand_as(x_recon)            # (B,H,784)
    se = ((x_recon - target) ** 2).mean(dim=2)            # (B,H) per neuron-sample
    if GATING:
        gate = (activity.detach() > 0).float()            # (B,H) 1 if neuron fired
        return (se * gate).sum() / gate.sum().clamp(min=1)
    return se.mean()

# Training
def train(network, trainloader, opti, epoch):

    network = network.train()
    train_loss_hist = []
    for batch_idx, (real_img, labels) in enumerate(trainloader):
        opti.zero_grad()
        real_img = real_img.to(device)
        labels = labels.to(device)

        # Pass data into network, and return reconstructed image from Membrane Potential at t = -1
        x, x_recon, spk_rec, activity = network(real_img)

        # Reconstruction loss
        recon_loss = calc_recon_loss(x_recon, x, activity)

        # Spike-based lateral inhibition loss
        inhib_loss = lat_inhib_loss(spk_rec)

        loss_val = recon_loss + inhib_scaler * inhib_loss

        if batch_idx%50 == 0:
            print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] '
              f'Loss: {loss_val.item():.5f}  Recon: {recon_loss.item():.5f}  Inhib: {inhib_loss.item():.5f}')

        loss_val.backward()
        opti.step()
    return loss_val


# Testing
def test(network, testloader, opti, epoch):
    network = network.eval()
    test_loss_hist = []
    with torch.no_grad():  # no gradient this time
        for batch_idx, (real_img, labels) in enumerate(testloader):
            real_img = real_img.to(device)
            labels = labels.to(device)
            x, x_recon, spk_rec, _ = network(real_img)

            recon_loss = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))  # x: [64, 784]
            inhib_loss = lat_inhib_loss(spk_rec)
            loss_val = recon_loss + inhib_scaler * inhib_loss

            print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  '
                  f'Loss: {loss_val.item():.5f}  Recon: {recon_loss.item():.5f}  Inhib: {inhib_loss.item():.5f}')
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
beta = 0.9      # decay rate of neurons
num_steps = 5   # time
thresh = 1      # spiking threshold (lower = more spikes are let through)
epochs = 1      # number of epochs
inhib_scaler = 0.1        # weight on the inhibition term (0.0 recovers your original SAE)
max_epoch = epochs
learn_rate = 1
# Define Network and optimizer
net = SAE()
net = net.to(device)

optimizer = torch.optim.SGD(net.parameters(),
                            lr=learn_rate)

# Run training and testing
for e in range(epochs):
    train_loss = train(net, train_loader, optimizer, e)
    #test_loss = test(net, test_loader, optimizer, e)

W  = net.encoder[0].weight.detach().cpu().numpy()   # (20, 784)
W2 = net.decoder[0].weight.detach().cpu().numpy()
W3 = net.decoder[0].bias.detach().cpu().numpy()

for title, mat in [('encoder', W), ('decoder', W2), ('decoder_bias', W3)]:
    fig, axes = plt.subplots(1, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        filt = mat[i].reshape(28, 28)
        ax.imshow(filt, cmap='seismic',
                  vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric, centred at 0
        ax.set_title(f'{title} {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"Spiking_Neural_Networks/NaN/inhib/lat_inhib/images/gating/{title}_{learn_rate:.2e}.png")
    plt.show()