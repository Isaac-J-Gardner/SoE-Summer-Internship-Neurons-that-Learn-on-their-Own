#this code involves the a spiking NaN trained on MSE error, every few epochs, an standard MLP is created with the same weights as the NaN encoder, and a readout layer is trained
#this code trials using differing learning rates forthe decoder bias to observe how this effects the encoders ability to learn on raw MNIST (does it collapse)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import utils
from snntorch import surrogate


EIG_FLOOR = 1e-12 #Used when calculating effective rank, removes zeros to prevent NaNs

batch_size  = 100

epochs      = 20

N = 5 #number of hidden neurons (neuron autoencoders)

num_steps   = 50     #5 time constants, each image is shown for 50 time steps
theta      = 2.0     #starting threshold for each hidden neuron (readout threshold are set really high, potential is used to infer recon)
beta = 0.9           #membrane decay rate, produces time constant of ~10 time steps

alpha = 1 #mutual inhibition learning rate
learning_beta = 0.01 #task loss optimiser learning rate
gamma = 0.1 #adaptive threshold learning rate
p = 0.05 #desired firing rate (per time step, neurons should fire 2.5 times per image)

GATING = True

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

data_dir = './data'
print('data_dir =', data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

sur_grad = surrogate.atan(alpha = 2.0)
class SpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=N, beta=0.9):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, spike_grad=sur_grad, threshold=theta)

        W = torch.zeros(n_hidden, n_hidden)
        self.register_buffer("W_inh", W)

        self.register_buffer("theta", torch.zeros(n_hidden))

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):

            cur = self.fc(x) - (spk @ self.W_inh.t()).detach() - self.theta.detach()
            spk, mem = self.lif(cur, mem)

            spk_rec.append(spk)
            mem_rec.append(mem)

        return spk_rec, mem_rec

    @torch.no_grad()
    def update_inhibition(self, activity, alpha):
        coinc = (activity.T @ activity) / activity.shape[0]   # <n_i n_m>
        dW = coinc - (p*num_steps)**2                               # n_i n_m - p^2
        dW.fill_diagonal_(0)
        self.W_inh.add_(alpha * dW)
        self.W_inh.clamp_(min=0.0)

    @torch.no_grad()
    def update_threshold(self, activity, gamma): #SAILnet thresholding rule
        batch_dtheta = gamma*(activity - p*num_steps)
        dtheta = torch.mean(batch_dtheta, dim=0) #average the change across all batches
        self.theta.add_(dtheta)

def _normalize(img):
    """Same per-image normalisation used during AE training."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std  = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std

class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpikingEncoder(784, N, beta=0.9)

    def forward(self, x):
        utils.reset(self.encoder)
        x = x.view(x.size(0), -1)                    # (B,784)

        # encode
        spk_rec_list, _ = self.encode(x)
        spk_rec = torch.stack(spk_rec_list, dim=2)        # (B, H, T)

        # NEW: per-(sample, neuron) activity = total spikes over time -> the gate
        activity = spk_rec.sum(dim=2)                # (B, H)

        return spk_rec_list, x, activity

    def encode(self, x):
        return self.encoder(x, num_steps=num_steps)

    def learn(self, activity, X):
        # Q_ik : encoder weight, shape [H, N_in] = [20, 784]
        Q = self.encoder.fc.weight                       # [H, N_in]
        N_in = Q.shape[1]

        # per-(sample, neuron) reconstruction:  x̄_k^(i) = n_i * Q_ik
        nq = activity.unsqueeze(2) * Q.unsqueeze(0)      # [B, H, N_in]

        # reconstruction error:  (x_k - n_i Q_ik)
        err = X.unsqueeze(1) - nq                        # [B, H, N_in]

        # gradient  dL/dQ_ik = -(2/N_in) * n_i * (x_k - n_i Q_ik)
        grad = -(2.0 / N_in) * activity.unsqueeze(2) * err   # [B, H, N_in]
        grad = grad.mean(dim=0)                          # average over batch -> [H, N_in]

        # local gradient-descent update (no autograd)
        with torch.no_grad():
            Q -= learning_beta * grad

        # optional: return the reconstruction loss for monitoring
        return err.pow(2).mean().item()

def recon_loss(x_recon, x, activity):
    target = x.unsqueeze(1).expand_as(x_recon)            #(B,H,784)
    se = ((x_recon - target) ** 2).mean(dim=2)            #(B,H) per neuron-sample
    if GATING:
        gate = (activity.detach() > 0).float()            #(B,H) 1 if neuron fired
        return (se * gate).sum() / gate.sum().clamp(min=1) #averaging gated squared error
    return se.mean()


def train(network, loader, epoch):
    network.train()
    for i, (img, _) in enumerate(loader):
        img = img.to(device)
        spk_rec, x, activity = network(img)
        loss = network.learn(activity, x)
        network.encoder.update_inhibition(activity, alpha)
        network.encoder.update_threshold(activity, gamma)
        if i % 50 == 0:
            print(f'Train[{epoch+1}/{epochs}][{i}/{len(loader)}] Loss: {loss:.5f} ')


    return loss

def spk_effective_rank(spk_rec, n_neurons=N, eig_floor=EIG_FLOOR, jitter=1e-6):
    N = n_neurons
    spikes = spk_rec.permute(1, 0, 2).reshape(N, -1)      # (N, batch*num_steps)

    cov = torch.cov(spikes)                               # (N, N)

    cov = cov + jitter * torch.eye(N, device=cov.device, dtype=cov.dtype)

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clip(eigvals, 0.0, None)
    total = eigvals.sum()

    #If nothing spiked in the whole batch, return a graph-connected zero
    #(rather than nan)
    if total <= eig_floor:
        return spikes.sum() * 0.0

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = torch.sum(-(p * torch.log(p)))
    r_eff = torch.exp(entropy)                            #effective rank
    return r_eff

def alt_effective_rank(mat, eig_floor=EIG_FLOOR, jitter = 1e-6):
    cov = np.cov(mat)
    
    cov = cov + jitter * np.eye(N, device=cov.device, dtype=cov.dtype)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = np.sum(-(p * np.log(p)))
    r_eff = np.exp(entropy)                            #effective rank
    return r_eff


@torch.no_grad()
def test_encoder(network, loader):
    network.eval()
    losses = []
    spikes = []
    for img, _ in loader:
        img = img.to(device)
        spk_rec, x, activity = network(img)
        spikes.append(torch.cat(spk_rec, dim=0))
    spikes = torch.cat(spikes, dim=0)
    r_eff = spk_effective_rank(spikes.unsqueeze(-1))
    avg_rate = spikes.float().mean()
    thresholds = network.encoder.theta
    inhibition = network.encoder.W_inh
    avg_thresh = thresholds.mean()
    inhibition_prop = (inhibition > 0).float().mean() #proportion of existing inhibition weights over possible inhibition weights
    inhibition_prop = inhibition_prop * 20/19 #scaling to account for zeroed diagonal
    return r_eff.item(), avg_rate.item(), avg_thresh.item(), inhibition_prop.item()

class LinearReadout(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc1 = nn.Linear(784, N)
        self.fc1.weight = nn.Parameter(weight)
        self.fc2 = nn.Linear(N, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x.detach())
        return x

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()

def train_linear(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(_normalize(data))

        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")
    return train_loss

def test_linear(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(_normalize(data))

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return accuracy

SAVE_DIR = "Internship_W10/Spiking_NaNs/images"
os.makedirs(SAVE_DIR, exist_ok=True)          # create it once, up front
 
TEST_EVERY = 4
SEEDS = range(1)
 

PARAM_SPECS = [
    ("encoder",      lambda net: net.encoder.fc.weight),
]
 

def to_numpy(t):
    return t.detach().cpu().numpy()
 
 
def save_line_plot(x, series, title, ylabel, filename, xlabel="Epoch"):
    """Plot one or more series and save into SAVE_DIR.
 
    series: list of (label, y, yerr) tuples.
            label=None  -> single unlabelled line (no legend)
            yerr=None   -> plain line, otherwise an error bar
    """
    plt.figure()
    for label, y, yerr in series:
        if yerr is None:
            plt.plot(x, y, label=label)
        else:
            plt.errorbar(x, y, yerr, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if any(label is not None for label, _, _ in series):
        plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()
 
 
def save_feature_grid(mat, title, epoch, rows=4, cols=5, shape=(28, 28)):
    """Show the first rows*cols rows of a 2-D weight matrix as images."""
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        f = mat[i].reshape(shape)
        m = np.abs(f).max() + 1e-9
        ax.imshow(f, cmap="seismic", vmin=-m, vmax=m)
        ax.set_title(f"neuron {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"{title}_{epoch}.png"))
    plt.close(fig)
 
 
def save_matrix(mat, title, epoch, filename_prefix,
                xlabel="source neuron j", ylabel="target neuron i"):
    """Show a single 2-D matrix (e.g. inhibition weights) with a colour bar."""
    fig, ax = plt.subplots()
    fig.suptitle(title)
    m = np.abs(mat).max() + 1e-9
    im = ax.imshow(mat, cmap="seismic", vmin=-m, vmax=m)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"{filename_prefix}_{epoch}.png"))
    plt.close(fig)
 
 
def train_readout(net, seed):
    """Train a fresh linear read-out on top of the current encoder features."""
    torch.manual_seed(seed + 100)             # identical init for every read-out
    readout = LinearReadout(net.encoder.fc.weight).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
 
    prev_loss, curr_loss, i = 1.0, 0.0, 0
    while (i < 3) or (prev_loss - curr_loss > 0.01):
        prev_loss = curr_loss
        curr_loss = train_linear(train_loader, readout, criterion, optimizer)
        print(f"Read-out epoch {i + 1}: loss {curr_loss:.4f} "
              f"(diff {prev_loss - curr_loss:.4f})")
        i += 1
    return test_linear(test_loader, readout, criterion)
 
 
from collections import defaultdict

for seed in SEEDS:
    torch.manual_seed(seed)
    net = SAE().to(device)
    metrics = defaultdict(list)                             # scalar time-series
    param_hist = {name: defaultdict(list) for name, _ in PARAM_SPECS}
 
    for e in range(epochs + 1):
        if e % TEST_EVERY == 0:
            metrics["epoch"].append(e)
 
            r_eff, avg_rate, avg_thresh, inhib_prop = test_encoder(net, test_loader)
            metrics["r_eff"].append(r_eff)
            metrics["avg_rate"].append(avg_rate)
            metrics["avg_thresh"].append(avg_thresh)
            metrics["inhibition_prop"].append(inhib_prop)
 
            for name, getter in PARAM_SPECS:
                mat = to_numpy(getter(net))
                if mat.ndim == 2:                           # bias is 1-D, skip grid
                    save_feature_grid(mat, name, e)
                param_hist[name]["mean"].append(mat.mean())
                param_hist[name]["std"].append(mat.std())   # ddof=0; use ddof=1 to match torch
                param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                param_hist[name]["eff_rank"].append(alt_effective_rank(mat))
 
            save_matrix(to_numpy(net.encoder.W_inh), "Inhibition Weights", e, "inhib")
 
            metrics["accuracy"].append(train_readout(net, seed))

        if e == epochs:
            break

        train(net, train_loader, e)
 

    x = metrics["epoch"]
 
    save_line_plot(x, [(None, metrics["accuracy"], None)],
                   "Read-out Test Accuracy across Epochs",
                   "Test Set Accuracy", "Test_accuracy.png")
    save_line_plot(x, [(None, metrics["r_eff"], None)],
                   "Encoder Activation Effective Rank across Epochs",
                   "R_eff", "eff_rank.png")
    save_line_plot(x, [(None, metrics["avg_rate"], None)],
                   "Average Encoder Firing Rate across Epochs",
                   "Firing Rate", "firing_rate.png")
    save_line_plot(x, [(None, metrics["avg_thresh"], None)],
                   "Average Encoder Neuron Threshold (p=0.05) across Epochs",
                   "Average Threshold", "Threshold.png")
    save_line_plot(x, [(None, metrics["inhibition_prop"], None)],
                   "Proportion of W_inh > 0 across Epochs",
                   "W_inh > 0", "W_inh.png")
 
    save_line_plot(x, [(f"mean |{name}|", param_hist[name]["abs_mean"], None)
                       for name, _ in PARAM_SPECS],
                   "Mean of Absolute Parameters across Epochs",
                   "", "Absolute_Means.png")
    save_line_plot(x, [(f"mean {name}", param_hist[name]["mean"], param_hist[name]["std"])
                       for name, _ in PARAM_SPECS],
                   "Mean (±std) of Signed Parameters across Epochs",
                   "", "Signed_Means.png")
    save_line_plot(x, [(f"{name} r_eff", param_hist[name]["eff_rank"], None)
                       for name, _ in PARAM_SPECS],
                   "Effective Rank of Parameters across Epochs",
                   "R_eff", "params_R_eff.png")