import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt
import os

EIG_FLOOR = 1e-12
N = 20

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


train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

def _normalize(img):
    """Same per-image normalisation used during AE training."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std  = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    images = _normalize(images)
    total += images.sum(dim=0)   
    n += images.size(0)
mean_image = nn.Flatten()(total / n)        
mean_image = mean_image.to(device)

class NeuronAutoencoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)                       # (20, 784) weight
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x                                        # [batch, 784]
        h = torch.sigmoid(self.encoder(x))                  # [batch, 20]  one latent per neuron
        activations = h
        # neuron i reconstructs the input as  h_i * decoder_weights[i] + decoder_bias[i]
        decoded = (h.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))        # [batch, 20, 784]
        return decoded, features, activations

model = NeuronAutoencoder().to(device)
print(model)

recon_criterion = nn.MSELoss()
optimizer_recon = torch.optim.SGD(model.parameters(), lr=10)

def effective_rank(mat):
    cov = torch.cov(mat)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clip(eigenvalues, 0.0, None)
    total = torch.sum(eigenvalues)
    if total <= EIG_FLOOR:
        return float("nan")
    p = eigenvalues/total
    p = p[p>EIG_FLOOR]
    entropy = torch.sum(-(p*torch.log(p)))
    r_eff = torch.exp(entropy)
    return r_eff

def lat_inhib_loss(mat):
    cov = torch.cov(mat)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clip(eigenvalues, 0.0, None)
    total = torch.sum(eigenvalues)
    if total <= EIG_FLOOR:
        return float("nan")
    p = eigenvalues/total
    p = p[p>EIG_FLOOR]
    entropy = torch.sum(-(p*torch.log(p)))
    r_eff = torch.exp(entropy)
    r_spec_loss = 1-r_eff/N
    return r_spec_loss  

inhib_scaler = 0.002

def train_recon(data_loader, model, recon_criterion, optimizer):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        decoded, features, activations = model(_normalize(data))
        

        recon_loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        inhib_loss = lat_inhib_loss(activations.T)
        loss = recon_loss + inhib_scaler*(inhib_loss)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/num_batches
    print(f"Average loss: {train_loss:7f}")

class LinearReadout(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc1 = nn.Linear(784, N)
        self.fc1.weight = nn.Parameter(weight)
        self.fc2 = nn.Linear(N, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
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

SAVE_DIR = "Internship_W10/NANs/images"
os.makedirs(SAVE_DIR, exist_ok=True)          # create it once, up front
 
TEST_EVERY = 4
SEEDS = range(1)
 

PARAM_SPECS = [
    ("encoder",      lambda net: net.encoder.weight),
    ("decoder",      lambda net: net.decoder_weights),
    ("decoder_bias", lambda net: net.decoder_bias),
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
 
 
def save_feature_grid(mat, title, epoch, rows=int(N/5), cols=5, shape=(28, 28)):
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
    readout = LinearReadout(net.encoder.weight).to(device)
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

@torch.no_grad()
def test_encoder(model, loader):
    model.eval()
    acts = []
    for img, _ in loader:
        img = img.to(device)
        decoded, features, activations = model(_normalize(img))
        acts.append(activations)
    r_eff = effective_rank(torch.cat(acts, dim=0))
    return r_eff.item()


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
 
from collections import defaultdict

learning_rate = 1
epochs = 20

for seed in SEEDS:
    torch.manual_seed(seed)
    net = NeuronAutoencoder().to(device)
    optimizer = torch.optim.SGD([
        {"params": net.encoder.parameters(), "lr": learning_rate},
        {"params": net.decoder_weights,    "lr": learning_rate},
        {"params": net.decoder_bias, "lr": learning_rate}
    ])
 
    metrics = defaultdict(list)                             # scalar time-series
    param_hist = {name: defaultdict(list) for name, _ in PARAM_SPECS}
 
    for e in range(epochs + 1):
        if e % TEST_EVERY == 0:
            metrics["epoch"].append(e)
 
            r_eff = test_encoder(net, test_loader)
            metrics["r_eff"].append(r_eff)
 
            for name, getter in PARAM_SPECS:
                mat = to_numpy(getter(net))
                if mat.ndim == 2:                           # bias is 1-D, skip grid
                    save_feature_grid(mat, name, e)
                param_hist[name]["mean"].append(mat.mean())
                param_hist[name]["std"].append(mat.std())   # ddof=0; use ddof=1 to match torch
                param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                param_hist[name]["eff_rank"].append(alt_effective_rank(mat))
 
 
            metrics["accuracy"].append(train_readout(net, seed))

        if e == epochs:
            break

        train_recon(train_loader, net, recon_criterion, optimizer)
        
    x = metrics["epoch"]
 
    save_line_plot(x, [(None, metrics["accuracy"], None)],
                   "Read-out Test Accuracy across Epochs",
                   "Test Set Accuracy", "Test_accuracy.png")
    save_line_plot(x, [(None, metrics["r_eff"], None)],
                   "Encoder Activation Effective Rank across Epochs",
                   "R_eff", "eff_rank.png")
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