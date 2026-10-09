import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.') 
    device = torch.device('cpu')
    
batch_size = 100 #batch size 100 matches SNNs
epochs = 20

data_dir = './data'
print('data_dir =', data_dir)

def fit_zca(X, eps):
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)                
    evals = torch.clamp(evals, min=0.0)
    W = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
    return mean, W

@torch.no_grad()
def compute_zca(dataset, eps):
    # one big (N, 784) matrix; MNIST is ~188 MB in float32, fine on CPU
    X = torch.stack([img.view(-1) for img, _ in dataset])   # (60000, 784)
    return fit_zca(X, eps)                                   # mean:(1,784)  W:(784,784)

base = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mean, W = compute_zca(base, eps=1e-2)

class ZCAWhiten:
    def __init__(self, mean, W):
        self.mean = mean          # (1, 784), CPU
        self.W = W                # (784, 784), CPU
    def __call__(self, x):        # x: (1, 28, 28)
        flat = x.reshape(1, -1)             # (1, 784)
        white = (flat - self.mean) @ self.W # (1, 784)
        return white.reshape(x.shape)       # (1, 28, 28)

tfm = transforms.Compose([transforms.ToTensor(), ZCAWhiten(mean, W)])
train_loader = DataLoader(datasets.MNIST('./data', train=True,  download=True, transform=tfm),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tfm),
                          batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

N=20

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, N)
        self.fc2 = nn.Linear(N, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x_fc1 = torch.relu(x)
        x = self.fc2(x_fc1)
        return x, x_fc1

def _normalize(img):
    """Same per-image normalisation used during AE training."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std  = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()          

def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        #data = _normalize(data)
        # Do a forward pass
        output, _ = model(data)
        
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



def test(test_loader, model, criterion):
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
            #data = _normalize(data)
        
            # Do a forward pass
            output, _ = model(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")

    return accuracy

EIG_FLOOR = 1e-12
SAVE_DIR = "Internship_W10/MLPs/Images"
os.makedirs(SAVE_DIR, exist_ok=True)          

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

def np_effective_rank(mat):
    cov = np.cov(mat)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= EIG_FLOOR:
            return float("nan")
    p = eigvals / total
    p = p[p > EIG_FLOOR]
    entropy = np.sum(-(p * np.log(p)))
    r_eff = np.exp(entropy)                            #effective rank
    return r_eff

@torch.no_grad()
def test_encoder(network, loader):
    network.eval()
    activations = []
    for img, _ in loader:
        img = img.to(device)
        output, feats = network(img)
        activations.append(feats) 
    activations = torch.cat(activations, dim=0)
    r_eff = effective_rank(activations)
    return r_eff.item()

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
        return x, 1

def train_readout(net, seed):
    """Train a fresh linear read-out on top of the current encoder features."""
    torch.manual_seed(seed + 100)             
    readout = LinearReadout(net.fc1.weight).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
 
    prev_loss, curr_loss, i = 1.0, 0.0, 0
    while (i < 3) or (prev_loss - curr_loss > 0.01):
        prev_loss = curr_loss
        curr_loss = train(train_loader, readout, criterion, optimizer)
        print(f"Read-out epoch {i + 1}: loss {curr_loss:.4f} "
              f"(diff {prev_loss - curr_loss:.4f})")
        i += 1
    return test(test_loader, readout, criterion)

from collections import defaultdict

SEEDS = range(1)
TEST_EVERY = 4

for seed in SEEDS:
    torch.manual_seed(seed)
    net = SimpleMLP().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    metrics = defaultdict(list)     

    for e in range(epochs + 1):
        if e % TEST_EVERY == 0:
            metrics["epoch"].append(e)
 
            act_eff_r = test_encoder(net, test_loader)
            metrics["act_eff_r"].append(act_eff_r)
 
            mat = net.fc1.weight.detach().cpu().numpy()
            if mat.ndim == 2:                           
                save_feature_grid(mat, "Encoder Weights", e)
            metrics["mean"].append(mat.mean())
            metrics["std"].append(mat.std())   
            metrics["abs_mean"].append(np.abs(mat).mean())
            metrics["weight_eff_r"].append(np_effective_rank(mat))

 
            metrics["accuracy"].append(train_readout(net, seed))

        if e == epochs:
            break

        train(train_loader, net, criterion, optimizer)
 

    x = metrics["epoch"]
 
    save_line_plot(x, [(None, metrics["accuracy"], None)],
                   "Read-out Test Accuracy across Epochs",
                   "Test Set Accuracy", "Test_accuracy.png")
    save_line_plot(x, [(None, metrics["act_eff_r"], None)],
                   "Encoder Activation Effective Rank across Epochs",
                   "act_eff_r", "act_eff_r.png")
    save_line_plot(x, [(None, metrics["abs_mean"], None)],
                   "Mean of Absolute Encoder Weight across Epochs",
                   "", "Absolute_Means.png")
    save_line_plot(x, [(None, metrics["mean"], metrics["std"])],
                   "Mean (±std) of Signed Encoder Weight across Epochs",
                   "", "Signed_Means.png")
    save_line_plot(x, [(None, metrics["weight_eff_r"], None)],
                   "Effective Rank of Encoder Weight across Epochs",
                   "R_eff", "weight_R_eff.png")