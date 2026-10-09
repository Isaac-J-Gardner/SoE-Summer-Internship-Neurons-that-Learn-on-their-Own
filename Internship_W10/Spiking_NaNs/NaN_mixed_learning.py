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

N = 20 #number of hidden neurons (neuron autoencoders)

num_steps   = 50     #5 time constants, each image is shown for 50 time steps
theta      = 2.0     #starting threshold for each hidden neuron (readout threshold are set really high, potential is used to infer recon)
beta = 0.9           #membrane decay rate, produces time constant of ~10 time steps

alpha = 1 #mutual inhibition learning rate
learning_beta = 0.01 #task loss optimiser learning rate
gamma = 0.1 #adaptive threshold learning rate
p = 0.05

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

class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=N, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        #h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   #(batch, n_neurons, in_dim)

sur_grad = surrogate.atan(alpha = 2.0)
class SpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=N, beta=beta):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, spike_grad=sur_grad, threshold=theta)

        W = torch.zeros(n_hidden, n_hidden) #recurrent inhibitory weight matrix
        self.register_buffer("W_inh", W)

        self.register_buffer("theta", torch.zeros(n_hidden)) #per neuron threshold matrix, this threshold is added onto theta.

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
    def update_inhibition(self, activity, alpha): #SAILnet Mutual inhibition rule
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
        self.encoder = SpikingEncoder(784, N)
        self.decoder = nn.Sequential(
            NeuronDecoder(N, 784),
            snn.Leaky(beta=0.9, spike_grad=sur_grad, init_hidden=True,
                      output=True, threshold=200000)   #The decoder neurons never spike, so act as integrator
        )

    def forward(self, x):
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        x = x.view(x.size(0), -1)                    #(B,784) from (B, 1, 28, 28)

        # encode
        spk_rec_list, _ = self.encode(x) #array of spiking activity for each timestep, (T, B, H), useful for later
        spk_rec = torch.stack(spk_rec_list, dim=2)        # (B, H, T) #stacked into torch tensor

        # NEW: per-(sample, neuron) activity = total spikes over time -> the gate
        activity = spk_rec.sum(dim=2)                # (B, H) #total spikes per neuron per batch

        # decode: integrate each neuron's reconstruction over time
        spk_mem2 = []
        for step in range(num_steps):
            _, x_mem_recon = self.decode(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]   # membrane at last step (B,H,784)
        return spk_rec_list, x, out, activity #spk_rec_list is returned, again (T, B, H)

    def encode(self, x):
        return self.encoder(x, num_steps=num_steps)

    def decode(self, x):
        return self.decoder(x)

def recon_loss(x_recon, x, activity):
    target = x.unsqueeze(1).expand_as(x_recon)            #(B,H,784)
    se = ((target - x_recon) ** 2).mean(dim=2)            #(B,H) per neuron-sample
    if GATING:
        gate = (activity.detach() > 0).float()            #(B,H) 1 if neuron fired
        return (se * gate).sum() / gate.sum().clamp(min=1) #averaging gated squared error
    return se.mean()

@torch.no_grad()
def record_neuron_stats(network, dataset, batch_size=batch_size, top_k=1, device=device):
    """
    For every image in `dataset`, record per encoder neuron:
      deltas[j,i]   = sum_k R_ik e_ik              (δ_i, your backprojected recon error)
      firings[j,i]  = total spikes neuron i emitted for image j   (n_i, "how much it fired")
      currents[j,i] = sum_k Q_ik X_k + B_i = fc(x) (C_i, feed-forward current)

    Rows are aligned to dataset order (shuffle=False), so row j <-> dataset[j].
    Also returns, per neuron, the dataset image(s) giving the largest current
    (and the largest spike count, in case you want that instead).
    """
    network.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    R = network.decoder[0].weight                     # (H, 784)  == R_ik
    H = R.shape[0]
    n_images = len(dataset)

    deltas   = torch.empty(n_images, H)
    firings  = torch.empty(n_images, H)
    currents = torch.empty(n_images, H)

    pos = 0
    for img, _ in loader:
        b = img.size(0)
        img = img.to(device)
        x_norm = _normalize(img)

        spk_rec_list, x, x_recon, activity = network(x_norm)
        # x        : (b, 784)    normalised flattened input  == X_k
        # x_recon  : (b, H, 784) per-neuron reconstruction   == X_recon_ik
        # activity : (b, H)      total spikes per neuron      == n_i

        e       = x.unsqueeze(1) - x_recon            # (b, H, 784) == e_ik
        delta   = (e * R.unsqueeze(0)).sum(dim=2)     # (b, H)      == δ_i
        current = network.encoder.fc(x)               # (b, H)      == C_i

        deltas[pos:pos+b]   = delta.cpu()
        firings[pos:pos+b]  = activity.cpu()
        currents[pos:pos+b] = current.cpu()
        pos += b

    # per-neuron winners (indices into the dataset)
    top_by_current = currents.topk(top_k, dim=0).indices   # (top_k, H)
    top_by_firing  = firings.topk(top_k,  dim=0).indices   # (top_k, H)

    # pull the actual images for the current ranking (your explicit ask)
    top_imgs_current = torch.stack([
        torch.stack([dataset[int(top_by_current[k, i])][0] for i in range(H)])
        for k in range(top_k)
    ])   # (top_k, H, 1, 28, 28)

    top_imgs_current = _normalize(top_imgs_current)

    return {
        "deltas":           deltas,            # (n_images, H)
        "firings":          firings,           # (n_images, H)
        "currents":         currents,          # (n_images, H)
        "top_by_current":   top_by_current,    # (top_k, H) dataset indices
        "top_by_firing":    top_by_firing,     # (top_k, H) dataset indices
        "top_imgs_current": top_imgs_current,  # (top_k, H, 1, 28, 28)
    }
        

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
        spk_rec, x, x_recon, activity = network(_normalize(img))
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
    def __init__(self, encoder):
        super().__init__()
        # store the encoder WITHOUT registering it as a submodule,
        # so readout.parameters() contains ONLY fc2 and the encoder
        # is never handed to the read-out optimiser
        self._encoder = [encoder]
        self.fc2 = nn.Linear(N, 10)

    @property
    def encoder(self):
        return self._encoder[0]

    def forward(self, x):
        x = nn.Flatten()(x)
        # non-spiking approximation of the encoder feed-forward.
        # .detach() cuts the graph so NO gradient flows into the encoder;
        # only fc2 is updated by the read-out loss.
        h = torch.sigmoid(self.encoder.fc(x).detach())
        return self.fc2(h)

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()

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

READOUT_PROB = 0.5   # Bull: autoencoding vs task cycle chosen with equal probability

def train(network, readout, loader, opti, readout_opti, readout_criterion, epoch):
    network.train()
    readout.train()
    last_loss = None
    total_correct = 0
    num_items = len(loader)
    for i, (img, target) in enumerate(loader):
        img = img.to(device)
        target = target.to(device)
        x_norm = _normalize(img)

        # ---- task / read-out cycle ----
        readout_opti.zero_grad()
        output = readout(x_norm)                 # fc2-only grads, encoder detached
        task_loss = readout_criterion(output, target)
        task_loss.backward()
        readout_opti.step()
        total_correct += correct(output, target)

        opti.zero_grad()
        spk_rec_list, x, x_recon, activity = network(x_norm)
        rec_loss = recon_loss(x_recon, x, activity)
        rec_loss.backward()
        opti.step()
        network.encoder.update_inhibition(activity, alpha)
        network.encoder.update_threshold(activity, gamma)


        last_loss = rec_loss
        if i % 50 == 0:
            print(f'Train[{epoch+1}/{epochs}][{i}/{len(loader)}] '
                  f'Rec Loss: {rec_loss.item():.5f}, Task Loss: {task_loss.item():.5f}')
            
    print(f'Epoch Accuracy: {total_correct/num_items}')
    return last_loss

SAVE_DIR = "Internship_W10/Spiking_NaNs/images"
os.makedirs(SAVE_DIR, exist_ok=True)          # create it once, up front
 
TEST_EVERY = 4
SEEDS = range(1)
 

PARAM_SPECS = [
    ("encoder",      lambda net: net.encoder.fc.weight),
    ("decoder",      lambda net: net.decoder[0].weight),
    ("decoder_bias", lambda net: net.decoder[0].bias),
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
 
from collections import defaultdict

for seed in SEEDS:
    torch.manual_seed(seed)
    net = SAE().to(device)
    optimizer = torch.optim.SGD([
        {"params": net.encoder.fc.parameters(), "lr": learning_beta},
        {"params": net.decoder[0].weight,       "lr": learning_beta},
        {"params": net.decoder[0].bias,         "lr": learning_beta},
    ])

    # persistent read-out: built ONCE, never reset ------------------
    readout           = LinearReadout(net.encoder).to(device)
    readout_criterion = nn.CrossEntropyLoss()
    readout_optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)  # fc2 only
    # ---------------------------------------------------------------

    metrics = defaultdict(list)
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
                if mat.ndim == 2:
                    save_feature_grid(mat, name, e)
                param_hist[name]["mean"].append(mat.mean())
                param_hist[name]["std"].append(mat.std())
                param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                param_hist[name]["eff_rank"].append(alt_effective_rank(mat))

            save_matrix(to_numpy(net.encoder.W_inh), "Inhibition Weights", e, "inhib")

            # evaluate the SAME persistent read-out — no reset, no fresh training
            metrics["accuracy"].append(test_linear(test_loader, readout, readout_criterion))

        if e == epochs:
            break

        train(net, readout, train_loader, optimizer,
              readout_optimizer, readout_criterion, e)

    stats = record_neuron_stats(net, train_dataset, batch_size, 1)
    save_feature_grid(torch.squeeze(stats["top_imgs_current"]), "Highest_Current_Image", 0)
    indexes = stats["top_by_current"]
    deltas = np.zeros(N)
    firings = np.zeros(N)
    for i in range(N):
        deltas[i] = stats["deltas"][indexes[0,i], i]
        firings[i] = stats["firings"][indexes[0,i], i]

    save_line_plot(range(N), [(None, deltas, None)], "Deltas for Highest Current Images", "Delta", "deltas.png", xlabel="Neuron ID")
    save_line_plot(range(N), [(None, firings, None)], "Activity for Highest Current Image", "Number of Firiings", "firings.png", xlabel="Neuron ID")
        
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