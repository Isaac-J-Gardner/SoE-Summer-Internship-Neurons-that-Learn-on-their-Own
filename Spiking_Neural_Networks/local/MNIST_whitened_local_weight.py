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




batch_size  = 100

epochs      = 50
N = 20

membrane_decay = 0.9
num_steps   = 50          # timesteps per image
theta      = 2.0        # base spiking threshold

alpha = 1
lr_beta = 0.01
gamma = 0.1
p = 1/N

         

spike_grad = surrogate.atan(alpha=2.0)




class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=N, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        #self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        # h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight #+ self.bias   # (batch, n_neurons, in_dim)


class WTASpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=N, beta=0.9):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=theta)

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
    def update_threshold(self, spk_rec, gamma):
        S_flat = torch.cat(spk_rec, dim=0)
        rate = S_flat.mean(0)
        dtheta = gamma*(rate-p)
        self.theta.add_(dtheta)



class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WTASpikingEncoder(784, N, beta=0.9)

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
            Q -= lr_beta * grad

        # optional: return the reconstruction loss for monitoring
        return err.pow(2).mean().item()

@torch.no_grad()
def test_encoder(network, loader):
    network.eval()
    losses = []
    spikes = []
    for img, _ in loader:
        img = img.to(device)
        img = _normalize(img)          # match training / readout
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
    def __init__(self, n_neurons=20, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_classes)

    def forward(self, feat):            # feat: (B, n_neurons)
        return self.fc(feat)

def _normalize(img):
    """Same per-image normalisation used during AE training."""
    img_flat = img.view(img.size(0), -1)
    mean = img_flat.mean(1).view(-1, 1, 1, 1)
    std  = img_flat.std(1).view(-1, 1, 1, 1)
    return (img - mean) / std

@torch.no_grad()
def extract_features(network, img):
    """Frozen forward pass -> per-neuron spike RATE feature (B, n_hidden)."""
    _, _, activity = network(_normalize(img))   # activity = spikes summed over T
    return activity / num_steps                     # rate in ~[0,1] per neuron

def train_readout(network, readout, loader, opti):
    network.eval(); readout.train()
    run_loss, correct, total = 0.0, 0, 0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        feat   = extract_features(network, img)     # detached from SAE graph
        logits = readout(feat)
        loss   = F.cross_entropy(logits, label)
        opti.zero_grad(); loss.backward(); opti.step()
        run_loss += loss.item() * img.size(0)
        correct  += (logits.argmax(1) == label).sum().item()
        total    += img.size(0)
    return run_loss / total, correct / total

@torch.no_grad()
def eval_readout(network, readout, loader):
    network.eval(); readout.eval()
    correct, total = 0, 0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        logits = readout(extract_features(network, img))
        correct += (logits.argmax(1) == label).sum().item()
        total   += img.size(0)
    return correct / total

def spk_effective_rank(spk_rec, n_neurons=20, eig_floor=1e-12, jitter=1e-6):
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
    return r_eff                           # spectral redundancy

def train(network, loader, epoch):
    network.train()
    for i, (img, _) in enumerate(loader):
        img = img.to(device)
        spk_rec, x, activity = network(_normalize(img))
        loss = network.learn(activity, x)
        network.encoder.update_inhibition(activity, alpha)
        network.encoder.update_threshold(spk_rec, gamma)
        if i % 50 == 0:
            print(f'Train[{epoch+1}/{epochs}][{i}/{len(loader)}] Loss: {loss:.5f} ')


# ---- setup -----------------------------------------------------------------
print('PyTorch:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tfm = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('./data', train=True,  download=True, transform=tfm),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tfm),
                          batch_size=batch_size, shuffle=False)


torch.manual_seed(0)

net = SAE().to(device)

test_epochs = []
accuracy = []
r_effs = []
avg_rates =[]
avg_threshs = []
inhibition_props = []
for e in range(epochs+1):
    if (e)%5 == 0:
        test_epochs.append(e)
        r_eff, avg_rate, avg_thresh, inhibition_prop = test_encoder(net, test_loader)
        r_effs.append(r_eff)
        avg_rates.append(avg_rate)
        avg_threshs.append(avg_thresh)
        inhibition_props.append(inhibition_prop)

        W  = net.encoder.fc.weight.detach().cpu().numpy()
        for title, mat in [("encoder", W)]:
            fig, axes = plt.subplots(4, 5, figsize=(10, 8)); fig.suptitle(title)
            for i, ax in enumerate(axes.flat):
                f = mat[i].reshape(28, 28); m = np.abs(f).max() + 1e-9
                ax.imshow(f, cmap='seismic', vmin=-m, vmax=m); ax.set_title(f'neuron {i}'); ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"Spiking_Neural_Networks/Images/local/local_weight/{title}_e{e}.png")
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
        plt.savefig(f"Spiking_Neural_Networks/Images/local/local_weight/inhib_e{e}.png")
        plt.close(fig)


        readout        = LinearReadout(n_neurons=20, n_classes=10).to(device)
        readout_opt    = torch.optim.SGD(readout.parameters(), lr=0.1)

        te_acc = 0
        tr_loss = 990
        previous_loss = 999
        while abs(previous_loss - tr_loss) > 7.5e-3:
            previous_loss = tr_loss
            tr_loss, tr_acc = train_readout(net, readout, train_loader, readout_opt)
            print(f'loss {tr_loss:.4f}, loss_diff = {previous_loss-tr_loss:.3e} '
                f'train_acc {tr_acc:.4f}')
        te_acc = eval_readout(net, readout, test_loader)
        accuracy.append(te_acc)

    train(net, train_loader, e)

plt.figure()
plt.plot(test_epochs, accuracy)
plt.title("Readout Accuracy across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Test Set Accuracy')
plt.savefig("Spiking_Neural_Networks/Images/local/local_weight/accuracy.png")
plt.close()

plt.figure()
plt.plot(test_epochs, r_effs)
plt.title("Encoder Activation Effective Rank across Epochs")
plt.xlabel('Epoch')
plt.ylabel('R_eff')
plt.savefig("Spiking_Neural_Networks/Images/local/local_weight/effective_rank.png")
plt.close()

plt.figure()
plt.plot(test_epochs, avg_rates)
plt.title("Average Encoder Firing Rate across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Firing Rate')
plt.savefig("Spiking_Neural_Networks/Images/local/local_weight/firing_rate.png")
plt.close()

plt.figure()
plt.plot(test_epochs, avg_threshs)
plt.title("Average Encoder Neuron Threshold (p=0.05) across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Average Threshold')
plt.savefig("Spiking_Neural_Networks/Images/local/local_weight/threshold.png")
plt.close()

plt.figure()
plt.plot(test_epochs, inhibition_props)
plt.title("Proportiong of W_ing > 0 across Epochs")
plt.xlabel('Epoch')
plt.ylabel('W_ing > 0')
plt.savefig("Spiking_Neural_Networks/Images/local/local_weight/W_inh.png")
plt.close()