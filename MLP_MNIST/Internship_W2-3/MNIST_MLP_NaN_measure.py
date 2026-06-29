import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

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

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    total += images.sum(dim=0)   # sum over the batch
    n += images.size(0)
mean_image = nn.Flatten()(total / n)        # shape [1, 784]
mean_image = mean_image.to(device)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28*28, 20)
        self.decoder_weights = nn.Parameter(torch.randn(20, 784) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(20, 784))
        self.readout = nn.Linear(20, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = x - mean_image                # was `x -= mean_image` (in-place on a view of the
                                          # input batch); out-of-place is behaviour-equivalent
                                          # here and avoids mutating the loader's tensor.
        features = x                      # shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.sigmoid(x)
        decoded = None
        if self.training:
            decoded = x.unsqueeze(2) * self.decoder_weights.unsqueeze(0) + self.decoder_bias.unsqueeze(0)  # [batch, 20, 784]
        output = self.readout(x.detach())
        return output, decoded, features


model = SimpleMLP().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()


def train(data_loader, model, criterion, recon_criterion, optimizer):
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
        output, decoded, features = model(data)

        # Calculate the loss
        cycle = random.randint(0, 1)
        if cycle == 0:
            loss = criterion(output, target)
        else:
            loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        total_loss += loss.item()                 # .item() so we don't retain the graph each batch

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")


# ===========================================================================
#  Representation diagnostics
#  ---------------------------------------------------------------------------
#  Four probes of WHAT the hidden layer has learned, run at intervals so you
#  can watch the representation either form or collapse:
#    1. hidden weight norm        -> is the encoder learning, or stuck at init?
#    2. latent-ablation recon     -> does the latent carry info, or just bias?
#    3. effective rank / |corr|   -> 20 distinct features, or all the same one?
#    4. linear probe              -> fairest measure of representation quality
# ===========================================================================

@torch.no_grad()
def extract_hidden(model, loader, device):
    """Return hidden activations H [N, 20] (on CPU) and labels y [N]."""
    model.eval()
    Hs, ys = [], []
    for data, target in loader:
        data = data.to(device)
        flat = nn.Flatten()(data) - mean_image          # same preprocessing as forward()
        h = torch.sigmoid(model.encoder(flat))          # [b, 20]
        Hs.append(h.cpu())
        ys.append(target.clone())
    return torch.cat(Hs), torch.cat(ys)


@torch.no_grad()
def reconstruction_mse(model, loader, device, mean_latent):
    """Full vs latent-ablated reconstruction MSE.

    'Ablated' replaces each neuron's latent with its dataset-mean value, so the
    decoder emits its best CONSTANT (mean_latent * W + bias) for every input.
    If the full MSE is barely lower than the ablated MSE, the varying latent is
    contributing almost nothing and the decoder is reconstructing from its bias
    alone  ->  mean / posterior collapse (mode 1).
    """
    model.eval()
    W = model.decoder_weights            # [20, 784]
    b = model.decoder_bias               # [20, 784]
    abl_const = mean_latent.to(device).unsqueeze(1) * W + b   # [20, 784] best constant per neuron
    full_sse, abl_sse, count = 0.0, 0.0, 0
    for data, _ in loader:
        data = data.to(device)
        flat = nn.Flatten()(data) - mean_image                 # [b, 784] = features / targets
        h = torch.sigmoid(model.encoder(flat))                 # [b, 20]
        full = h.unsqueeze(2) * W.unsqueeze(0) + b.unsqueeze(0)  # [b, 20, 784]
        tgt = flat.unsqueeze(1)                                 # [b, 1, 784]
        full_sse += ((full - tgt) ** 2).sum().item()
        abl_sse += ((abl_const.unsqueeze(0) - tgt) ** 2).sum().item()
        count += flat.size(0) * W.size(0) * W.size(1)
    full_mse = full_sse / count
    abl_mse = abl_sse / count
    contribution = (abl_mse - full_mse) / abl_mse if abl_mse > 0 else 0.0
    return full_mse, abl_mse, contribution


def effective_rank(H):
    """Entropy-based effective rank (Roy & Vetterli) and participation ratio of
    the centered hidden activations H [N, D].  Both range ~[1, D]:
    ~D  -> 20 independent features (healthy);  ~1 -> all neurons collapsed onto
    one direction (redundancy collapse, mode 2)."""
    Hc = H - H.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Hc)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 1.0, 1.0
    p = s / s.sum()
    erank = torch.exp(-(p * p.log()).sum()).item()
    lam = s ** 2
    pr = (lam.sum() ** 2 / (lam ** 2).sum()).item()
    return erank, pr


def mean_abs_corr(H):
    """Mean absolute off-diagonal correlation between the D hidden units.
    ~0 -> decorrelated / diverse;  ~1 -> all units encode the same thing."""
    C = torch.corrcoef(H.t())                    # [D, D]
    D = C.size(0)
    off = C[~torch.eye(D, dtype=torch.bool)]
    off = off[~torch.isnan(off)]                 # guard against constant (dead) units
    return off.abs().mean().item() if off.numel() else float('nan')


def linear_probe(model, train_loader, test_loader, device, steps=300, lr=0.05):
    """Freeze the hidden layer, fit a fresh logistic-regression head on the
    frozen features, return test accuracy.  This decouples representation
    quality from the (detached) readout's own training, so it's the fairest
    single number for 'how separable are the classes in the hidden space'."""
    Htr, ytr = extract_hidden(model, train_loader, device)
    Hte, yte = extract_hidden(model, test_loader, device)
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6      # standardize for stable, fair probing
    Htr = ((Htr - mu) / sd).to(device); ytr = ytr.to(device)
    Hte = ((Hte - mu) / sd).to(device); yte = yte.to(device)

    probe = nn.Linear(Htr.size(1), 10).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(steps):                       # full-batch: 60k x 20 is tiny
        opt.zero_grad()
        lossf(probe(Htr), ytr).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        acc = (probe(Hte).argmax(1) == yte).float().mean().item()
    return acc


def run_diagnostics(model, train_loader, test_loader, device, with_probe=True):
    Hte, _ = extract_hidden(model, test_loader, device)
    mean_latent = Hte.mean(0)                                   # per-neuron mean latent over test set
    full_mse, abl_mse, contribution = reconstruction_mse(model, test_loader, device, mean_latent)
    erank, pr = effective_rank(Hte)
    corr = mean_abs_corr(Hte)
    wnorm = model.encoder.weight.detach().norm(dim=1)          # per-neuron L2 of incoming weights
    probe_acc = linear_probe(model, train_loader, test_loader, device) if with_probe else float('nan')
    return {
        'weight_norm_mean': wnorm.mean().item(),
        'weight_norm_max': wnorm.max().item(),
        'recon_full_mse': full_mse,
        'recon_ablated_mse': abl_mse,
        'latent_contribution': contribution,
        'effective_rank': erank,
        'participation_ratio': pr,
        'mean_abs_corr': corr,
        'probe_acc': probe_acc,
    }


def print_diag(tag, d):
    print(f"  [{tag}] probe={d['probe_acc']:.2%}  erank={d['effective_rank']:.2f}/20  "
          f"|corr|={d['mean_abs_corr']:.2f}  latent_contrib={d['latent_contribution']:.2%}  "
          f"|W|mean={d['weight_norm_mean']:.3f}  full/abl MSE={d['recon_full_mse']:.4f}/{d['recon_ablated_mse']:.4f}")


# ===========================================================================
#  Training loop with diagnostics
# ===========================================================================
DIAG_EVERY = 1            # run diagnostics every N epochs (raise to 2-5 to save time)
history = []

print("Baseline (random init, before training):")
d0 = run_diagnostics(model, train_loader, test_loader, device)
d0['epoch'] = 0
history.append(d0)
print_diag("epoch 0", d0)

epochs = 20
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, recon_criterion, optimizer)
    if (epoch + 1) % DIAG_EVERY == 0 or (epoch + 1) == epochs:
        d = run_diagnostics(model, train_loader, test_loader, device)
        d['epoch'] = epoch + 1
        history.append(d)
        print_diag(f"epoch {epoch+1}", d)


def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output, _, _ = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")


test(test_loader, model, criterion)


# ===========================================================================
#  Plots
# ===========================================================================
ep = [h['epoch'] for h in history]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# (a) Linear probe accuracy — the headline representation-quality metric
ax = axes[0, 0]
ax.plot(ep, [h['probe_acc'] for h in history], 'o-', color='tab:blue')
ax.set_title('Linear probe accuracy (frozen hidden -> class)')
ax.set_xlabel('epoch'); ax.set_ylabel('test accuracy'); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

# (b) Effective rank (left) and mean |corr| (right) — diversity vs redundancy
ax = axes[0, 1]
ax.plot(ep, [h['effective_rank'] for h in history], 'o-', color='tab:green', label='effective rank')
ax.axhline(20, ls='--', color='gray', lw=1, label='max (20)')
ax.set_ylabel('effective rank', color='tab:green'); ax.set_ylim(0, 21)
ax.set_xlabel('epoch'); ax.set_title('Diversity of hidden units')
ax2 = ax.twinx()
ax2.plot(ep, [h['mean_abs_corr'] for h in history], 's-', color='tab:red', label='mean |corr|')
ax2.set_ylabel('mean |corr|', color='tab:red'); ax2.set_ylim(0, 1)
ax.grid(alpha=0.3)

# (c) Reconstruction: full vs latent-ablated MSE
ax = axes[1, 0]
ax.plot(ep, [h['recon_full_mse'] for h in history], 'o-', label='full recon')
ax.plot(ep, [h['recon_ablated_mse'] for h in history], 's--', label='ablated (bias-only)')
ax.set_title('Reconstruction MSE: full vs latent-ablated')
ax.set_xlabel('epoch'); ax.set_ylabel('MSE'); ax.legend(); ax.grid(alpha=0.3)

# (d) Latent contribution (left) and encoder weight norm (right)
ax = axes[1, 1]
ax.plot(ep, [h['latent_contribution'] for h in history], 'o-', color='tab:purple', label='latent contribution')
ax.set_ylabel('latent contribution', color='tab:purple'); ax.set_ylim(0, 1)
ax.set_xlabel('epoch'); ax.set_title('Latent usefulness & encoder weight norm')
ax3 = ax.twinx()
ax3.plot(ep, [h['weight_norm_mean'] for h in history], 's-', color='tab:orange', label='mean |W| per neuron')
ax3.set_ylabel('mean encoder weight norm', color='tab:orange')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics_over_training.png', dpi=120)
plt.show()


# Encoder weight visualisation (unchanged)
W = model.encoder.weight.detach().cpu().numpy()   # (20, 784)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()