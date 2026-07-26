"""
Neuron-Autoencoder (NaN) collapse, measured in ACTIVATION space.

For each of four conditions we train the neuron autoencoder on reconstruction only
(exactly the training rule in the original script) and, once per epoch, push a fixed
evaluation batch through the encoder, form the 20x20 covariance/correlation of the
hidden activations h = sigmoid(encoder(x)), and log:

  - r_eff   = exp(H_lambda)           effective rank of the activation covariance
  - R_spec  = 1 - r_eff / D           spectral (geometric) redundancy
  - R_KL    = -1/2 log det C          Gaussian total correlation (C = correlation matrix)
  - R_frob  = 1/4 ||C - I||_F^2       weak-dependence quadratic approximation of R_KL
  - cos     = mean off-diagonal cosine similarity of the ENCODER rows (weight-space
              diversity, kept alongside the activation-space measures for comparison)

Definitions follow Zhang et al., "Redundancy ..." (2510.10938v1), Sec 3.2-3.3.
D = 20 hidden neurons.

Four conditions:
  NK-NaN               : Kauffman NK inputs (random independent bits, N=784)
  MNIST-NaN raw        : MNIST pixels in [0,1]
  MNIST-NaN mean-sub   : MNIST with the per-pixel mean removed
  MNIST-NaN whitened   : MNIST ZCA-whitened (input covariance ~ I, i.e. NK-like)
"""

import gzip, struct, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 20            # hidden neurons
IN_DIM = 784
BATCH = 64
EVAL_N = 2000     # fixed batch used for the activation-space metrics
SEED = 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(*shape)


def load_mnist_flat(n=None, data_dir='./data'):
    """Prefer torchvision (matches the original script); fall back to raw IDX files."""
    try:
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        ds = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
        X = ds.data.float().reshape(len(ds), -1) / 255.0
    except Exception:
        imgs = _read_idx(f'{data_dir}/train-images-idx3-ubyte.gz').astype(np.float32) / 255.0
        X = torch.from_numpy(imgs.reshape(imgs.shape[0], -1).copy())
    if n is not None:
        X = X[:n]
    return X.contiguous()


def make_nk_inputs(n, N=IN_DIM, seed=SEED):
    """NK genomes: random independent bits. The autoencoder reconstructs the input,
    so only the input distribution matters for the collapse analysis; the NK fitness
    target is not needed for reconstruction training."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, N, generator=g) < 0.5).float()


def zca_whiten(X, eps_frac=1e-2):
    """ZCA whitening with an eigenvalue floor (MNIST has many ~0-variance pixels)."""
    mean = X.mean(0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    floor = evals.max() * eps_frac
    inv_sqrt = torch.clamp(evals, min=floor).rsqrt()
    W = (evecs * inv_sqrt) @ evecs.T           # ZCA matrix, symmetric
    return Xc @ W


def input_effective_rank(X):
    """Static diagnostic: effective rank of the INPUT covariance (explains *why*)."""
    Xc = X - X.mean(0, keepdim=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    ev = ev[ev > 0]
    p = ev / ev.sum()
    H = -(p * p.log()).sum()
    return torch.exp(H).item()


# --------------------------------------------------------------------------- #
# Model  (identical to the original NeuronAutoencoder)
# --------------------------------------------------------------------------- #
class NeuronAutoencoder(nn.Module):
    def __init__(self, n_neurons=D, in_dim=IN_DIM):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x
        h = torch.sigmoid(self.encoder(x))
        decoded = (h.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))
        return decoded, features


# --------------------------------------------------------------------------- #
# Activation-space redundancy metrics
# --------------------------------------------------------------------------- #
def redundancy_metrics(H, encoder_W, ridge=1e-6):
    """H: (B, D) hidden activations. encoder_W: (D, in_dim)."""
    B = H.shape[0]
    Hc = H - H.mean(0, keepdim=True)
    cov = (Hc.T @ Hc) / (B - 1)                      # (D, D)

    # --- spectral: effective rank & spectral redundancy (covariance eigenspectrum)
    evals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = evals / evals.sum()
    H_lambda = -(p * p.log()).sum()
    r_eff = torch.exp(H_lambda).item()
    R_spec = 1.0 - r_eff / D

    # --- correlation matrix for the KL / Frobenius forms
    std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
    C = cov / (std[:, None] * std[None, :])
    C = 0.5 * (C + C.T)
    I = torch.eye(D, device=C.device)
    Cr = C + ridge * I                               # keep log det finite under collapse
    sign, logdet = torch.linalg.slogdet(Cr)
    R_KL = (-0.5 * logdet).item()
    R_frob = (0.25 * ((C - I) ** 2).sum()).item()

    # --- encoder-row cosine similarity (weight-space diversity, signed off-diagonal)
    Wn = encoder_W / encoder_W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    S = Wn @ Wn.T
    off = S[~torch.eye(D, dtype=torch.bool, device=S.device)]
    cos = off.mean().item()

    return dict(r_eff=r_eff, R_spec=R_spec, R_KL=R_KL, R_frob=R_frob, cos=cos)


# --------------------------------------------------------------------------- #
# Train one condition, logging metrics per epoch
# --------------------------------------------------------------------------- #
def run_condition(name, X, epochs=20, lr=10.0, seed=SEED):
    torch.manual_seed(seed)                          # identical init across conditions
    model = NeuronAutoencoder().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True)
    Xeval = X[:EVAL_N].to(DEVICE)

    def log_epoch():
        model.eval()
        with torch.no_grad():
            Heval = torch.sigmoid(model.encoder(Xeval))
        return redundancy_metrics(Heval, model.encoder.weight.detach())

    history = [log_epoch()]                           # epoch 0 = initialisation
    for ep in range(epochs):
        model.train()
        running = 0.0
        for (data,) in loader:
            data = data.to(DEVICE)
            decoded, features = model(data)
            loss = crit(decoded, features.unsqueeze(1).expand_as(decoded))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        m = log_epoch()
        m['recon'] = running / len(loader)
        history.append(m)
    # give epoch-0 a recon entry too (pre-training loss placeholder = first epoch's)
    history[0]['recon'] = history[1]['recon']
    print(f"[{name}] final: r_eff={history[-1]['r_eff']:.2f}  "
          f"R_spec={history[-1]['R_spec']:.3f}  R_KL={history[-1]['R_KL']:.2f}  "
          f"cos={history[-1]['cos']:.3f}")
    return history, model


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=100.0)
    ap.add_argument('--n', type=int, default=12000)   # train subset size
    args = ap.parse_args()

    mnist = load_mnist_flat(args.n)
    conditions = {
        'NK-NaN':             make_nk_inputs(args.n),
        'MNIST-NaN raw':      mnist,
        'MNIST-NaN mean-sub': mnist - mnist.mean(0, keepdim=True),
        'MNIST-NaN whitened': zca_whiten(mnist),
    }

    print("Input-space effective rank (static diagnostic, D_in=784):")
    for k, v in conditions.items():
        print(f"  {k:20s} r_eff_in = {input_effective_rank(v):7.2f}")
    print()

    results = {}
    for name, X in conditions.items():
        hist, _ = run_condition(name, X, epochs=args.epochs, lr=args.lr)
        results[name] = hist

    with open('results.json', 'w') as f:
        json.dump(results, f)
    print("\nsaved results.json")