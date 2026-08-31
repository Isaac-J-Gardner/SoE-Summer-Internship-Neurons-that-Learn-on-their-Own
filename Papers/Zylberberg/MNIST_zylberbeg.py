"""
SAILnet in PyTorch / snnTorch
=============================
A faithful port of Zylberberg, Murphy & DeWeese (2011),
"A sparse coding model with synaptically local plasticity and spiking neurons..."
PLoS Comput Biol 7(10): e1002250.

"""

import torch
import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt



# --------------------------------------------------------------------------- #
#  Data: whitened natural-image patches (matches init.m / SAILnet.m)
# --------------------------------------------------------------------------- #
def load_images(path="IMAGES.mat", key="IMAGES", device="cpu"):
    """Load the Olshausen whitened image array (imsize x imsize x num_images)."""
    from scipy.io import loadmat
    IMAGES = loadmat(path)[key].astype("float32")
    return torch.from_numpy(IMAGES).to(device)


def sample_patches(IMAGES, batch_size, sz=16, BUFF=20):
    """Extract `batch_size` random sz x sz patches, each zero-mean / unit-std.

    Mirrors the inner loop of SAILnet.m. Uses column-major (Fortran) flattening
    so patch vectors line up with the RF display in show_rfs().
    """
    num_images, _, H, Wd = IMAGES.shape
    N = sz * sz
    device = IMAGES.device
    X = torch.empty(batch_size, N, device=device)
    for i in range(batch_size):
        r = torch.randint(BUFF, H - sz - BUFF, (1,)).item()
        c = torch.randint(BUFF, Wd - sz - BUFF, (1,)).item()
        img = torch.randint(0, num_images, (1,)).item()
        patch = IMAGES[r:r + sz, c:c + sz, img]
        # column-major flatten to match MATLAB reshape(...,N,1)
        v = patch.t().reshape(-1)
        v = v - v.mean()
        v = v / (v.std() + 1e-8)
        X[i] = v
    return X


# --------------------------------------------------------------------------- #
#  SAILnet
# --------------------------------------------------------------------------- #
class SAILnet:
    """
    N : number of input pixels (256 for 16x16 patches)
    M : number of output neurons (OC * N; paper uses OC=6 -> 1536)
    p : target spikes/neuron/image (lifetime sparseness), 0.05 in the paper

    theta_init: init.m uses 2.0; the paper's Methods text says 5.0. Both work;
                it only affects transient dynamics, not the learned solution.
    """
    def __init__(self, N=28*28, M=256, p=0.05,
                 alpha=1.0, beta=0.01, gamma=0.1,
                 n_steps=50, eta=0.1, theta_init=2.0, device="cpu"):
        self.N, self.M, self.p = N, M, p
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.n_steps, self.eta, self.device = n_steps, eta, device

        # feed-forward weights Q (M x N), rows L2-normalized  (init.m)
        Q = torch.randn(M, N, device=device)
        self.Q = Q / Q.norm(dim=1, keepdim=True)

        # lateral inhibitory weights W (M x M), start at zero  (init.m)
        self.W = torch.zeros(M, M, device=device)

        # per-neuron thresholds theta
        self.theta = theta_init * torch.ones(M, device=device)

        # LIF cell configured to match activities.m exactly (see module docstring)
        self.lif = snn.Leaky(beta=1 - eta, threshold=self.theta,
                             reset_mechanism="zero", reset_delay=False)

    # ---- inference: activities.m ---------------------------------------- #
    @torch.no_grad()
    def infer(self, X):
        """X: (batch, N) -> Y: (batch, M) integer-valued spike counts."""
        b = X.shape[0]
        B = X @ self.Q.t()                              # projections Q*X  (batch, M)
        mem = torch.zeros(b, self.M, device=self.device)
        spk = torch.zeros(b, self.M, device=self.device)
        Y   = torch.zeros(b, self.M, device=self.device)

        self.lif.threshold = self.theta                 # thresholds may have changed
        for _ in range(self.n_steps):
            # lateral inhibition uses spikes from the PREVIOUS step (one-step delay)
            inp = self.eta * (B - spk @ self.W.t())     # eta*(B - W*as)
            spk, mem = self.lif(inp, mem)               # leak + threshold + reset->0
            Y = Y + spk                                 # accumulate spike counts
        return Y

    # ---- learning: SAILnet.m -------------------------------------------- #
    @torch.no_grad()
    def learn(self, X, Y):
        b = X.shape[0]

        # W: Foldiak's rule  dW = alpha*(<n_i n_m> - p^2), inhibitory only
        Cyy = (Y.t() @ Y) / b                           # activity correlation (M x M)
        self.W += self.alpha * (Cyy - self.p ** 2)
        self.W.fill_diagonal_(0.0)                       # no self-connection
        self.W.clamp_(min=0.0)                           # keep connections inhibitory

        # Q: Oja's rule  dQ = beta*n_i*(X_k - n_i*Q_ik)
        sq = (Y * Y).sum(0)                              # sum_batch n_i^2   (M,)
        self.Q += self.beta * (Y.t() @ X) / b \
                - self.beta * (sq[:, None] * self.Q) / b

        # theta: homeostasis  dtheta = gamma*(<n_i> - p)
        self.theta += self.gamma * (Y.mean(0) - self.p)

    # ---- receptive fields ----------------------------------------------- #
    @torch.no_grad()
    def receptive_fields(self):
        """RFs equal the feed-forward weights Q (proven in the paper's Methods)."""
        return self.Q.clone()


# --------------------------------------------------------------------------- #
#  Visualization: showrfs.m
# --------------------------------------------------------------------------- #
def show_rfs(Q, title="Q", save=None):
    """Tile the rows of Q (M x N) into an image grid, like showrfs.m."""
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    Q = Q.detach().cpu().numpy()
    M, N = Q.shape
    sz = int(round(math.sqrt(N)))
    buf = 1
    if int(math.sqrt(M)) ** 2 != M:
        n = int(math.sqrt(M / 2)); m = M // n
    else:
        m = n = int(math.sqrt(M))

    array = 0.5 * np.ones((buf + n * (sz + buf), buf + m * (sz + buf)))
    k = 0
    for j in range(m):
        for i in range(n):
            clim = np.max(np.abs(Q[k])) + 1e-12
            tile = Q[k].reshape(sz, sz, order="F") / clim   # column-major, matches MATLAB
            r0 = buf + i * (sz + buf)
            c0 = buf + j * (sz + buf)
            array[r0:r0 + sz, c0:c0 + sz] = tile
            k += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap="gray"); plt.axis("image"); plt.axis("off")
    plt.title(title)
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
#  Training loop: SAILnet.m
# --------------------------------------------------------------------------- #
def train(images_path="IMAGES.mat", num_trials=25000, batch_size=64,
          OC=1, p=0.05, device=None, lr_schedule=True, log_every=500):
    
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('Using GPU, device name:', torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        print('No GPU found, using CPU instead.') 
        device = torch.device('cpu')

    data_dir = './data'
    print('data_dir =', data_dir)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = 28*28
    net = SAILnet(N=N, M=20, p=p, alpha=5, beta=0.01, gamma=3, device=device)

    for i, (data, _) in enumerate(train_loader):
        # anneal learning rates for the final stretch, per the paper's Methods
        if i > 600:
            net.alpha, net.beta, net.gamma = 0.1, 0.001, 0.1

        data = data.to(device)
        data=data.view(-1, 28*28)
        X = torch.empty(data.size(0), N, device=device)
        for j in range(data.size(0)):
            v = data[j]
            v = v - v.mean()
            v = v / (v.std() + 1e-8)
            X[j] = v

        Y = net.infer(X)
        net.learn(X, Y)

        if i % 50 == 0:
            print(f"batch {i:6d} | mean rate {Y.mean():.3f} (target {p}) | "
                    f"W>0 {float((net.W > 0).float().mean()):.3f} | "
                    f"theta mean {net.theta.mean():.2f}")

    return net


if __name__ == "__main__":
    net = train(num_trials=25000, OC=1)                 # OC=6 to match the paper's 1536 units
    show_rfs(net.receptive_fields(), title="SAILnet RFs", save="sailnet_rfs.png")
    print("Saved receptive fields to sailnet_rfs.png")