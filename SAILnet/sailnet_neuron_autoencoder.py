"""
SAILnet backbone + per-neuron autoencoder learning rule  (snnTorch)
===================================================================
This takes the spiking SAILnet dynamics from the previous port and swaps out its
learning rule for the "Neuron-as-Network" (NaN) reconstruction rule from your
work with Larry Bull: every hidden neuron owns a *private decoder* and is itself
a (very lossy) autoencoder that reconstructs the whole input from its single
scalar activation n_i.

WHAT CHANGED vs the SAILnet port
--------------------------------
1. Separate decoder weights.   SAILnet reused the encoder weights Q as the
   generative/decoder weights. Here each neuron i has its own decoder weights
   R_ik (and the encoder Q_ik is learned separately).
2. Decoder bias.               New per-neuron, per-pixel decoder bias B_ik.
   (An encoder bias B_i is added too, since the rule updates it.)
3. Subtraction of a decoded signal.  Each neuron forms its own reconstruction
   Xhat_k^(i) = n_i R_ik + B_ik and its own error e_k^(i) = X_k - Xhat_k^(i).
   That per-neuron error (not SAILnet's shared X - sum_i n_i Q_i) drives learning.
4. Derivative in the encoder update.  Q is no longer updated by Oja's rule but by
   a backprop-through-the-decoder term that contains the spike nonlinearity's
   surrogate derivative S'(C_i)  (arctan surrogate).

WHAT WAS KEPT (unchanged from SAILnet, as requested)
----------------------------------------------------
- Training data: whitened natural-image patches.
- Inhibition:  lateral weights W with Foldiak's anti-Hebbian rule, W>=0, zero diag.
- Thresholding: per-neuron theta with the homeostatic rule dtheta = gamma(n_i - p).
- LIF inference dynamics (activities.m), reproduced by snn.Leaky.

THE ONE REAL MODELLING CHOICE:  what is S'(C_i) for a *spiking* neuron?
----------------------------------------------------------------------
The image writes n_i = S(C_i) with C_i = sum_k Q_ik X_k + B_i, a single feed-forward
pre-activation. In SAILnet n_i is instead a spike *count* produced by 50 steps of
recurrent LIF dynamics, so there is no unique C_i. Two defensible definitions are
provided via `surr_mode`:

  "feedforward" (default): C_i = Q_i . X + B_i (the feed-forward drive, exactly the
      image's C_i), and S'(C_i) = atan_surrogate(C_i - theta_i). One evaluation per
      image. This matches the written rule literally; the recurrent inhibition is
      treated as part of the (black-box) readout that produces n_i.

  "accumulated": S'_i = sum_t atan_surrogate(mem_i(t) - theta_i), accumulated over
      the 50 inference steps. This approximates dn_i/d(drive) through the spikes
      (the quantity a BPTT/surrogate-gradient trainer would use), ignoring the
      recurrent terms. Closer to your existing BPTT code, but departs from the
      single-S'(C_i) form in the figure.

Everything else is identical between the two modes.
"""

import math
import torch
import snntorch as snn


# --------------------------------------------------------------------------- #
#  arctan surrogate gradient  (matches snntorch.surrogate.atan(alpha))
# --------------------------------------------------------------------------- #
def atan_surrogate(u, alpha=2.0):
    """S'(u) for the Heaviside spike function; alpha=2 -> 1/(1+(pi u)^2)."""
    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * u) ** 2)


# --------------------------------------------------------------------------- #
#  Data (unchanged from the SAILnet port)
# --------------------------------------------------------------------------- #
def load_images(path="IMAGES.mat", key="IMAGES", device="cpu"):
    from scipy.io import loadmat
    IMAGES = loadmat(path)[key].astype("float32")
    return torch.from_numpy(IMAGES).to(device)


def sample_patches(IMAGES, batch_size, sz=16, BUFF=20):
    H, Wd, num_images = IMAGES.shape
    N = sz * sz
    device = IMAGES.device
    X = torch.empty(batch_size, N, device=device)
    for i in range(batch_size):
        r = torch.randint(BUFF, H - sz - BUFF, (1,)).item()
        c = torch.randint(BUFF, Wd - sz - BUFF, (1,)).item()
        img = torch.randint(0, num_images, (1,)).item()
        patch = IMAGES[r:r + sz, c:c + sz, img]
        v = patch.t().reshape(-1)                      # column-major, matches MATLAB
        v = v - v.mean()
        v = v / (v.std() + 1e-8)
        X[i] = v
    return X


# --------------------------------------------------------------------------- #
#  SAILnet + neuron-autoencoder learning
# --------------------------------------------------------------------------- #
class NeuronAutoencoderNet:
    """
    Parameters
    ----------
    N, M      : input pixels, number of neurons (M = OC*N).
    p         : target spikes/neuron/image (SAILnet sparseness / homeostasis target).
    eta, n_steps, theta_init : LIF inference settings (as in SAILnet).
    alpha_W, gamma           : SAILnet inhibition / threshold learning rates.
    lr_Q, lr_Benc, lr_R, lr_Bdec : the four reconstruction-rule learning rates
                                   (eta_Q, eta_Q, eta_R, eta_B in the figure).
    surr_alpha : arctan surrogate sharpness.
    surr_mode  : "feedforward" (default) or "accumulated" (see module docstring).
    """
    def __init__(self, N=256, M=256, p=0.05,
                 eta=0.1, n_steps=50, theta_init=2.0,
                 alpha_W=1.0, gamma=0.1,
                 lr_Q=0.01, lr_Benc=0.01, lr_R=0.01, lr_Bdec=0.01,
                 surr_alpha=2.0, surr_mode="accumulated", device="cpu"):
        self.N, self.M, self.p = N, M, p
        self.eta, self.n_steps, self.device = eta, n_steps, device
        self.alpha_W, self.gamma = alpha_W, gamma
        self.lr_Q, self.lr_Benc, self.lr_R, self.lr_Bdec = lr_Q, lr_Benc, lr_R, lr_Bdec
        self.surr_alpha, self.surr_mode = surr_alpha, surr_mode

        # encoder
        Q = torch.randn(M, N, device=device)
        self.Q = Q / Q.norm(dim=1, keepdim=True)       # feed-forward weights Q_ik
        self.B_enc = torch.zeros(M, device=device)     # encoder bias  B_i

        # decoder (each neuron owns a full N-pixel decoder)
        self.R = 0.1 * torch.randn(M, N, device=device)  # decoder weights R_ik
        self.B_dec = torch.zeros(M, N, device=device)    # decoder bias   B_ik

        # SAILnet inhibition + thresholds (unchanged)
        self.W = torch.zeros(M, M, device=device)
        self.theta = theta_init * torch.ones(M, device=device)

        # LIF cell matching activities.m exactly
        self.lif = snn.Leaky(beta=1 - eta, threshold=self.theta,
                             reset_mechanism="zero", reset_delay=False)

    # ---- inference: SAILnet LIF dynamics, drive now includes encoder bias --- #
    @torch.no_grad()
    def infer(self, X):
        """X: (batch, N) -> (Y spike counts, C feed-forward drive, accumulated S')."""
        b = X.shape[0]
        C = X @ self.Q.t() + self.B_enc                 # C_i = Q_i.X + B_i   (batch, M)
        mem = torch.zeros(b, self.M, device=self.device)
        spk = torch.zeros(b, self.M, device=self.device)
        Y = torch.zeros(b, self.M, device=self.device)
        surr_acc = torch.zeros(b, self.M, device=self.device)

        self.lif.threshold = self.theta
        for _ in range(self.n_steps):
            inp = self.eta * (C - spk @ self.W.t())     # drive - lateral inhibition
            spk, mem = self.lif(inp, mem)               # leak + threshold + reset->0
            Y = Y + spk
            surr_acc = surr_acc + atan_surrogate(mem - self.theta, self.surr_alpha)
        return Y, C, surr_acc

    # ---- learning: the four NaN updates + SAILnet keepers ------------------- #
    @torch.no_grad()
    def learn(self, X, Y, C, surr_acc):
        b = X.shape[0]
        nbar = Y.mean(0)                                # <n_i>            (M,)
        nsq = (Y * Y).mean(0)                           # <n_i^2>          (M,)
        Xbar = X.mean(0)                                # <X_k>            (N,)
        nX = (Y.t() @ X) / b                            # <n_i X_k>        (M, N)

        # delta_i = sum_k e_k^(i) R_ik, expanded so no (batch,M,N) tensor is needed:
        #   e_k^(i) = X_k - n_i R_ik - B_ik
        rsq = (self.R * self.R).sum(1)                  # sum_k R_ik^2     (M,)
        br = (self.B_dec * self.R).sum(1)               # sum_k B_ik R_ik  (M,)
        delta = X @ self.R.t() - Y * rsq[None, :] - br[None, :]   # (batch, M)

        # surrogate derivative of the spike nonlinearity
        if self.surr_mode == "feedforward":
            surr = atan_surrogate(C - self.theta, self.surr_alpha)  # S'(C_i)
        elif self.surr_mode == "accumulated":
            surr = surr_acc                                          # sum_t S'(mem-theta)
        else:
            raise ValueError("surr_mode must be 'feedforward' or 'accumulated'")

        g = surr * delta                                # S'(C_i) * delta_i  (batch, M)

        # ---- encoder (backprop through the decoder) ----
        self.Q     += self.lr_Q    * ((g.t() @ X) / b)              # dQ_ik = eta S'(C_i) delta_i X_k
        self.B_enc += self.lr_Benc * g.mean(0)                      # dB_i  = eta S'(C_i) delta_i

        # ---- decoder (delta rule on the reconstruction) ----
        # dR_ik   = eta n_i e_k   -> <n_i X_k> - <n_i^2> R_ik - <n_i> B_ik   (per-batch mean)
        self.R     += self.lr_R    * (nX - nsq[:, None] * self.R - nbar[:, None] * self.B_dec)
        # dB_ik   = eta e_k       -> <X_k> - <n_i> R_ik - B_ik
        self.B_dec += self.lr_Bdec * (Xbar[None, :] - nbar[:, None] * self.R - self.B_dec)

        # ---- SAILnet keepers: inhibition + homeostatic thresholds ----
        Cyy = (Y.t() @ Y) / b
        self.W += self.alpha_W * (Cyy - self.p ** 2)
        self.W.fill_diagonal_(0.0)
        self.W.clamp_(min=0.0)
        self.theta += self.gamma * (nbar - self.p)

    # ---- monitoring ---------------------------------------------------------- #
    @torch.no_grad()
    def reconstruction_mse(self, X, Y):
        """Mean per-neuron reconstruction MSE (materializes (batch,M,N); use small M)."""
        Xhat = Y[:, :, None] * self.R[None] + self.B_dec[None]
        return ((X[:, None, :] - Xhat) ** 2).mean().item()


# --------------------------------------------------------------------------- #
#  Visualization (generic: works on Q, R, or B_dec)
# --------------------------------------------------------------------------- #
def show_weights(Wmat, title="", save=None):
    import numpy as np
    import matplotlib.pyplot as plt
    Wmat = Wmat.detach().cpu().numpy()
    M, N = Wmat.shape
    sz = int(round(math.sqrt(N)))
    buf = 1
    if int(math.sqrt(M)) ** 2 != M:
        n = int(math.sqrt(M / 2)); m = M // n
    else:
        m = n = int(math.sqrt(M))
    arr = 0.5 * np.ones((buf + n * (sz + buf), buf + m * (sz + buf)))
    k = 0
    for j in range(m):
        for i in range(n):
            clim = np.max(np.abs(Wmat[k])) + 1e-12
            arr[buf + i*(sz+buf):buf + i*(sz+buf)+sz,
                buf + j*(sz+buf):buf + j*(sz+buf)+sz] = Wmat[k].reshape(sz, sz, order="F")/clim
            k += 1
    plt.figure(figsize=(7, 7)); plt.imshow(arr, cmap="gray")
    plt.axis("image"); plt.axis("off"); plt.title(title)
    if save: plt.savefig(save, dpi=120, bbox_inches="tight"); plt.close()
    else: plt.show()


# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #
def train(images_path="IMAGES.mat", num_trials=25000, batch_size=100,
          OC=1, p=0.05, surr_mode="feedforward", device=None, log_every=500):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    IMAGES = load_images(images_path, device=device)
    N = 256
    net = NeuronAutoencoderNet(N=N, M=OC * N, p=p, surr_mode=surr_mode, device=device)

    for t in range(1, num_trials + 1):
        X = sample_patches(IMAGES, batch_size)
        Y, C, surr_acc = net.infer(X)
        net.learn(X, Y, C, surr_acc)
        if t % log_every == 0:
            msg = (f"trial {t:6d} | rate {Y.mean():.3f} (target {p}) | "
                   f"theta {net.theta.mean():.2f} | W>0 {float((net.W>0).float().mean()):.3f}")
            if OC == 1:  # cheap enough to also print recon error
                msg += f" | reconMSE {net.reconstruction_mse(X, Y):.3f}"
            print(msg)
    return net


if __name__ == "__main__":
    net = train(num_trials=25000, OC=1, surr_mode="feedforward")
    show_weights(net.Q,     "Encoder weights Q",  save="nan_encoder.png")
    show_weights(net.R,     "Decoder weights R",  save="nan_decoder.png")
    show_weights(net.B_dec, "Decoder bias B_dec", save="nan_decoder_bias.png")
    print("Saved nan_encoder.png, nan_decoder.png, nan_decoder_bias.png")
