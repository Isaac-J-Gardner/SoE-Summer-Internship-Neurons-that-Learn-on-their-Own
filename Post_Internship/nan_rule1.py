"""
nan_rule1.py
================
Spiking Neuron-Autoencoder trained with **paired learning Rule 1** (the first of the
four options): independent decoder weights and decoder bias (R_ik != Q_ik), encoder
weights Q_ik and encoder bias B_i, derived with the full chain rule.

This replaces the autograd / MSE-backprop update of the autoencoder weights with the
explicit per-synapse rule.  SAILnet's inhibition and threshold rules are unchanged.
The linear read-out probe is still trained by ordinary backprop (it is only a probe).

Per-sample, per-neuron (i) / per-pixel (k) gradients of L_k = e_k^2, with
    e_ik = x_k - xbar_ik,   xbar_ik = n_i R_ik + B_ik,   C_i = Q_ik x_k + B_i :

    dL/dB_ik = -2 e_ik                         (decoder bias,   per pixel)
    dL/dR_ik = -2 n_i e_ik                     (decoder weight, per pixel)
    dL/dB_i  = -2 S'(C_i) * sum_k R_ik e_ik    (encoder bias,   shared -> summed)
    dL/dQ_ik = -2 S'(C_i) R_ik e_ik x_k        (encoder weight, LOCAL per-pixel error)

Gradient descent => update = -lr * dL/dparam, averaged over the batch and gated by
whether neuron i fired (matching the gated reconstruction loss).

------------------------------------------------------------------------------------
KEY MODELLING CHOICE -- the S'(C_i) gain
------------------------------------------------------------------------------------
S'(C_i) is the one genuinely ambiguous ingredient: the encoder is a 50-step spiking
LIF unit, so "the derivative of the transfer function" is not literally defined. It is
isolated in ONE place -- ``gain_fn`` -- so you can swap it:

  * USE_CHAIN_RULE = True  : S'(C_i) ~ surrogate(ATan) derivative at the feed-forward
                             drive past threshold (u = fc(x) - theta_base - theta_i).
                             This is the differentiable-autoencoder reading of Rule 1.
  * USE_CHAIN_RULE = False : S'(C_i) == 1, i.e. treat n_i as fixed during the weight
                             step (the SAILnet / envelope-theorem reading). This keeps
                             the Q rule fully local.

Note also the deliberate *locality* choice in the Q rule: it uses only pixel k's error
e_ik, NOT the summed back-projection sum_k R_ik e_ik. The true network gradient would
use the sum (since Q_ik feeds C_i which drives every reconstructed pixel); using the
per-pixel error is exactly the SAILnet-style locality you are after. The encoder *bias*
B_i is a genuinely shared parameter, so its rule does sum over pixels.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils as snnutils
from snntorch import surrogate

import nan_utils as U


# =============================================================================== config
@dataclass
class Config:
    n_in: int = 784
    N: int = 20                 # hidden neurons (neuron-autoencoders)
    num_steps: int = 50         # time steps per image (~5 time constants)
    theta_base: float = 2.0     # base LIF threshold
    beta: float = 0.9           # membrane decay (time constant ~10 steps)

    alpha: float = 1.0          # SAILnet mutual-inhibition learning rate
    gamma: float = 0.1          # SAILnet adaptive-threshold learning rate
    p: float = field(init=False)   # target firing prob = 1/N

    # paired Rule-1 learning rates (vary lr_Bik for your decoder-bias experiment)
    lr_Q: float = 0.01
    lr_R: float = 0.001
    lr_Bi: float = 0.001
    lr_Bik: float = 0.001

    surrogate_alpha: float = 2.0
    use_chain_rule: bool = True    # S'(C) via surrogate deriv (True) or == 1 (False)
    gating: bool = True

    epochs: int = 20
    batch_size: int = 100
    test_every: int = 4
    seeds: tuple = (0,)

    data_dir: str = "./data"
    save_dir: str = "Post_Internship/rule_1"
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.p = 1.0 / self.N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================== models
class NeuronDecoder(nn.Module):
    """Each neuron i decodes pixel k independently: xbar_ik = h_i * R_ik + B_ik."""
    def __init__(self, n_neurons, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)   # R_ik
        self.bias = nn.Parameter(torch.zeros(n_neurons, in_dim))            # B_ik

    def forward(self, h):                         # h: (B, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias                    # (B, H, in_dim)


class SpikingEncoder(nn.Module):
    """LIF encoder with SAILnet recurrent inhibition and adaptive thresholds."""
    def __init__(self, cfg, spike_grad):
        super().__init__()
        self.cfg = cfg
        self.n_hidden = cfg.N
        self.num_steps = cfg.num_steps
        self.fc = nn.Linear(cfg.n_in, cfg.N)                      # Q_ik (+ B_i bias)
        self.lif = snn.Leaky(beta=cfg.beta, spike_grad=spike_grad,
                             threshold=cfg.theta_base)

        self.register_buffer("W_inh", torch.zeros(cfg.N, cfg.N))  # recurrent inhibition
        self.register_buffer("theta", torch.zeros(cfg.N))         # adaptive threshold

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
    def update_inhibition(self, activity, alpha):             # SAILnet dW ~ n_i n_m - p^2
        p, T = self.cfg.p, self.cfg.num_steps
        coinc = (activity.T @ activity) / activity.shape[0]
        dW = coinc - (p * T) ** 2
        dW.fill_diagonal_(0)
        self.W_inh.add_(alpha * dW)
        self.W_inh.clamp_(min=0.0)

    @torch.no_grad()
    def update_threshold(self, activity, gamma):              # SAILnet dtheta ~ n_i - p
        p, T = self.cfg.p, self.cfg.num_steps
        dtheta = (gamma * (activity - p * T)).mean(dim=0)
        self.theta.add_(dtheta)


class SAE(nn.Module):
    def __init__(self, cfg, spike_grad):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpikingEncoder(cfg, spike_grad)
        self.decoder = nn.Sequential(
            NeuronDecoder(cfg.N, cfg.n_in),
            snn.Leaky(beta=1, spike_grad=spike_grad, init_hidden=True,
                      output=True, threshold=200000),   # never spikes -> pure integrator
        )

    def forward(self, x):
        snnutils.reset(self.encoder)
        snnutils.reset(self.decoder)
        x = x.view(x.size(0), -1)                              # (B, 784)

        spk_rec_list, _ = self.encode(x)
        spk_rec = torch.stack(spk_rec_list, dim=2)             # (B, H, T)
        activity = spk_rec.sum(dim=2)                          # (B, H)  n_i

        spk_mem2 = []
        for step in range(self.cfg.num_steps):
            _, x_mem_recon = self.decode(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]        # (B, H, 784) last-step mem
        return spk_rec_list, x, out, activity

    def encode(self, x):
        return self.encoder(x, num_steps=self.cfg.num_steps)

    def decode(self, x):
        return self.decoder(x)


# ================================================================ the S'(C) gain term
def atan_surrogate_grad(u, alpha=2.0):
    """Derivative of snnTorch's ATan surrogate wrt (membrane - threshold) u."""
    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * u) ** 2)


def make_gain_fn(cfg):
    if cfg.use_chain_rule:
        return lambda u: atan_surrogate_grad(u, cfg.surrogate_alpha)
    return lambda u: torch.ones_like(u)


# ================================================================== paired Rule 1 update
@torch.no_grad()
def apply_paired_rule1(net, x, x_recon, activity, cfg, gain_fn):
    """Explicit local update of Q_ik, B_i, R_ik, B_ik for one batch (Rule 1)."""
    enc = net.encoder
    R = net.decoder[0].weight                               # (H, 784) R_ik
    e = x.unsqueeze(1) - x_recon                            # (B, H, 784) e_ik
    n = activity                                            # (B, H) n_i

    # feed-forward drive past total threshold -> argument of S'
    u = enc.fc(x) - cfg.theta_base - enc.theta              # (B, H)
    Sp = gain_fn(u)                                         # (B, H) S'(C_i)

    gate = (n > 0).float() if cfg.gating else torch.ones_like(n)   # (B, H)

    # -- gradient-descent updates (= -lr * dL/dparam), batch-averaged, factor 2 kept
    #    decoder bias   B_ik : -lr * (-2 e)           = +2 lr <g e>
    dB_ik = 2.0 * (gate.unsqueeze(-1) * e).mean(0)                         # (H, 784)
    #    decoder weight R_ik : -lr * (-2 n e)         = +2 lr <g n e>
    dR = 2.0 * ((gate * n).unsqueeze(-1) * e).mean(0)                      # (H, 784)
    #    encoder bias   B_i  : summed back-projection delta_i = sum_k R_ik e_ik
    delta = (R.unsqueeze(0) * e).sum(-1)                                   # (B, H)
    dB_i = 2.0 * (gate * Sp * delta).mean(0)                              # (H,)
    #    encoder weight Q_ik : LOCAL -- per-pixel error e_ik, not the sum
    gSp = (gate * Sp).unsqueeze(-1)                                       # (B, H, 1)
    dQ = 2.0 * (gSp * R.unsqueeze(0) * e * x.unsqueeze(1)).mean(0)        # (H, 784)

    enc.fc.weight.add_(cfg.lr_Q * dQ)
    enc.fc.bias.add_(cfg.lr_Bi * dB_i)
    net.decoder[0].weight.add_(cfg.lr_R * dR)
    net.decoder[0].bias.add_(cfg.lr_Bik * dB_ik)


@torch.no_grad()
def train_rule1(net, loader, epoch, cfg, gain_fn):
    net.train()
    last_loss = float("nan")
    for i, (img, _) in enumerate(loader):
        img = img.to(cfg.device)
        spk_list, x, x_recon, activity = net(U._normalize(img))

        apply_paired_rule1(net, x, x_recon, activity, cfg, gain_fn)
        net.encoder.update_inhibition(activity, cfg.alpha)
        net.encoder.update_threshold(activity, cfg.gamma)

        if i % 50 == 0:
            last_loss = U.recon_loss(x_recon, x, activity, cfg.gating).item()
            print(f"Train[{epoch + 1}/{cfg.epochs}][{i}/{len(loader)}] "
                  f"Loss: {last_loss:.5f}")
    return last_loss


# ===================================================================== experiment loop
PARAM_SPECS = [
    ("encoder",      lambda net: net.encoder.fc.weight),
    ("decoder",      lambda net: net.decoder[0].weight),
    ("decoder_bias", lambda net: net.decoder[0].bias),
]


def run(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    print("PyTorch", torch.__version__, "| device:", cfg.device)

    train_dataset, _test_dataset, train_loader, test_loader = _data(cfg)
    spike_grad = surrogate.atan(alpha=cfg.surrogate_alpha)
    gain_fn = make_gain_fn(cfg)

    from collections import defaultdict
    for seed in cfg.seeds:
        torch.manual_seed(seed)
        net = SAE(cfg, spike_grad).to(cfg.device)
        # explicit rule trains these; no autograd optimiser needed for the NAN
        for prm in net.parameters():
            prm.requires_grad_(False)

        metrics = defaultdict(list)
        param_hist = {name: defaultdict(list) for name, _ in PARAM_SPECS}

        for e in range(cfg.epochs + 1):
            if e % cfg.test_every == 0:
                metrics["epoch"].append(e)
                r_eff, avg_rate, avg_thresh, inhib_prop = U.test_encoder(
                    net, test_loader, cfg.device, cfg.N)
                metrics["r_eff"].append(r_eff)
                metrics["avg_rate"].append(avg_rate)
                metrics["avg_thresh"].append(avg_thresh)
                metrics["inhibition_prop"].append(inhib_prop)

                for name, getter in PARAM_SPECS:
                    mat = U.to_numpy(getter(net))
                    if mat.ndim == 2:
                        U.save_feature_grid(mat, name, e, cfg.save_dir)
                    param_hist[name]["mean"].append(mat.mean())
                    param_hist[name]["std"].append(mat.std())
                    param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                    param_hist[name]["eff_rank"].append(U.alt_effective_rank(mat))

                U.save_matrix(U.to_numpy(net.encoder.W_inh),
                              "Inhibition Weights", e, "inhib", cfg.save_dir)

                # read-out probe needs gradients -> run outside the no-grad rule loop
                metrics["accuracy"].append(
                    U.train_readout(net, seed, train_loader, test_loader,
                                    cfg.device, cfg.N))

            if e == cfg.epochs:
                break
            train_rule1(net, train_loader, e, cfg, gain_fn)

        _final_plots(net, train_dataset, metrics, param_hist, cfg)


def _data(cfg):
    return U.get_mnist_loaders(cfg.data_dir, cfg.batch_size)


def _final_plots(net, train_dataset, metrics, param_hist, cfg):
    stats = U.record_neuron_stats(net, train_dataset, cfg.device, cfg.batch_size,
                                  cfg.N, top_k=1)
    U.save_feature_grid(U.to_numpy(torch.squeeze(stats["top_imgs_current"])),
                        "Highest_Current_Image", 0, cfg.save_dir)

    idx = stats["top_by_current"]
    deltas = np.array([stats["deltas"][idx[0, i], i].item() for i in range(cfg.N)])
    firings = np.array([stats["firings"][idx[0, i], i].item() for i in range(cfg.N)])
    U.save_line_plot(range(cfg.N), [(None, deltas, None)],
                     "Deltas for Highest Current Images", "Delta", "deltas.png",
                     cfg.save_dir, xlabel="Neuron ID")
    U.save_line_plot(range(cfg.N), [(None, firings, None)],
                     "Activity for Highest Current Image", "Number of Firings",
                     "firings.png", cfg.save_dir, xlabel="Neuron ID")

    x = metrics["epoch"]
    U.save_line_plot(x, [(None, metrics["accuracy"], None)],
                     "Read-out Test Accuracy across Epochs", "Test Set Accuracy",
                     "Test_accuracy.png", cfg.save_dir)
    U.save_line_plot(x, [(None, metrics["r_eff"], None)],
                     "Encoder Activation Effective Rank across Epochs", "R_eff",
                     "eff_rank.png", cfg.save_dir)
    U.save_line_plot(x, [(None, metrics["avg_rate"], None)],
                     "Average Encoder Firing Rate across Epochs", "Firing Rate",
                     "firing_rate.png", cfg.save_dir)
    U.save_line_plot(x, [(None, metrics["avg_thresh"], None)],
                     "Average Encoder Neuron Threshold across Epochs",
                     "Average Threshold", "Threshold.png", cfg.save_dir)
    U.save_line_plot(x, [(None, metrics["inhibition_prop"], None)],
                     "Proportion of W_inh > 0 across Epochs", "W_inh > 0",
                     "W_inh.png", cfg.save_dir)

    U.save_line_plot(x, [(f"mean |{name}|", param_hist[name]["abs_mean"], None)
                         for name, _ in PARAM_SPECS],
                     "Mean of Absolute Parameters across Epochs", "",
                     "Absolute_Means.png", cfg.save_dir)
    U.save_line_plot(x, [(f"mean {name}", param_hist[name]["mean"],
                          param_hist[name]["std"]) for name, _ in PARAM_SPECS],
                     "Mean (+/-std) of Signed Parameters across Epochs", "",
                     "Signed_Means.png", cfg.save_dir)
    U.save_line_plot(x, [(f"{name} r_eff", param_hist[name]["eff_rank"], None)
                         for name, _ in PARAM_SPECS],
                     "Effective Rank of Parameters across Epochs", "R_eff",
                     "params_R_eff.png", cfg.save_dir)


if __name__ == "__main__":
    run(Config())
