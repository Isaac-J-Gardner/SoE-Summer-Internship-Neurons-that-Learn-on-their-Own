"""
nan_core.py
================
Shared engine for all four paired learning rules of the spiking Neuron-Autoencoder.

The four options differ along two binary axes, both set by flags on ``Config``:

    rule   tie_weights   decoder_bias
    ----   -----------   ------------
      1      False          True        R_ik independent, decoder bias B_ik
      2      False          False       R_ik independent, no decoder bias
      3      True           True        R_ik = Q_ik (tied), decoder bias B_ik
      4      True           False       R_ik = Q_ik (tied), no decoder bias   (SAILnet-like)

One unified ``apply_paired_rule`` implements all four (derivations in the reference
sheet). The encoder bias B_i is present throughout (it is the fc bias); "bias" in the
four options refers to the *decoder* bias B_ik.

Per-sample, per-neuron (i) / per-pixel (k) rule, with W = decoder weight actually used
(W = R if untied, W = Q if tied), e_ik = x_k - xbar_ik, n_i = total spikes, S' = S'(C_i):

    B_ik : dL/dB_ik = -2 e_ik                             (only if decoder_bias)
    R_ik : dL/dR_ik = -2 n_i e_ik                         (only if untied)
    B_i  : dL/dB_i  = -2 S' sum_k W_ik e_ik               (shared -> summed over k)
    Q_ik : untied   = -2 S' W_ik e_ik x_k                 (encoder chain, LOCAL)
           tied     = -2 (S' Q_ik e_ik x_k + n_i e_ik)    (encoder chain + decoder part)

LOCALITY: the encoder-chain term uses pixel k's own error e_ik, not the summed
back-projection sum_k W_ik e_ik that the true network gradient would use. That is the
deliberate SAILnet-style locality. B_i is genuinely shared, so it does sum over k.

S'(C_i) is set by ``gain_fn`` (see ``make_gain_fn``): surrogate derivative when
``use_chain_rule`` is True, or identically 1 (treat n_i as fixed) when False.
"""

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import utils as snnutils
from snntorch import surrogate

import nan_utils as U


# =============================================================================== config
RULE_PRESETS = {                      # (tie_weights, decoder_bias)
    1: (False, True),
    2: (False, False),
    3: (True,  True),
    4: (True,  False),
}


@dataclass
class Config:
    rule: int = 1                     # 1..4; sets tie_weights / decoder_bias below

    n_in: int = 784
    N: int = 20
    num_steps: int = 50
    theta_base: float = 2.0
    beta: float = 0.9

    alpha: float = 1.0                # SAILnet inhibition lr
    gamma: float = 0.1                # SAILnet threshold lr
    p: float = field(init=False)

    lr_Q: float = 0.01
    lr_R: float = 0.01
    lr_Bi: float = 0.01
    lr_Bik: float = 0.01

    surrogate_alpha: float = 2.0
    use_chain_rule: bool = True
    gating: bool = True

    epochs: int = 20
    batch_size: int = 100
    test_every: int = 4
    seeds: tuple = (0,)

    # collapse-detection thresholds (all tunable)
    dead_rate_thresh: float = 0.5     # mean spikes/image below this -> neuron "dead"
    sat_rate_frac: float = 0.8        # mean spikes/image above frac*num_steps -> "saturated"
    dead_frac_thresh: float = 0.5     # collapse if this fraction of neurons dead
    min_act_rank: float = 2.0         # collapse if activation effective rank below this

    make_plots: bool = True
    verbose: bool = True

    data_dir: str = "./data"
    save_dir: str = field(init=False)

    tie_weights: bool = field(init=False)
    decoder_bias: bool = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.p = 1.0 / self.N
        self.tie_weights, self.decoder_bias = RULE_PRESETS[self.rule]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = f"Post_Internship/Paired_rules/images_rule{self.rule}"


# =============================================================================== models
class NeuronDecoder(nn.Module):
    """xbar_ik = h_i * W_ik (+ B_ik). W is R_ik, or a reference to Q_ik when tied."""
    def __init__(self, cfg, encoder_weight):
        super().__init__()
        self.tie = cfg.tie_weights
        if self.tie:
            # keep a non-registered reference (tuple hides it from nn.Module)
            self._enc_w_box = (encoder_weight,)
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.randn(cfg.N, cfg.n_in) * 0.01)
        if cfg.decoder_bias:
            self.bias = nn.Parameter(torch.zeros(cfg.N, cfg.n_in))
        else:
            self.register_parameter("bias", None)

    def dec_weight(self):
        return self._enc_w_box[0] if self.tie else self.weight

    def forward(self, h):                         # h: (B, H)
        out = h.unsqueeze(-1) * self.dec_weight()
        if self.bias is not None:
            out = out + self.bias
        return out


class SpikingEncoder(nn.Module):
    """LIF encoder with SAILnet recurrent inhibition and adaptive thresholds."""
    def __init__(self, cfg, spike_grad):
        super().__init__()
        self.cfg = cfg
        self.n_hidden = cfg.N
        self.fc = nn.Linear(cfg.n_in, cfg.N)                      # Q_ik (+ B_i)
        self.lif = snn.Leaky(beta=cfg.beta, spike_grad=spike_grad,
                             threshold=cfg.theta_base)
        self.register_buffer("W_inh", torch.zeros(cfg.N, cfg.N))
        self.register_buffer("theta", torch.zeros(cfg.N))

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
        p, T = self.cfg.p, self.cfg.num_steps
        coinc = (activity.T @ activity) / activity.shape[0]
        dW = coinc - (p * T) ** 2
        dW.fill_diagonal_(0)
        self.W_inh.add_(alpha * dW)
        self.W_inh.clamp_(min=0.0)

    @torch.no_grad()
    def update_threshold(self, activity, gamma):
        p, T = self.cfg.p, self.cfg.num_steps
        dtheta = (gamma * (activity - p * T)).mean(dim=0)
        self.theta.add_(dtheta)


class SAE(nn.Module):
    def __init__(self, cfg, spike_grad):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpikingEncoder(cfg, spike_grad)
        self.decoder = nn.Sequential(
            NeuronDecoder(cfg, self.encoder.fc.weight),
            snn.Leaky(beta=1, spike_grad=spike_grad, init_hidden=True,
                      output=True, threshold=200000),
        )

    def forward(self, x):
        snnutils.reset(self.encoder)
        snnutils.reset(self.decoder)
        x = x.view(x.size(0), -1)
        spk_rec_list, _ = self.encoder(x, self.cfg.num_steps)
        spk_rec = torch.stack(spk_rec_list, dim=2)             # (B, H, T)
        activity = spk_rec.sum(dim=2)                          # (B, H)
        spk_mem2 = []
        for step in range(self.cfg.num_steps):
            _, x_mem_recon = self.decoder(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]        # (B, H, 784)
        return spk_rec_list, x, out, activity


# ================================================================ the S'(C) gain term
def atan_surrogate_grad(u, alpha=2.0):
    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * u) ** 2)


def make_gain_fn(cfg):
    if cfg.use_chain_rule:
        return lambda u: atan_surrogate_grad(u, cfg.surrogate_alpha)
    return lambda u: torch.ones_like(u)


# ============================================================== unified paired update
@torch.no_grad()
def apply_paired_rule(net, x, x_recon, activity, cfg, gain_fn):
    enc = net.encoder
    Q = enc.fc.weight
    W = enc.fc.weight if cfg.tie_weights else net.decoder[0].weight   # decoder weight
    e = x.unsqueeze(1) - x_recon                           # (B, H, 784)
    n = activity                                           # (B, H)
    u = enc.fc(x) - cfg.theta_base - enc.theta             # (B, H)
    Sp = gain_fn(u)                                        # (B, H)
    g = (n > 0).float() if cfg.gating else torch.ones_like(n)

    # decoder bias  B_ik
    if cfg.decoder_bias:
        dB_ik = 2.0 * (g.unsqueeze(-1) * e).mean(0)
        net.decoder[0].bias.add_(cfg.lr_Bik * dB_ik)

    # encoder bias  B_i : -2 S' sum_k W_ik e_ik   (summed back-projection)
    delta = (W.unsqueeze(0) * e).sum(-1)                   # (B, H)
    dB_i = 2.0 * (g * Sp * delta).mean(0)
    enc.fc.bias.add_(cfg.lr_Bi * dB_i)

    # weights
    gSp = (g * Sp).unsqueeze(-1)
    enc_chain = gSp * W.unsqueeze(0) * e * x.unsqueeze(1)  # (B, H, 784) LOCAL per-pixel
    if cfg.tie_weights:
        dec_part = (g * n).unsqueeze(-1) * e
        enc.fc.weight.add_(cfg.lr_Q * 2.0 * enc_chain.mean(0)
                           + cfg.lr_R * 2.0 * dec_part.mean(0))
    else:
        enc.fc.weight.add_(cfg.lr_Q * 2.0 * enc_chain.mean(0))
        dR = 2.0 * ((g * n).unsqueeze(-1) * e).mean(0)
        net.decoder[0].weight.add_(cfg.lr_R * dR)


# ============================================================== collapse detection
@torch.no_grad()
def collapse_report(net, loader, cfg, n_batches=5):
    """Cheap health check from a few eval batches. Returns a metrics/flags dict."""
    net.eval()
    acts = []
    for bi, (img, _) in enumerate(loader):
        if bi >= n_batches:
            break
        _, _, _, activity = net(U._normalize(img.to(cfg.device)))
        acts.append(activity)
    A = torch.cat(acts, 0)                                 # (samples, H)
    rate = A.mean(0)                                       # mean spikes/image per neuron
    dead_frac = (rate < cfg.dead_rate_thresh).float().mean().item()
    sat_frac = (rate > cfg.sat_rate_frac * cfg.num_steps).float().mean().item()
    act_rank = U.activation_effective_rank(A)
    mean_rate = A.float().mean().item()
    collapsed = (dead_frac >= cfg.dead_frac_thresh) or (act_rank < cfg.min_act_rank)
    return {"dead_frac": dead_frac, "sat_frac": sat_frac, "act_rank": act_rank,
            "mean_rate": mean_rate, "collapsed": collapsed}


def _collapse_banner(rep, epoch):
    flags = []
    if rep["collapsed"]:
        flags.append("COLLAPSE")
    if rep["dead_frac"] >= 0.5:
        flags.append(f"dead={rep['dead_frac']:.0%}")
    if rep["sat_frac"] >= 0.25:
        flags.append(f"sat={rep['sat_frac']:.0%}")
    tag = ("  <<< " + ", ".join(flags)) if flags else ""
    return (f"  [health e{epoch}] rate={rep['mean_rate']:.2f} "
            f"act_rank={rep['act_rank']:.2f} dead={rep['dead_frac']:.0%}{tag}")


# ===================================================================== training loops
@torch.no_grad()
def train_epoch(net, loader, epoch, cfg, gain_fn):
    net.train()
    last = float("nan")
    for i, (img, _) in enumerate(loader):
        spk_list, x, x_recon, activity = net(U._normalize(img.to(cfg.device)))
        apply_paired_rule(net, x, x_recon, activity, cfg, gain_fn)
        net.encoder.update_inhibition(activity, cfg.alpha)
        net.encoder.update_threshold(activity, cfg.gamma)
        if cfg.verbose and i % 50 == 0:
            last = U.recon_loss(x_recon, x, activity, cfg.gating).item()
            print(f"Train[{epoch + 1}/{cfg.epochs}][{i}/{len(loader)}] Loss: {last:.5f}")
    return last


def _param_specs(cfg):
    specs = [("encoder", lambda net: net.encoder.fc.weight)]
    if not cfg.tie_weights:
        specs.append(("decoder", lambda net: net.decoder[0].weight))
    if cfg.decoder_bias:
        specs.append(("decoder_bias", lambda net: net.decoder[0].bias))
    return specs


def run(cfg):
    """Train one rule; returns a summary dict. Honours cfg.make_plots / cfg.verbose."""
    if cfg.make_plots:
        os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.verbose:
        print(f"\n=== Rule {cfg.rule} (tie={cfg.tie_weights}, "
              f"decoder_bias={cfg.decoder_bias}, chain_rule={cfg.use_chain_rule}) "
              f"| lr_Bik={cfg.lr_Bik} | device={cfg.device} ===")

    train_dataset, _test, train_loader, test_loader = U.get_mnist_loaders(
        cfg.data_dir, cfg.batch_size)
    spike_grad = surrogate.atan(alpha=cfg.surrogate_alpha)
    gain_fn = make_gain_fn(cfg)
    param_specs = _param_specs(cfg)

    summary = {"rule": cfg.rule, "lr_Bik": cfg.lr_Bik,
               "ever_collapsed": False, "history": defaultdict(list)}

    for seed in cfg.seeds:
        torch.manual_seed(seed)
        net = SAE(cfg, spike_grad).to(cfg.device)
        for prm in net.parameters():
            prm.requires_grad_(False)

        metrics = defaultdict(list)
        param_hist = {name: defaultdict(list) for name, _ in param_specs}

        for e in range(cfg.epochs + 1):
            if e % cfg.test_every == 0:
                metrics["epoch"].append(e)
                r_eff, avg_rate, avg_thresh, inhib_prop = U.test_encoder(
                    net, test_loader, cfg.device, cfg.N)
                metrics["r_eff"].append(r_eff)
                metrics["avg_rate"].append(avg_rate)
                metrics["avg_thresh"].append(avg_thresh)
                metrics["inhibition_prop"].append(inhib_prop)

                rep = collapse_report(net, test_loader, cfg)
                for k in ("dead_frac", "sat_frac", "act_rank", "mean_rate"):
                    metrics[k].append(rep[k])
                summary["ever_collapsed"] |= rep["collapsed"]
                if cfg.verbose:
                    print(_collapse_banner(rep, e))

                if cfg.make_plots:
                    for name, getter in param_specs:
                        mat = U.to_numpy(getter(net))
                        if mat.ndim == 2:
                            U.save_feature_grid(mat, name, e, cfg.save_dir)
                        param_hist[name]["mean"].append(mat.mean())
                        param_hist[name]["std"].append(mat.std())
                        param_hist[name]["abs_mean"].append(np.abs(mat).mean())
                        param_hist[name]["eff_rank"].append(U.alt_effective_rank(mat))
                    U.save_matrix(U.to_numpy(net.encoder.W_inh),
                                  "Inhibition Weights", e, "inhib", cfg.save_dir)

                metrics["accuracy"].append(
                    U.train_readout(net, seed, train_loader, test_loader,
                                    cfg.device, cfg.N) if cfg.verbose
                    else _quiet_readout(net, seed, train_loader, test_loader, cfg))

            if e == cfg.epochs:
                break
            train_epoch(net, train_loader, e, cfg, gain_fn)

        if cfg.make_plots:
            _final_plots(net, train_dataset, metrics, param_hist, param_specs, cfg)

        for k, v in metrics.items():
            summary["history"][k] = v

    # final-epoch summary numbers
    summary["final_acc"] = metrics["accuracy"][-1] if metrics["accuracy"] else float("nan")
    summary["final_act_rank"] = metrics["act_rank"][-1]
    summary["final_dead_frac"] = metrics["dead_frac"][-1]
    summary["final_mean_rate"] = metrics["mean_rate"][-1]
    summary["final_collapsed"] = (summary["final_dead_frac"] >= cfg.dead_frac_thresh
                                  or summary["final_act_rank"] < cfg.min_act_rank)
    return summary


def _quiet_readout(net, seed, train_loader, test_loader, cfg):
    """train_readout without the per-epoch prints (for sweeps)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return U.train_readout(net, seed, train_loader, test_loader, cfg.device, cfg.N)


# ===================================================================== lr_Bik sweep
def sweep_lr_Bik(base_cfg, values):
    """Run `base_cfg`'s rule for each lr_Bik in `values`; report collapse + accuracy."""
    results = []
    for v in values:
        cfg = replace(base_cfg, lr_Bik=v, make_plots=False, verbose=False, seeds=(0,))
        s = run(cfg)
        results.append(s)
        print(f"lr_Bik={v:<10.4g} acc={s['final_acc']:.3f} "
              f"act_rank={s['final_act_rank']:.2f} dead={s['final_dead_frac']:.0%} "
              f"rate={s['final_mean_rate']:.2f} "
              f"{'COLLAPSED' if s['final_collapsed'] or s['ever_collapsed'] else 'ok'}")

    if base_cfg.make_plots:
        os.makedirs(base_cfg.save_dir, exist_ok=True)
        vs = [s["lr_Bik"] for s in results]
        U.save_line_plot(
            vs,
            [("read-out acc", [s["final_acc"] for s in results], None),
             ("act. eff. rank", [s["final_act_rank"] for s in results], None),
             ("dead fraction", [s["final_dead_frac"] for s in results], None)],
            f"Rule {base_cfg.rule}: decoder-bias lr sweep",
            "value", f"sweep_lr_Bik_rule{base_cfg.rule}.png",
            base_cfg.save_dir, xlabel="lr_Bik")
    return results


# ===================================================================== final plots
def _final_plots(net, train_dataset, metrics, param_hist, param_specs, cfg):
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
    simple = [("accuracy", "Read-out Test Accuracy", "Test Set Accuracy", "Test_accuracy.png"),
              ("r_eff", "Encoder Activation Effective Rank", "R_eff", "eff_rank.png"),
              ("avg_rate", "Average Encoder Firing Rate", "Firing Rate", "firing_rate.png"),
              ("avg_thresh", "Average Encoder Threshold", "Threshold", "Threshold.png"),
              ("inhibition_prop", "Proportion of W_inh > 0", "W_inh > 0", "W_inh.png"),
              ("act_rank", "Activation Effective Rank (collapse)", "rank", "collapse_rank.png"),
              ("dead_frac", "Dead-neuron Fraction (collapse)", "fraction", "collapse_dead.png")]
    for key, title, ylab, fname in simple:
        U.save_line_plot(x, [(None, metrics[key], None)], title + " across Epochs",
                         ylab, fname, cfg.save_dir)

    U.save_line_plot(x, [(f"mean |{name}|", param_hist[name]["abs_mean"], None)
                         for name, _ in param_specs],
                     "Mean of Absolute Parameters across Epochs", "",
                     "Absolute_Means.png", cfg.save_dir)
    U.save_line_plot(x, [(f"mean {name}", param_hist[name]["mean"],
                          param_hist[name]["std"]) for name, _ in param_specs],
                     "Mean (+/-std) of Signed Parameters across Epochs", "",
                     "Signed_Means.png", cfg.save_dir)
    U.save_line_plot(x, [(f"{name} r_eff", param_hist[name]["eff_rank"], None)
                         for name, _ in param_specs],
                     "Effective Rank of Parameters across Epochs", "R_eff",
                     "params_R_eff.png", cfg.save_dir)
