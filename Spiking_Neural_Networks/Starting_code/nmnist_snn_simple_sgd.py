# ============================================================================
# A TRADITIONAL fully-connected Spiking Neural Network for N-MNIST, trained with SGD.
#
# Goal: the plainest, most canonical snnTorch SNN that still works, so the moving
# parts are easy to see. Compared with the earlier Kaggle script, everything
# "fancy" has been removed on purpose:
#   * ONE fixed, shared membrane decay (beta) instead of per-neuron / learnable decay
#   * plain SGD (+momentum) instead of Adam
#   * the surrogate gradient is made EXPLICIT so you can see it
#   * a standard CrossEntropyLoss on the output membrane (nothing hidden in SF)
#   * num_steps stored on the module (no reliance on a global variable)
#
# Architecture:  input(1156) --fc1--> LIF hidden(128) --fc2--> LIF output(10)
# ============================================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

# ---- reproducibility: seed ALL the RNGs we actually use --------------------
# (The earlier script only seeded numpy, which left torch's weight init and the
#  DataLoader shuffle non-deterministic. Seeding all three makes runs repeatable.)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters, all in one place -------------------------------------
NUM_INPUTS  = 34 * 34      # 1156 input neurons, one per pixel of the 34x34 sensor
NUM_HIDDEN  = 20         # size of the single hidden layer
NUM_OUTPUTS = 10           # one output neuron per digit class 0-9

# beta = per-time-step membrane decay (the leak).  U[t+1] = beta*U[t] + I[t+1] - reset
# Relation to a continuous LIF time constant: beta = exp(-dt/tau_mem).
# Fixed and shared across all neurons here — the simplest possible choice.
BETA = 0.9

# Time discretization of the event stream (see event_to_spike_train).
# DELTA_T_US = width of each time bin in microseconds; NUM_STEPS = number of bins.
# N-MNIST recordings are ~300 ms, so 100 bins x 3 ms ~= 300 ms of coverage.
# These are your main speed/resolution knob: fewer, wider bins = faster training
# (shorter backprop-through-time) at the cost of temporal detail. The original
# used 1 ms bins x 350 steps, which is ~3.5x more expensive.
DELTA_T_US = 3000
NUM_STEPS  = 100
SENSOR = 34
BATCH_SIZE = 128
EPOCHS     = 3
LR         = 0.5           # SGD learning rate. SGD is MORE sensitive than Adam;
MOMENTUM   = 0.9           # if loss stalls, tune LR first (0.1-1.0 is typical here).


# ============================================================================
# DATA PIPELINE
# N-MNIST is event-based: each sample is a stream of (x, y, polarity, timestamp)
# events from a neuromorphic camera watching an MNIST digit move. We parse the
# raw bytes, then bin the events into a (time, neuron) spike tensor the SNN can
# step through. (Parsing logic is unchanged from the original — it's correct.)
# ============================================================================

def read_nmnist_file(file_path):
    """Parse one N-MNIST binary file into parallel event lists.

    Format: each event is 5 bytes (40 bits), big-endian:
        bits 39..32 -> x (8b),  bits 31..24 -> y (8b),
        bit 23 -> polarity,     bits 22..0 -> timestamp in microseconds.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    b = np.frombuffer(raw, dtype=np.uint8)
    b = b[:(len(b) // 5) * 5].reshape(-1, 5).astype(np.uint32)  # (N_events, 5)
 
    x = b[:, 0]                                   # byte 0
    y = b[:, 1]                                   # byte 1
    p = (b[:, 2] >> 7) & 1                        # top bit of byte 2
    t = ((b[:, 2] & 0x7F) << 16) | (b[:, 3] << 8) | b[:, 4]   # low 23 bits
    return x, y, p, t


def event_to_spike_train(x, y, p, t):
    """Bin events into a (NUM_STEPS, NUM_INPUTS) binary spike tensor.

    Rows = time bins (width DELTA_T_US), cols = flattened pixel index y*34 + x.
    A 1 means "at least one event hit this pixel during this time bin".

    Simplifications kept deliberately simple:
      * polarity is ignored (ON and OFF events both write a 1),
      * values are binary, not counts.
    One robustness fix vs. the original: we CLAMP the bin index to the last bin,
    so events past our time window can't cause an IndexError.
    """
    st = torch.zeros(NUM_STEPS, NUM_INPUTS)
    # Cast to int64 FIRST. If x/y arrive as uint8, "y * 34 + x" overflows uint8
    # (33*34 = 1122 > 255) and silently wraps -> wrong neuron indices. This bit
    # us exactly once; casting up front makes the function safe for any int dtype.
    x = x.astype(np.int64); y = y.astype(np.int64); t = t.astype(np.int64)
    step   = np.minimum(t // DELTA_T_US, NUM_STEPS - 1)
    neuron = y * SENSOR + x
    st[step, neuron] = 1
    return st


class NMNISTDataset(torch.utils.data.Dataset):
    """Wraps (file_path, label) pairs; parses + bins lazily on access."""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, target = self.file_paths[idx]
        events = read_nmnist_file(file_path)
        spike_train = event_to_spike_train(*events)   # (NUM_STEPS, NUM_INPUTS)
        return spike_train, target


def get_file_paths_and_targets(root_dir):
    """Each recording's label is the name of the folder it lives in (0-9)."""
    file_paths = []
    for root, _dirs, files in os.walk(root_dir):
        for file in files:
            label = int(os.path.basename(root))
            file_paths.append((os.path.join(root, file), label))
    return file_paths


train_files = get_file_paths_and_targets("n-mnist_data/Train/Train")
test_files  = get_file_paths_and_targets("n-mnist_data/Test/Test")

train_dataset = NMNISTDataset(train_files)
test_dataset  = NMNISTDataset(test_files)

# num_workers=0 keeps things simple (single process). Because parsing happens on
# the fly on the CPU, data loading is the throughput bottleneck; if you want it
# faster, raise num_workers (and guard the run in `if __name__ == "__main__":`).
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================================
# THE SURROGATE GRADIENT  (made explicit on purpose)
# ----------------------------------------------------------------------------
# A spike is a hard threshold: S = Heaviside(U - threshold). Its derivative is a
# Dirac delta (zero everywhere, undefined at threshold) so ordinary backprop
# can't learn through it. The FIX is to swap in a smooth "surrogate" derivative
# during the BACKWARD pass only; the forward pass still emits crisp spikes.
#
# Here we pick the fast-sigmoid surrogate explicitly and hand it to every LIF
# layer via `spike_grad`. (If you omit spike_grad, snnTorch silently uses its
# default ATan surrogate — which is exactly why it was invisible in the other
# script.) `slope` controls how sharp the approximation is: larger = closer to
# the true step but noisier gradients; smaller = smoother but more biased.
# ============================================================================
spike_grad = surrogate.fast_sigmoid(slope=25)


# ============================================================================
# THE NETWORK
# ============================================================================
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, num_steps):
        super().__init__()
        self.num_steps = num_steps   # stored on the module (not a global)

        # A "layer" here = ordinary Linear weights feeding a bank of LIF neurons.
        # The Linear does the usual weighted sum; the LIF adds membrane leak +
        # threshold-and-fire dynamics over time. Same fixed beta for both layers,
        # not learnable — the plain-vanilla setup.
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """x: (batch, num_steps, num_inputs) binary spike trains.

        Returns:
            spk_rec: (num_steps, batch, num_outputs) output spikes  -> used for the prediction
            mem_rec: (num_steps, batch, num_outputs) output membrane -> used for the loss
        """
        # Reset/allocate membrane state at t=0. Must happen every forward pass,
        # otherwise state would bleed from one sample into the next.
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec, mem_rec = [], []

        # --- unroll over time (this loop IS the backprop-through-time graph) ---
        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])        # weighted input to hidden layer
            spk1, mem1 = self.lif1(cur1, mem1)    # hidden LIF: spikes out, membrane carried forward
            cur2 = self.fc2(spk1)                 # hidden spikes drive the output layer
            spk2, mem2 = self.lif2(cur2, mem2)    # output LIF

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


net = SNN(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS, BETA, NUM_STEPS).to(device)


# ============================================================================
# LOSS + OPTIMIZER
# ----------------------------------------------------------------------------
# We treat the output neurons' MEMBRANE POTENTIAL at each time step as class
# "logits" and apply a standard CrossEntropyLoss at every step, then average
# over time. This is the canonical snnTorch classification recipe and uses only
# familiar PyTorch pieces. Averaging over time (rather than summing) keeps the
# loss magnitude independent of NUM_STEPS, so a good learning rate doesn't have
# to change when you change the number of time bins.
#
# Optimizer: plain SGD with momentum — the traditional first choice.
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0)


def spike_count_accuracy(spk_rec, targets):
    """Prediction = the output neuron that spiked most over the whole window
    (rate decoding). Returns accuracy for the batch."""
    _, predicted = spk_rec.sum(dim=0).max(dim=1)   # sum over time -> (batch, 10) -> argmax
    return (predicted == targets).float().mean().item()


@torch.no_grad()
def evaluate(loader):
    """Full-pass accuracy on a data loader (used for the test set)."""
    net.eval()
    correct, total = 0, 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_rec, _ = net(data)
        _, predicted = spk_rec.sum(dim=0).max(dim=1)
        correct += (predicted == targets).sum().item()
        total   += targets.size(0)
    return correct / total


# ============================================================================
# TRAINING LOOP
# ============================================================================
loss_hist, acc_hist = [], []

for epoch in range(EPOCHS):
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)   # data: (batch, num_steps, 1156)

        net.train()
        spk_rec, mem_rec = net(data)                 # forward (builds the BPTT graph)

        # accumulate CrossEntropy over time on the output membrane, then average
        loss = torch.zeros(1, device=device)
        for step in range(NUM_STEPS):
            loss = loss + criterion(mem_rec[step], targets)
        loss = loss / NUM_STEPS

        optimizer.zero_grad()   # clear last step's gradients
        loss.backward()         # backprop THROUGH TIME; spikes are differentiated
                                # via the fast-sigmoid surrogate defined above
        optimizer.step()        # SGD update on fc1/fc2 weights & biases

        loss_hist.append(loss.item())

        if i % 20 == 0:
            acc = spike_count_accuracy(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Epoch {epoch}  Iter {i:4d}  Loss {loss.item():.3f}  BatchAcc {acc*100:5.1f}%")

    # end-of-epoch check on held-out test data
    test_acc = evaluate(test_loader)
    print(f"==> Epoch {epoch} finished | Test accuracy: {test_acc*100:.2f}%\n")

print("Training complete.")
