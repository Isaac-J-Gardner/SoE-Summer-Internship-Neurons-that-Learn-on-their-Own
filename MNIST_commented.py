##############################################################################
# IMPORTS
##############################################################################

import argparse
# argparse: Python's standard library for parsing command-line arguments.
# Lets you run: python mnist.py --epochs 5 --lr 0.5 etc.

import torch
# The core PyTorch library. Provides:
#   - torch.Tensor: the fundamental n-dimensional array (like numpy but with GPU support + autograd)
#   - Autograd engine: automatic differentiation — records every operation on tensors so
#     .backward() can compute gradients via the chain rule
#   - Device management: moving data between CPU and GPU

import torch.nn as nn
# torch.nn: contains all the building blocks for neural networks.
#   - nn.Module: the base class every model inherits from. It:
#       1) Tracks all learnable parameters (weights/biases) so the optimizer can find them
#       2) Provides .train()/.eval() mode switching (affects dropout, batchnorm, etc.)
#       3) Handles .to(device) to move all parameters to CPU/GPU at once
#   - nn.Conv2d, nn.Linear, nn.Dropout: layer classes that hold their own parameters
#   - nn.Parameter: a tensor that is auto-registered as a learnable weight when assigned
#     as an attribute of an nn.Module

import torch.nn.functional as F
# torch.nn.functional: stateless versions of operations — same math as nn layers but
# without stored parameters. Used for:
#   - Activations: F.relu, F.log_softmax (no learnable weights, so no need for a module)
#   - Pooling: F.max_pool2d (no learnable weights)
#   - Loss functions: F.nll_loss
# Rule of thumb: if the operation HAS learnable parameters → use nn.Module version (nn.Linear)
#                if it's purely a function (relu, pooling, loss) → use F.xxx

import torch.optim as optim
# torch.optim: optimization algorithms that update model weights using gradients.
#   - Each optimizer wraps model.parameters() and applies a specific update rule:
#       SGD:      w = w - lr * grad
#       Adam:     w = w - lr * (adaptive per-parameter learning rate using momentum + RMSprop)
#       Adadelta: w = w - (RMS of recent weight updates / RMS of recent gradients) * grad
#         Adadelta is used here — it adapts the learning rate per-parameter using a running
#         average of squared gradients, so it's less sensitive to the initial lr choice.

from torchvision import datasets, transforms
# torchvision.datasets: pre-built dataset classes for common vision benchmarks.
#   datasets.MNIST handles downloading, caching, and loading the 70,000 handwritten digits.
#   It returns (PIL_Image, label) pairs — the transforms convert PIL→Tensor.
#
# torchvision.transforms: image preprocessing pipeline.
#   transforms.Compose chains multiple transforms: PIL→Tensor→Normalize

from torch.optim.lr_scheduler import StepLR
# StepLR: decays the learning rate by a multiplicative factor (gamma) every N epochs.
# Why? Early in training, large lr = fast convergence toward the right region.
# Later, smaller lr = fine-tuning to settle into a sharper minimum.
# Without decay, the optimizer might oscillate around the minimum instead of converging.


##############################################################################
# MODEL DEFINITION
##############################################################################

class Net(nn.Module):
    # nn.Module is the base class for all neural network modules in PyTorch.
    # By inheriting from it, our Net class gets:
    #   1) Automatic parameter registration: any nn.Module or nn.Parameter assigned as
    #      self.xxx is tracked. model.parameters() iterates all of them.
    #   2) .train() / .eval() mode: flips a flag that layers like Dropout check.
    #   3) .to(device): recursively moves every parameter to the target device.
    #   4) Hooks, state_dict serialization, and more.

    def __init__(self):
        super(Net, self).__init__()
        # super().__init__() calls nn.Module's constructor, which initializes
        # internal bookkeeping: the parameter registry, sub-module registry,
        # hooks, training-mode flag, etc. Without this call, none of the
        # automatic parameter tracking would work.

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        #
        # What this creates:
        #   - 32 convolutional filters (kernels), each of shape (1, 3, 3)
        #     = 1 input channel × 3 height × 3 width = 9 weights per filter
        #   - 32 bias terms (one per output channel)
        #   - Total learnable parameters: 32 * (1*3*3) + 32 = 320
        #
        # What it does during forward pass:
        #   - Each filter slides across the input image with stride=1
        #   - At each position, it computes: sum(filter * input_patch) + bias
        #   - With no padding (default padding=0), a 28×28 input becomes 26×26
        #     because: output_size = (input_size - kernel_size) / stride + 1
        #                          = (28 - 3) / 1 + 1 = 26
        #
        # Input shape:  (batch, 1, 28, 28)   — 1 grayscale channel
        # Output shape: (batch, 32, 26, 26)  — 32 feature maps
        #
        # Intuition: each filter learns to detect a different low-level feature:
        # edges at various angles, corners, curves, blobs. The 32 output "images"
        # are activation maps showing WHERE each feature was detected.

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        #
        # What this creates:
        #   - 64 filters, each of shape (32, 3, 3)
        #     = 32 input channels × 3 × 3 = 288 weights per filter
        #   - 64 bias terms
        #   - Total learnable parameters: 64 * (32*3*3) + 64 = 18,496
        #
        # CRITICAL DETAIL — how multi-channel convolution works:
        #   Each of the 64 filters has 32 "sub-kernels" (one per input channel).
        #   For a single output pixel at position (r,c) in output channel k:
        #     output[k, r, c] = sum over all 32 channels of (3×3 patch × sub-kernel) + bias[k]
        #   So each output pixel is influenced by a 3×3×32 = 288-element volume of the input.
        #   This is how the network combines features across channels.
        #
        # Input shape:  (batch, 32, 26, 26)
        # Output shape: (batch, 64, 24, 24)  — again, 26-3+1=24
        #
        # Intuition: these filters learn to detect COMBINATIONS of the low-level
        # features from conv1. E.g., "a horizontal edge ABOVE a curve" = part of a "2".

        self.dropout1 = nn.Dropout(0.25)
        # nn.Dropout(p=0.25)
        #
        # During training (.train() mode):
        #   - Each element of the input tensor is independently set to 0 with probability 0.25
        #   - Surviving elements are scaled up by 1/(1-0.25) = 1.333...
        #   - This scaling ensures the expected sum stays the same (so the next layer
        #     sees roughly the same magnitude whether in train or eval mode)
        #
        # During evaluation (.eval() mode):
        #   - Does nothing (identity function). No zeroing, no scaling.
        #
        # Why? Dropout is a regularization technique. By randomly killing neurons:
        #   1) It prevents co-adaptation: neurons can't rely on specific other neurons
        #      always being present, so each must learn independently useful features
        #   2) It's like training an ensemble of 2^N sub-networks simultaneously
        #   3) At test time, using all neurons with no dropout approximates averaging
        #      the predictions of all those sub-networks
        #
        # 0.25 is mild — used here right after the conv layers where we don't want
        # to throw away too much spatial information.

        self.dropout2 = nn.Dropout(0.5)
        # Same mechanism, but p=0.5 (half the neurons zeroed).
        # Used after the fully-connected layer where there are many more parameters
        # and overfitting risk is higher. The FC layers contain the vast majority of
        # the model's parameters (9216*128 + 128*10 ≈ 1.2M) compared to the conv
        # layers (~19K), so they need stronger regularization.

        self.fc1 = nn.Linear(9216, 128)
        # nn.Linear(in_features=9216, out_features=128)
        #
        # A fully-connected (dense) layer:
        #   - Weight matrix: shape (128, 9216) — 1,179,648 parameters
        #   - Bias vector: shape (128) — 128 parameters
        #   - Total: 1,179,776 parameters (the vast majority of the model!)
        #
        # Computation: output = input @ weight.T + bias
        #   For each of the 128 output neurons: dot product of all 9216 inputs with
        #   that neuron's 9216 weights, plus a bias.
        #
        # Input shape:  (batch, 9216) — flattened feature maps
        # Output shape: (batch, 128)
        #
        # Where does 9216 come from?
        #   After conv2: (batch, 64, 24, 24)
        #   After max_pool2d with kernel 2: (batch, 64, 12, 12)
        #   Flattened: 64 * 12 * 12 = 9,216
        #
        # This layer's job: learn which COMBINATIONS of spatial features (from the
        # conv layers) are diagnostic for each digit. The conv layers extracted
        # local features; this layer does global reasoning.

        self.fc2 = nn.Linear(128, 10)
        # nn.Linear(in_features=128, out_features=10)
        #
        # The classification head:
        #   - Weight matrix: (10, 128) — 1,280 parameters
        #   - Bias vector: (10) — 10 parameters
        #
        # Input shape:  (batch, 128)
        # Output shape: (batch, 10)
        #
        # Produces 10 raw scores (logits), one for each digit class 0-9.
        # These get passed through log_softmax to become log-probabilities.

    def forward(self, x):
        # forward() defines the computation graph — what happens to data as it flows
        # through the network. PyTorch's autograd records every operation here so it
        # can later compute gradients via .backward().
        #
        # x starts as shape: (batch_size, 1, 28, 28)
        #   batch_size: typically 64 (set by --batch-size arg)
        #   1: one grayscale color channel
        #   28×28: pixel dimensions of each MNIST image
        #
        # Pixel values have been normalized by transforms.Normalize((0.1307,), (0.3081,))
        # so they're centered around 0 with stdev ≈ 1, not raw 0-255 or 0.0-1.0.

        x = self.conv1(x)
        # Shape: (batch, 1, 28, 28) → (batch, 32, 26, 26)
        # Each of the 32 filters slides its 3×3 kernel across the image.
        # No padding → edges shrink by 1 pixel on each side → 28-2=26.

        x = F.relu(x)
        # Shape: unchanged — (batch, 32, 26, 26)
        # ReLU(x) = max(0, x) — applied element-wise to every value.
        #
        # Why a non-linearity is essential:
        #   Without it, stacking linear operations (conv, linear) collapses to a single
        #   linear operation: W2(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)
        #   No matter how many layers, it's equivalent to ONE linear layer.
        #   Non-linearities between layers let the network learn non-linear decision
        #   boundaries — which is what makes deep networks powerful.
        #
        # Why ReLU specifically?
        #   1) Computationally trivial (just a threshold)
        #   2) Gradient is either 0 or 1 — no vanishing gradient for positive values
        #   3) Induces sparsity: negative activations become exactly 0
        #   Downsides: "dying ReLU" — if a neuron's output is always negative, its
        #   gradient is always 0 and it never recovers. (Variants like LeakyReLU fix this.)

        x = self.conv2(x)
        # Shape: (batch, 32, 26, 26) → (batch, 64, 24, 24)
        # 64 filters, each with 32 sub-kernels of 3×3.
        # 26-2=24.

        x = F.relu(x)
        # Shape: unchanged — (batch, 64, 24, 24)

        x = F.max_pool2d(x, 2)
        # Shape: (batch, 64, 24, 24) → (batch, 64, 12, 12)
        #
        # Max pooling with kernel_size=2, stride=2 (stride defaults to kernel_size):
        #   Divides each 24×24 feature map into non-overlapping 2×2 blocks.
        #   Keeps only the maximum value from each block.
        #   24/2 = 12 in each spatial dimension.
        #
        # Why max pooling?
        #   1) Dimensionality reduction: cuts the number of values by 4× (24²→12² per channel)
        #      This means fewer parameters in the following FC layer and faster computation.
        #   2) Translation invariance: if a feature shifts by 1 pixel, the max within
        #      the 2×2 block often stays the same → the network becomes robust to small
        #      shifts in the input.
        #   3) Takes the strongest activation: "was this feature detected ANYWHERE in this
        #      2×2 region?" — a form of spatial summarization.
        #
        # Total values at this point: 64 channels × 12 × 12 = 9,216

        x = self.dropout1(x)
        # Shape: unchanged — (batch, 64, 12, 12)
        # In training: 25% of the 9,216 values randomly zeroed, rest scaled by 1.333.
        # In eval: no-op (pass-through).

        x = torch.flatten(x, 1)
        # Shape: (batch, 64, 12, 12) → (batch, 9216)
        #
        # Flattens all dimensions starting from dim=1 (preserving the batch dimension).
        # The 3D structure (channels × height × width) becomes a 1D vector per sample.
        #
        # This is necessary because nn.Linear expects a 1D input per sample.
        # We're transitioning from "spatial feature maps" to "global feature vector."
        # The spatial relationships are lost here — the FC layer treats all 9216 values
        # as independent inputs.

        x = self.fc1(x)
        # Shape: (batch, 9216) → (batch, 128)
        # Matrix multiplication: each of 128 neurons computes a weighted sum of all
        # 9216 inputs. This is where the network aggregates spatial features into
        # a compact 128-dimensional representation.

        x = F.relu(x)
        # Shape: unchanged — (batch, 128)
        # Non-linearity between the two FC layers.

        x = self.dropout2(x)
        # Shape: unchanged — (batch, 128)
        # In training: 50% of the 128 values zeroed, rest scaled by 2.0.
        # More aggressive than dropout1 because the FC layers have far more parameters
        # (1.18M in fc1 alone) and are more prone to memorizing training data.

        x = self.fc2(x)
        # Shape: (batch, 128) → (batch, 10)
        # Produces 10 raw scores (logits). The highest score = the model's prediction.

        output = F.log_softmax(x, dim=1)
        # Shape: unchanged — (batch, 10), but values are now log-probabilities.
        #
        # softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
        #   → converts 10 raw scores to probabilities that sum to 1.0
        #
        # log_softmax = log(softmax(x))
        #   → log-probabilities, range (-∞, 0]. More negative = less confident.
        #   → e.g., log(0.95) ≈ -0.05 (very confident), log(0.01) ≈ -4.6 (not confident)
        #
        # dim=1 means softmax is computed across the 10 classes (not across the batch).
        #
        # Why log_softmax instead of just softmax?
        #   1) Numerical stability: computing log(exp(x)/sum(exp(x))) directly can overflow
        #      on the exp() step. log_softmax uses the log-sum-exp trick to avoid this.
        #   2) Pairs with F.nll_loss: NLL loss expects log-probabilities. Together,
        #      log_softmax + nll_loss = cross-entropy loss, which is the standard loss
        #      for classification.

        return output


##############################################################################
# TRAINING LOOP
##############################################################################

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # Switches the model to training mode. This is NOT what starts training —
    # it sets a flag (self.training = True) that affects layer behavior:
    #   - Dropout layers: actively zero elements and rescale
    #   - BatchNorm layers (not used here): use batch statistics instead of running stats
    # The counterpart is model.eval() used in the test function.

    for batch_idx, (data, target) in enumerate(train_loader):
        # train_loader is a DataLoader that:
        #   1) Shuffles the 60,000 training images (if shuffle=True)
        #   2) Groups them into batches of 64
        #   3) Yields (data_tensor, label_tensor) tuples
        #
        # enumerate gives us the batch index (0, 1, 2, ..., 937) alongside each batch.
        # 60,000 / 64 = 937.5, so there are 938 batches per epoch (last one has 32 samples).
        #
        # data shape:   (64, 1, 28, 28) — 64 grayscale images
        # target shape: (64,)           — 64 integer labels, each in {0, 1, ..., 9}

        data, target = data.to(device), target.to(device)
        # Moves tensors to the same device (CPU or GPU) as the model.
        # All tensors in a computation must be on the same device.
        # .to(device) is a no-op if the tensor is already on the right device.
        # If moving to GPU, this triggers a CPU→GPU memory copy (relatively slow
        # compared to GPU computation, which is why pin_memory=True is used with
        # the DataLoader — it pre-stages the memory for faster transfer).

        optimizer.zero_grad()
        # Resets all parameter gradients to zero.
        #
        # Why is this needed? PyTorch ACCUMULATES gradients by default — each call to
        # .backward() ADDS to the .grad attribute rather than replacing it. This is
        # useful for some advanced techniques (e.g., gradient accumulation across
        # micro-batches), but for standard training you want fresh gradients each step.
        #
        # Without this, gradients from previous batches would contaminate the current
        # update, leading to incorrect optimization steps.

        output = model(data)
        # Calls model.forward(data) — the full forward pass defined above.
        #
        # PyTorch's autograd engine records every operation in a computational graph:
        #   data → conv1 → relu → conv2 → relu → maxpool → dropout → flatten
        #        → fc1 → relu → dropout → fc2 → log_softmax → output
        #
        # Each intermediate tensor knows what operation created it and what its inputs were.
        # This graph is what .backward() will traverse to compute gradients.
        #
        # output shape: (64, 10) — log-probabilities for 10 classes, for each of 64 images

        loss = F.nll_loss(output, target)
        # Negative Log-Likelihood Loss.
        #
        # For each sample i in the batch:
        #   loss_i = -output[i, target[i]]
        #
        # In other words: look up the log-probability the model assigned to the CORRECT
        # class, and negate it.
        #
        # Example: if target[0] = 7 and output[0] = [-3.2, -4.1, -5.0, -2.8, -6.1,
        #           -4.5, -3.9, -0.05, -7.2, -5.5]
        #   → loss_0 = -(-0.05) = 0.05  (very low loss — model is confident and correct)
        #
        # If the model assigned log-prob -4.6 to the correct class:
        #   → loss = 4.6  (high loss — model is not confident in the right answer)
        #
        # The final loss is the MEAN across the batch (default reduction='mean').
        #
        # Combined with log_softmax, this is mathematically identical to cross-entropy loss:
        #   CrossEntropy = -log(softmax(logit_correct_class))
        # Cross-entropy is THE standard loss for classification because:
        #   1) It's the negative log-likelihood under a categorical distribution
        #   2) Minimizing it = maximizing the probability assigned to correct labels
        #   3) Its gradient w.r.t. logits has a clean form: softmax(x) - one_hot(target)
        #
        # loss shape: scalar (single number) — the average loss across the batch

        loss.backward()
        # BACKPROPAGATION — the core of neural network training.
        #
        # Starting from the scalar loss, PyTorch traverses the computational graph
        # BACKWARDS, computing the gradient of the loss with respect to every tensor
        # that has requires_grad=True (all model parameters).
        #
        # It applies the chain rule at each node:
        #   d(loss)/d(weight) = d(loss)/d(output) * d(output)/d(intermediate) * ... * d(layer)/d(weight)
        #
        # After this call, every parameter tensor has its .grad attribute populated:
        #   model.conv1.weight.grad  — shape (32, 1, 3, 3), same as the weight
        #   model.fc1.weight.grad    — shape (128, 9216), same as the weight
        #   etc.
        #
        # The computational graph is then destroyed (by default) to free memory.
        # A new graph is built on the next forward pass.
        #
        # This is the most computationally expensive step — roughly 2× the cost of
        # the forward pass, because it must compute gradients for every operation.

        optimizer.step()
        # Applies the optimization algorithm using the computed gradients.
        #
        # For Adadelta specifically:
        #   Adadelta maintains two running averages per parameter:
        #     - E[g²]: exponential moving average of squared gradients
        #     - E[Δx²]: exponential moving average of squared parameter updates
        #   Update rule (simplified):
        #     RMS(g) = sqrt(E[g²] + ε)
        #     RMS(Δx) = sqrt(E[Δx²] + ε)
        #     Δx = -(RMS(Δx) / RMS(g)) * gradient
        #     x = x + Δx
        #
        #   Key insight: the learning rate is ADAPTIVE per-parameter. Parameters with
        #   consistently large gradients get smaller updates (denominator is large),
        #   and vice versa. The lr argument (default 1.0 here) is a global multiplier
        #   on top of this adaptive rate.
        #
        # After this call, all model weights have been updated. The .grad values are
        # NOT cleared (which is why we need optimizer.zero_grad() at the start).

        if batch_idx % args.log_interval == 0:
            # Every log_interval batches (default 10), print a progress report.
            # With batch_size=64 and 60,000 samples, there are 938 batches per epoch,
            # so this prints about 94 lines per epoch.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),       # number of samples processed so far
                len(train_loader.dataset),   # total samples (60,000)
                100. * batch_idx / len(train_loader),  # percentage complete
                loss.item()                  # .item() extracts a Python float from a scalar tensor
                                             # (avoids keeping the computation graph in memory)
            ))
            if args.dry_run:
                break
                # In dry-run mode, exit after the first log point.
                # Used for quick sanity checks: "does the code run at all?"


##############################################################################
# EVALUATION LOOP
##############################################################################

def test(model, device, test_loader):
    model.eval()
    # Switches to evaluation mode (self.training = False):
    #   - Dropout layers become identity (no zeroing) — use ALL neurons
    #   - BatchNorm uses running statistics instead of batch statistics
    # This ensures deterministic, reproducible predictions at test time.

    test_loss = 0
    correct = 0
    # Accumulators for computing average loss and accuracy across the full test set.

    with torch.no_grad():
        # Context manager that disables gradient computation.
        #
        # Why?
        #   1) We don't need gradients during evaluation (no .backward() call)
        #   2) Saves memory: PyTorch doesn't build the computational graph
        #   3) Speeds up computation: no gradient bookkeeping overhead
        #
        # Every tensor created inside this block has requires_grad=False,
        # regardless of the inputs' settings.

        for data, target in test_loader:
            # test_loader has batch_size=1000 and 10,000 test images → 10 batches.
            # data shape:   (1000, 1, 28, 28)
            # target shape: (1000,)

            data, target = data.to(device), target.to(device)

            output = model(data)
            # output shape: (1000, 10) — log-probabilities

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # reduction='sum' returns the SUM of losses across the batch (not the mean).
            # We accumulate the sum and divide by the dataset size later to get the
            # true average over all 10,000 test samples.
            #
            # Why not just use mean? Because the last batch might be smaller,
            # and averaging means of different-sized batches gives a biased result.
            # Summing all losses and dividing by total count is correct.

            pred = output.argmax(dim=1, keepdim=True)
            # argmax along dim=1: for each sample, find the class index with the
            # highest log-probability. This is the model's predicted digit.
            # keepdim=True: keeps the result as (1000, 1) instead of (1000,)
            # — needed for the .eq() comparison below.
            #
            # pred shape: (1000, 1)

            correct += pred.eq(target.view_as(pred)).sum().item()
            # target.view_as(pred): reshapes target from (1000,) to (1000, 1) to match pred's shape
            # .eq(): element-wise equality → a boolean tensor of shape (1000, 1)
            #   True where prediction matches the label, False otherwise
            # .sum(): counts the number of True values (correct predictions in this batch)
            # .item(): extracts as a Python int
            #
            # This accumulates the total number of correct predictions.

    test_loss /= len(test_loader.dataset)
    # Divide accumulated sum by total number of test samples (10,000)
    # to get the average loss per sample.

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),    # 10,000
        100. * correct / len(test_loader.dataset)  # accuracy as percentage
    ))

print("change")
# This line runs at import/load time, not inside any function.
# It's a debug marker left in the code — probably used to verify that a code
# change was actually loaded (e.g., after editing and re-running).


##############################################################################
# MAIN — ARGUMENT PARSING, DATA LOADING, AND TRAINING ORCHESTRATION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Creates an argument parser — a structured way to accept command-line configs.
    # Each add_argument call defines one option with its type, default, and help text.

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # Training batch size. Tradeoffs:
    #   Larger batch: more stable gradient estimates, better GPU utilization,
    #                 but more memory, and can converge to sharper (worse) minima
    #   Smaller batch: noisier gradients (acts as regularization), less memory,
    #                  but slower per epoch due to less parallelism
    # 64 is a common sweet spot for MNIST.

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    # Test batch can be much larger because we don't need to store gradients,
    # so memory usage is lower. 1000 means we process the 10K test set in 10 batches.

    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    # One epoch = one complete pass through all 60,000 training images.
    # 14 epochs means every image is seen 14 times.
    # With lr decay (gamma=0.7), the effective learning rate after 14 epochs is:
    #   1.0 * 0.7^14 ≈ 0.0068 — about 150× smaller than the initial lr.

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    # Initial learning rate. 1.0 seems high, but Adadelta's adaptive scaling means
    # the effective step size per parameter is much smaller than this raw value.
    # For SGD, 1.0 would likely diverge; for Adadelta, it's reasonable.

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    # Every epoch, lr is multiplied by gamma: lr_new = lr_old * 0.7
    # This implements a learning rate schedule that gradually reduces the step size.

    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    # Flag to force CPU-only training, even if a GPU/accelerator is available.
    # 'store_true' means: if the flag is present, set to True; default is False.

    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    # Run just one batch per epoch to verify the code works without errors.
    # Useful for development and debugging.

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # For reproducibility. Controls random weight initialization, dropout masks,
    # data shuffling order, etc.

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # Print training progress every N batches.

    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    # If set, saves the trained model weights to disk after training.

    args = parser.parse_args()
    # Parses sys.argv and returns a namespace object.
    # Access values as: args.batch_size, args.lr, args.epochs, etc.
    # Note: argparse converts '--batch-size' to args.batch_size (hyphens → underscores).

    use_accel = not args.no_accel and torch.accelerator.is_available()
    # Check if a hardware accelerator (GPU/MPS/etc.) is available AND not disabled.
    # torch.accelerator.is_available() returns True if CUDA, MPS, or another
    # accelerator backend is present.

    torch.manual_seed(args.seed)
    # Seeds PyTorch's random number generators for reproducibility.
    # Affects: random weight initialization, dropout mask generation, and
    # any other torch.rand/randn calls.
    # Note: for full reproducibility you'd also need to seed Python's random,
    # numpy, and set torch.backends.cudnn.deterministic = True.

    if use_accel:
        device = torch.accelerator.current_accelerator()
        # Gets the current accelerator device (e.g., cuda:0, mps:0).
    else:
        device = torch.device("cpu")
        # Explicit CPU device.

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    # Base keyword arguments for the DataLoader constructors.

    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                        'pin_memory': True,
                        'shuffle': True}
        # num_workers=1: Use 1 background process for data loading.
        #   This parallelizes data loading with GPU computation:
        #   while the GPU processes batch N, the CPU loads batch N+1.
        #
        # persistent_workers=True: Keep the worker process alive between batches.
        #   Without this, the worker is spawned and killed each batch — expensive
        #   on systems where process creation is slow.
        #
        # pin_memory=True: Allocates data in page-locked (pinned) CPU memory.
        #   Pinned memory enables faster CPU→GPU transfers via DMA (direct memory access)
        #   because the OS guarantees the memory won't be swapped to disk.
        #   Only beneficial when using a GPU — wastes memory on CPU-only.
        #
        # shuffle=True: Randomly reorder the training data each epoch.
        #   Prevents the model from learning patterns in the ordering of samples
        #   (e.g., if all 0s came first, then all 1s, the model might oscillate).
        #   Essential for stochastic gradient descent to work properly.
        #   Applied to test_kwargs too here, though shuffling test data doesn't affect
        #   the final accuracy — it's just for consistency.

        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Converts a PIL Image (H×W×C, uint8 0-255) to a FloatTensor (C×H×W, float32 0.0-1.0).
        # Two things happen:
        #   1) Dimension reorder: (28, 28, 1) → (1, 28, 28) — channels-first, which PyTorch expects
        #   2) Normalization: divide by 255.0 → values in [0.0, 1.0]

        transforms.Normalize((0.1307,), (0.3081,))
        # Standardizes each channel: pixel = (pixel - mean) / std
        #   mean=0.1307, std=0.3081 are the precomputed global mean and standard deviation
        #   of ALL pixels across the entire MNIST training set.
        #
        # After this transform, the pixel distribution has approximately mean=0, std=1.
        #
        # Why normalize?
        #   1) Helps optimization: gradient descent works best when inputs are centered
        #      around 0 and have similar scales. Without normalization, the loss landscape
        #      is elongated (different features have very different scales), making
        #      gradient descent take inefficient zig-zag paths.
        #   2) Matches the assumption of many initialization schemes (e.g., Xavier/He)
        #      which assume zero-mean, unit-variance inputs.
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # Loads (or downloads) the MNIST training split:
    #   - '../data': directory to store the downloaded files
    #   - train=True: use the 60,000 training images
    #   - download=True: if not already cached, download from the internet
    #   - transform: the Compose pipeline above is applied to each image when accessed
    #
    # The dataset is NOT loaded into memory all at once. It's accessed lazily —
    # dataset1[i] reads image i from disk, applies the transform, and returns
    # (tensor, label).

    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    # The test split: 10,000 held-out images never seen during training.
    # No download=True needed — it was already downloaded with the training set.
    # These images test how well the model GENERALIZES to new data.

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # DataLoader wraps a dataset and provides:
    #   1) Batching: groups individual samples into batches of batch_size
    #   2) Shuffling: randomizes order each epoch (if shuffle=True)
    #   3) Parallel loading: uses background workers (if num_workers > 0)
    #   4) Pinned memory: pre-stages data for GPU transfer (if pin_memory=True)
    #   5) Collation: stacks individual (1,28,28) tensors into (batch,1,28,28)
    #
    # The loader is an iterator: each iteration yields (batch_data, batch_targets).

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # Net(): instantiates the model, randomly initializing all weights.
    #   Conv2d uses Kaiming uniform initialization by default (designed for ReLU).
    #   Linear uses uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    #
    # .to(device): moves ALL model parameters to the specified device.
    #   On GPU: copies weight tensors from CPU RAM to GPU VRAM.
    #   This is recursive — it traverses all sub-modules (conv1, conv2, fc1, fc2, etc.).
    #
    # Total parameters: 32*1*3*3+32 + 64*32*3*3+64 + 128*9216+128 + 10*128+10
    #                 = 320 + 18,496 + 1,179,776 + 1,290 = 1,199,882 ≈ 1.2M parameters

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # Creates the optimizer, wrapping ALL model parameters.
    #
    # model.parameters() returns a generator over every nn.Parameter in the model.
    # The optimizer stores references to these tensors and will modify them in-place
    # when .step() is called.
    #
    # Adadelta: an adaptive learning rate optimizer. Unlike SGD where every parameter
    # uses the same lr, Adadelta computes a per-parameter effective learning rate
    # based on the history of gradients and updates for that parameter.
    # lr=1.0 is a global scaling factor on top of the adaptive rate.

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # Learning rate scheduler: after every step_size epochs (=1 here, so every epoch),
    # multiply the learning rate by gamma (=0.7).
    #
    # Epoch 1:  lr = 1.0
    # Epoch 2:  lr = 0.7
    # Epoch 3:  lr = 0.49
    # Epoch 4:  lr = 0.343
    # ...
    # Epoch 14: lr = 0.0068
    #
    # The scheduler wraps the optimizer and modifies its internal lr value.

    for epoch in range(1, args.epochs + 1):
        # range(1, 15) → epochs 1 through 14.

        train(args, model, device, train_loader, optimizer, epoch)
        # One full pass through all 60,000 training images.
        # Weights are updated after each batch of 64 images.
        # That's 938 weight updates per epoch.

        test(model, device, test_loader)
        # Evaluate on all 10,000 test images (no gradient computation).
        # Prints average loss and accuracy. This is how we monitor whether
        # the model is actually learning (and not just memorizing training data).

        scheduler.step()
        # Decay the learning rate: lr *= gamma.
        # Called once per epoch, after both training and testing.

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        # model.state_dict() returns an OrderedDict mapping parameter names to tensors:
        #   {'conv1.weight': tensor(...), 'conv1.bias': tensor(...), ...}
        # torch.save serializes it to disk using Python's pickle format.
        #
        # To reload later:
        #   model = Net()
        #   model.load_state_dict(torch.load("mnist_cnn.pt"))
        #
        # Only saves weights, not the model architecture — you need the Net class
        # definition to reconstruct the model.


if __name__ == '__main__':
    main()
    # Standard Python idiom: only run main() if this file is executed directly
    # (python mnist.py), not if it's imported as a module (import mnist).
    # When imported, __name__ == 'mnist', not '__main__'.