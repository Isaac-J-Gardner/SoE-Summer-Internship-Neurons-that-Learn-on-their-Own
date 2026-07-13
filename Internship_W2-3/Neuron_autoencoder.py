import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

def compute_zca_matrix(X, eps=1e-5):
    """
    Compute the ZCA whitening matrix and mean from data.

    Args:
        X:   (N, D) tensor of flattened samples (float, ideally in [0, 1]).
        eps: regularization added to eigenvalues to avoid blow-up on
             near-zero variance directions.

    Returns:
        zca:  (D, D) whitening matrix.
        mean: (1, D) per-feature mean used for centering.
    """
    X = X.to(torch.float64)                 # float64 for a stable eigendecomp
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    # Covariance matrix (D, D)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)

    # Symmetric eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # W = U diag(1/sqrt(lambda + eps)) U^T
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + eps))
    zca = eigvecs @ inv_sqrt @ eigvecs.T

    return zca.to(torch.float32), mean.to(torch.float32)


def apply_zca(X, zca, mean):
    """Apply a precomputed ZCA transform. X is (N, D)."""
    return (X - mean) @ zca.T


def zca_whiten_mnist(data_root="./data", eps=1e-5, batch_size=8):
    """
    Load MNIST, fit ZCA on the training set, and return whitened
    train/test DataLoaders. The transform is fit on train only and
    applied to both, which is the correct way to avoid leakage.
    """
    to_tensor = transforms.ToTensor()  # gives (1, 28, 28) in [0, 1]

    train = datasets.MNIST(data_root, train=True,  download=True, transform=to_tensor)
    test  = datasets.MNIST(data_root, train=False, download=True, transform=to_tensor)

    # Stack into flat (N, 784) tensors
    X_train = train.data.float().div(255.0).view(len(train), -1)
    X_test  = test.data.float().div(255.0).view(len(test), -1)
    y_train, y_test = train.targets, test.targets

    # Fit on train, apply to both
    zca, mean = compute_zca_matrix(X_train, eps=eps)
    X_train_w = apply_zca(X_train, zca, mean)
    X_test_w  = apply_zca(X_test,  zca, mean)

    # Reshape back to images if your model expects (N, 1, 28, 28)
    X_train_w = X_train_w.view(-1, 1, 28, 28)
    X_test_w  = X_test_w.view(-1, 1, 28, 28)

    train_loader = DataLoader(TensorDataset(X_train_w, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_w, y_test),
                              batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, zca, mean

train_loader, _, _, _ = zca_whiten_mnist()

class NeuronAutoencoder(nn.Module):
    """Each of the n_neurons is its own autoencoder: its scalar activation is
    the latent variable, from which it reconstructs the FULL input via its own
    decoder weight vector and bias image."""
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)                       # (20, 784) weight
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x                                        # [batch, 784]
        h = torch.sigmoid(self.encoder(x))                  # [batch, 20]  one latent per neuron
        # neuron i reconstructs the input as  h_i * decoder_weights[i] + decoder_bias[i]
        decoded = (h.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))        # [batch, 20, 784]
        return decoded, features


model = NeuronAutoencoder().to(device)
print(model)

recon_criterion = nn.MSELoss()
optimizer_recon = torch.optim.SGD(model.parameters(), lr=10)

def train_recon(data_loader, model, recon_criterion, optimizer):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        decoded, features = model(data)
        

        loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/num_batches
    print(f"Average loss: {train_loss:7f}")

epochs = 25
for epoch in range(epochs):
    print(f"Recon epoch: {epoch+1}")
    train_recon(train_loader, model, recon_criterion, optimizer_recon)

W = model.encoder.weight.detach().cpu().numpy()   # (20, 784)
W2 = model.decoder_weights.detach().cpu().numpy()
W3 = model.decoder_bias.detach().cpu().numpy()

encoder_mean = np.mean(abs(W))
decoder_mean = np.mean(abs(W2))
bias_mean = np.mean(abs(W3))
print(encoder_mean)

print(decoder_mean)
print(bias_mean)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W2[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W3[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()