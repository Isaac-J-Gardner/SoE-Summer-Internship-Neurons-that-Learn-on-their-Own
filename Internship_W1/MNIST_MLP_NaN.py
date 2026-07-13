import os
import random
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

train_loader, test_loader, _, _ = zca_whiten_mnist()

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28*28, 20)
        self.decoder_weights = nn.Parameter(torch.randn(20, 784) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(20, 784))
        self.readout = nn.Linear(20, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x #shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.sigmoid(x)
        decoded = None
        if self.training:
            decoded = x.unsqueeze(2) * self.decoder_weights.unsqueeze(0) + self.decoder_bias.unsqueeze(0) #shape = [batch_size, 20, 784]
        output = self.readout(x.detach())
        return output, decoded, features


model = SimpleMLP().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
recon_optimizer = torch.optim.SGD(model.parameters(), lr= 10)

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()          

def train(data_loader, model, criterion, recon_criterion, optimizer, recon_optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_task_loss = 0
    total_recon_loss = 0
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
            total_task_loss += loss
                # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
            total_recon_loss += loss
                # Backpropagation
            recon_optimizer.zero_grad()
            loss.backward()
            recon_optimizer.step()
        

        # Count number of correct digits
        total_correct += correct(output, target)
        
    train_task_loss = total_task_loss/num_batches
    train_recon_loss = total_recon_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_task_loss:7f}, Average loss: {train_recon_loss:7f}, accuracy: {accuracy:.2%}")

epochs = 20
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, recon_criterion, optimizer, recon_optimizer)

def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)
        
            # Do a forward pass
            output, _, _ = model(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")

test(test_loader, model, criterion)

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