import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math
import random


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
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    total += images.sum(dim=0)   
    n += images.size(0)
mean_image = nn.Flatten()(total / n)        
mean_image = mean_image.to(device)

class NeuronAutoencoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)                 
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01) #each neuron is itself and autoencoder, 784->1->784
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x                                        
        activations = torch.sigmoid(self.encoder(x))                  
        decoded = (activations.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))        
        return decoded, features, activations


model = NeuronAutoencoder().to(device)
print(model)

recon_criterion = nn.MSELoss()
optimizer_recon = torch.optim.SGD(model.parameters(), lr=1)

def retrieve_activations(test_loader, model):
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        _, _, activations = model(data)

    return activations.detach().cpu().numpy()

def calc_effective_rank(cov_matrix): #the input matrix can be the covariance matrix as named here, or the correlation matrix
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues *= 1/sum(eigenvalues)
    eig_entropy = 0
    for val in eigenvalues:
        eig_entropy += -(val*math.log(val))
    r_eff = math.exp(eig_entropy)
    return r_eff

def calc_spectral_redundancy(effective_rank, N): #a measure of redundancy, if all the neurons seem to measure along the same principle component, low effective rank, high redundancy
    return (1-effective_rank/N)

def encoder_row_cosine(model):
    """Mean |cosine| between distinct encoder rows -> redundancy index."""
    W = model.encoder.weight     # (H, in_dim)
    norm = torch.norm(W, dim=1, keepdim=True)
    norm[norm < 1e-12] = 1e-12
    Wn = W / norm
    C = Wn @ Wn.T                                          # (H, H) cosines
    H = C.shape[0]
    off = ~torch.eye(H, dtype=bool)
    return float(torch.abs(C[off]).mean())

def train_recon(data_loader, model, recon_criterion, optimizer):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        decoded, features, _ = model(data)
        

        recon_loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        inhibition_loss = encoder_row_cosine(model)
        loss = recon_loss + 10 * inhibition_loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/num_batches
    print(f"Average loss: {train_loss:7f}")

epochs = 10
for epoch in range(epochs):
    print(f"Recon epoch: {epoch+1}")
    train_recon(train_loader, model, recon_criterion, optimizer_recon)
    activations = retrieve_activations(test_loader, model) #10000x20
    print("effective rank:", calc_effective_rank(np.cov(activations.T))) #transposed gives 20x10000

W = model.encoder.weight.detach().cpu().numpy()   # (20, 784)
W2 = model.decoder_weights.detach().cpu().numpy()
W3 = model.decoder_bias.detach().cpu().numpy()

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

