import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import math


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
        decoded, features, _ = model(data)
        

        loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/num_batches
    print(f"Average loss: {train_loss:7f}")

def retrieve_activations(test_loader, model):
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        _, _, activations = model(data)

    return activations.detach().cpu().numpy()

def calc_effective_rank(activations):
    covariance = np.cov(activations) #shape should be N_neuronsXnum_activations e.g. 20x10000
    correlation = np.corrcoef(activations)
    eigenvalues_cov = np.linalg.eigvals(covariance)
    eigenvalues_corr = np.linalg.eigvals(correlation)
    eigenvalues_cov *= 1/sum(eigenvalues_cov)
    eigenvalues_corr *= 1/sum(eigenvalues_corr)
    cov_entropy = 0
    corr_entropy = 0
    for val in eigenvalues_cov:
        cov_entropy += -(val*math.log(val))
        corr_entropy += -(val*math.log(val))
    r_eff_cov = math.exp(cov_entropy)
    r_eff_corr = math.exp(corr_entropy)
    return r_eff_cov, r_eff_corr



epochs = 2
for epoch in range(epochs):
    print(f"Recon epoch: {epoch+1}")
    train_recon(train_loader, model, recon_criterion, optimizer_recon)
    activations = retrieve_activations(test_loader, model) #10000x20
    r_eff_cov, r_eff_corr = calc_effective_rank(activations.T)


