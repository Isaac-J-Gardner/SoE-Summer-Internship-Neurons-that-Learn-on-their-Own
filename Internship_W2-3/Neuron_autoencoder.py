import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

EIG_FLOOR = 1e-12
N_hid_neurons = 100

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
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

class NeuronAutoencoder(nn.Module):
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

epochs = 20
for epoch in range(epochs):
    print(f"Recon epoch: {epoch+1}")
    train_recon(train_loader, model, recon_criterion, optimizer_recon)

