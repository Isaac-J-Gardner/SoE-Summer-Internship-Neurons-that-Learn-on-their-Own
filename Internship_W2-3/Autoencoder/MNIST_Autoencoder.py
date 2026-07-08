import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

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

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28*28, 20) #standard autoencoder, 20 hidden neurons, all neurons fully connected to an output the same shape as the input.
        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x #shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.relu(x)
        decoded = self.decoder(x) #shape = [batch_size, 784]
        return decoded, features

model = SimpleMLP().to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=50) #trying to keep the learning as simple as possible means I can better attribute results to a more general case.

def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)

    total_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        decoded, features = model(data)
        
        loss = criterion(decoded, features)

        total_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    train_loss = total_loss/num_batches
    print(f"Average loss: {train_loss:7f}")

epochs = 50
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, optimizer)

W = model.encoder.weight.detach().cpu().numpy()   #(20, 784)
W2 = model.decoder.weight.detach().cpu().numpy() #(784, 20)
W3 = model.decoder.bias.detach().cpu().numpy() #(784)

W2 = np.transpose(W2, (1,0)) #(20, 784) means I can display it the same way I do the encoder weights, gives me a clear picture of what the hidden neurons are contributing to the recreation, this is generally the same pattern as the encoder weights

encoder_mean = np.mean(abs(W)) #usefull to check things are actually changing, and whether they move towards the MNIST mean
decoder_mean = np.mean(abs(W2))


fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  #symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W2[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
filt = W3.reshape(28, 28)
im = ax.imshow(filt, cmap='seismic',
               vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
ax.set_title('decoder bias (shared)')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046)   #handy for reading the scale
plt.tight_layout()
plt.show()

