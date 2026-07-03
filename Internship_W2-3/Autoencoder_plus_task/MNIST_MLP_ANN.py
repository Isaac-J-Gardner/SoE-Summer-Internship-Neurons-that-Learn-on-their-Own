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
        self.encoder = nn.Linear(784, 20)
        self.decoder = nn.Linear(20, 784)
        self.readout = nn.Linear(20, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x #shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.relu(x)
        decoded = self.decoder(x) #shape = [batch_size, 784]
        output = self.readout(x.detach())
        return decoded, features, output

model = SimpleMLP().to(device)
print(model)
recon_scaler = 13

criterion = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()          

def train(data_loader, model, criterion, optimizer):
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
        decoded, features, output = model(data)
        
        # Calculate the loss
        task_loss = criterion(output, target)
        recon_loss = recon_criterion(decoded, features)
        total_recon_loss += recon_loss
        total_task_loss += task_loss
        loss = task_loss + recon_scaler * recon_loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_task_loss = total_task_loss/num_batches
    train_recon_loss = total_recon_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average task loss: {train_task_loss:7f}, Average recon loss: {train_recon_loss:7f}, accuracy: {accuracy:.2%}")

epochs = 20
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, optimizer)

W = model.encoder.weight.detach().cpu().numpy()   # (20, 784)
W2 = model.decoder.weight.detach().cpu().numpy()
W3 = model.decoder.bias.detach().cpu().numpy()

W2 = np.transpose(W2, (1,0))

encoder_mean = np.mean(abs(W))
decoder_mean = np.mean(abs(W2))


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

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
filt = W3.reshape(28, 28)
im = ax.imshow(filt, cmap='seismic',
               vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
ax.set_title('decoder bias (shared)')
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046)   # optional, but handy for reading the scale
plt.tight_layout()
plt.show()

