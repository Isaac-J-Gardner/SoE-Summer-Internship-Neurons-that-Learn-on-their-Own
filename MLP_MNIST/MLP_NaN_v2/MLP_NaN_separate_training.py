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
    
batch_size = 8

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

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    total += images.sum(dim=0)   # sum over the batch
    n += images.size(0)
mean_image = nn.Flatten()(total / n)        # shape [1, 28, 28]
mean_image = mean_image.to(device)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28*28, 20)
        self.decoder_weights = nn.Parameter(torch.randn(20, 784) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(20, 784))
        self.readout = nn.Linear(20, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x -= mean_image
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

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()          

def train(data_loader, model, criterion, recon_criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
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
        else:
            loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

epochs = 20
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, recon_criterion, optimizer)

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

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()