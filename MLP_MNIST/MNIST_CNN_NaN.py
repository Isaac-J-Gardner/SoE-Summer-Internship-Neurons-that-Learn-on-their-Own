import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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
        self.conv1 = nn.Conv2d(1, 20, 3, 1) #20 26x26 images = 13,520 outputs, 3380 after maxpool
        self.conv_decoder_weights = nn.Parameter(torch.randn(20, 3, 3)*0.01)
        self.conv_decoder_biases = nn.Parameter(torch.zeros(20, 3, 3))
        self.fc2 = nn.Linear(3380, 10)

    def forward(self, x):
        features = torch.nn.functional.unfold(x, kernel_size=3, stride=1) #extracts 3x3 patches from the input image, stride of 1 means we move 1 pixel at a time, shape is batch, 9, 676
        features = features.view(x.size(0), 3, 3, 26, 26)
        #reshape patches to match decoder weights (26, 26, 3, 3) for each of the 32 output channels
        features = features.permute(0, 3, 4, 1, 2) #reorder dimensions to (batch_size, height, width, kernel_height, kernel_width) aka batch, 26, 26, 3, 3
        x = self.conv1(x) 
        x = torch.relu(x)
        #unsqueeze output from batch, 20, 26, 26 to batch, 20, 26, 26, 1, 1. unsqueeze decoder weights to 20, 1, 1, 3, 3. broadcasting should handle mismatch, same with bias not having batch
        decoded = x.unsqueeze(-1).unsqueeze(-1) * self.conv_decoder_weights.unsqueeze(1).unsqueeze(1) + self.conv_decoder_biases.unsqueeze(1).unsqueeze(1)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x, decoded, features

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
        task_loss = criterion(output, target)
        recon_loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        loss = task_loss + recon_loss
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

epochs = 10
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

W = model.conv1.weight.detach().cpu().numpy()   
n_filters = W.shape[0]

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i >= n_filters:
        ax.axis('off')
        continue
    filt = W[i, 0]                  
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
    ax.set_title(f'filter {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()