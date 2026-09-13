import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.') 
    device = torch.device('cpu')
    
batch_size = 100 #batch size 100 matches SNNs

def fit_zca(X, eps):
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)                
    evals = torch.clamp(evals, min=0.0)
    W = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
    return mean, W

@torch.no_grad()
def compute_zca(dataset, eps):
    # one big (N, 784) matrix; MNIST is ~188 MB in float32, fine on CPU
    X = torch.stack([img.view(-1) for img, _ in dataset])   # (60000, 784)
    return fit_zca(X, eps)                                   # mean:(1,784)  W:(784,784)

base = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mean, W = compute_zca(base, eps=1e-2)

class ZCAWhiten:
    def __init__(self, mean, W):
        self.mean = mean          # (1, 784), CPU
        self.W = W                # (784, 784), CPU
    def __call__(self, x):        # x: (1, 28, 28)
        flat = x.reshape(1, -1)             # (1, 784)
        white = (flat - self.mean) @ self.W # (1, 784)
        return white.reshape(x.shape)       # (1, 28, 28)

tfm = transforms.Compose([transforms.ToTensor(), ZCAWhiten(mean, W)])

train_loader = DataLoader(datasets.MNIST('./data', train=True,  download=True, transform=tfm),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=tfm),
                          batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28*28, 20)
        self.decoder = nn.Linear(20, 784)
        self.readout = nn.Linear(20, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x #shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.relu(x)
        decoded = None
        if self.training:
            decoded = self.decoder(x) #shape = [batch_size, 784]
        x = self.readout(x.detach())
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

def train(data_loader, model, criterion, optimizer):
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
        recon_loss = recon_criterion(decoded, features)
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
    train(train_loader, model, criterion, optimizer)

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
W2 = model.decoder.weight.detach().cpu().numpy()
W3 = model.decoder.bias.detach().cpu().numpy()

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    filt = W[i].reshape(28, 28)
    ax.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
    ax.set_title(f'neuron {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.close()


