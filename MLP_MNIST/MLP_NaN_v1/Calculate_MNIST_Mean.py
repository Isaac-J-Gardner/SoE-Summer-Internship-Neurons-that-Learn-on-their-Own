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

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    total += images.sum(dim=0)   # sum over the batch
    n += images.size(0)
mean_image = (total / n)        # shape [1, 28, 28]


fig, axes = plt.subplots(1, 1, figsize=(10, 8))
filt = mean_image.reshape(28, 28)
axes.imshow(filt, cmap='seismic',
              vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())  # symmetric colormap centered at 0
axes.set_title(f'MNIST Mean')
axes.axis('off')
plt.tight_layout()
plt.show()