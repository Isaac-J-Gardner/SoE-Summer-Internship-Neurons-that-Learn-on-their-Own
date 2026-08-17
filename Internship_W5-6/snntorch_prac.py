import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from snntorch import utils
from snntorch import spikegen
import snntorch.spikeplot as splt
from IPython.display import HTML

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

subset = 10
mnist_train = utils.data_subset(train_dataset, subset) #reduces set by factor subset (with 10, 60000 becomes 6000)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

num_steps = 100
data = iter(train_loader)
data_it, target_it = next(data)

spike_data = spikegen.rate(data_it, num_steps=num_steps)
print(spike_data.size())