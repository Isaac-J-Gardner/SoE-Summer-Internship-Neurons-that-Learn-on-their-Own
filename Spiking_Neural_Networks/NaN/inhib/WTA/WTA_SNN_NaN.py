import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import utils as utls

import snntorch as snn
from snntorch import utils
from snntorch import surrogate

import numpy as np

class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        # h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   # (batch, n_neurons, in_dim)

class WTASpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=20, beta=0.9, inhib_strength=1.0):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        # per-neuron adaptive threshold, updated each batch
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # every neuron inhibits every other; none inhibits itself
        W = -inhib_strength * (torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden))
        self.register_buffer("W_inh", W)   # fixed, not learned

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):
            # feedforward drive + inhibition from whoever fired last step
            cur = self.fc(x) + spk @ self.W_inh.t()
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
        return spk_rec, mem_rec   # [T, batch, 20]


#SAE class
class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        #Encoder
        self.encoder = WTASpikingEncoder(784, 20, beta=beta, inhib_strength=2)
        
        self.decoder = nn.Sequential(NeuronDecoder(20, 784),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,output=True,threshold=20000) #make large so membrane can be trained
                          )
    def forward(self, x): 
        utils.reset(self.encoder) #need to reset the hidden states of LIF 
        utils.reset(self.decoder)

        x = x.view(x.size(0), -1)   # [64, 1, 28, 28] -> [64, 784]
        
        #encode
        spk_mem=[];spk_rec=[];encoded_x=[]
        spk_rec,spk_mem=self.encode(x) #Output spike trains and neuron membrane states
        spk_rec=torch.stack(spk_rec,dim=2)
        spk_mem=torch.stack(spk_mem,dim=2) 
        
        #decode
        spk_mem2=[];spk_rec2=[];decoded_x=[]
        for step in range(num_steps): #for t in time
            x_recon,x_mem_recon=self.decode(spk_rec[...,step]) 
            spk_rec2.append(x_recon) 
            spk_mem2.append(x_mem_recon)
        spk_rec2=torch.stack(spk_rec2,dim=3)
        spk_mem2=torch.stack(spk_mem2,dim=3)  
        out = spk_mem2[:,:,:,-1] #return the membrane potential of the output neuron at t = -1 (last t)
        return x, out 

    def encode(self,x):
        spk_latent_x,mem_latent_x=self.encoder(x, num_steps=num_steps) 
        return spk_latent_x,mem_latent_x

    def decode(self,x):
        spk_x2,mem_x2=self.decoder(x)
        return spk_x2,mem_x2

#Training 
def train(network, trainloader, opti, epoch): 
    
    network=network.train()
    train_loss_hist=[]
    for batch_idx, (real_img, labels) in enumerate(trainloader):   
        opti.zero_grad()
        real_img = real_img.to(device)
        labels = labels.to(device)
        
        #Pass data into network, and return reconstructed image from Membrane Potential at t = -1
        x, x_recon = network(real_img) #Dimensions passed in: [Batch_size,Channels,Image_Width,Image_Length] 
        
        #Calculate loss        
        loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))  # x: [64, 784]
                
        print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item()}')

        loss_val.backward()
        opti.step()
    return loss_val

#Testing 
def test(network, testloader, opti, epoch):
    network=network.eval()
    test_loss_hist=[]
    with torch.no_grad(): #no gradient this time
        for batch_idx, (real_img, labels) in enumerate(testloader):   
            real_img = real_img.to(device)#
            labels = labels.to(device)
            x, x_recon = network(real_img) #Dimensions passed in: [Batch_size,Channels,Image_Width,Image_Length] 
                    
            #Calculate loss        
            loss_val = F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon))  # x: [64, 784]

            print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  Loss: {loss_val.item()}')#, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')
    return loss_val

#setup GPU
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


train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# SNN parameters
spike_grad = surrogate.atan(alpha=2.0)# alternate surrogate gradient: fast_sigmoid(slope=25) 
beta = 0.5 #decay rate of neurons 
num_steps=5 #time 
thresh=1#spiking threshold (lower = more spikes are let through)
epochs=1 #number of epochs
max_epoch=epochs

#Define Network and optimizer
net=SAE()
net = net.to(device)

optimizer = torch.optim.SGD(net.parameters(), 
                            lr=10)

#Run training and testing        
for e in range(epochs): 
    train_loss = train(net, train_loader, optimizer, e)
    test_loss = test(net,test_loader,optimizer,e)

W = net.encoder.fc.weight.detach().cpu().numpy()   # (20, 784)
W2 = net.decoder[0].weight.detach().cpu().numpy()
W3 = net.decoder[0].bias.detach().cpu().numpy()

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