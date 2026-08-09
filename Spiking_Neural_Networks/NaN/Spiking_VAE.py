import os

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

#SAE class
class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3,padding = 1,stride=2),
                          nn.BatchNorm2d(32),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.Conv2d(32, 64, 3,padding = 1,stride=2),
                          nn.BatchNorm2d(64),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.Conv2d(64, 128, 3,padding = 1,stride=2),
                          nn.BatchNorm2d(128),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.Flatten(start_dim = 1, end_dim = 3),
                          nn.Linear(2048, latent_dim), #this needs to be the final layer output size (channels * pixels * pixels)
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh)
                          )
       # From latent back to tensor for convolution
        self.latentToConv= nn.Sequential(nn.Linear(latent_dim,128*4*4),
                               snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=thresh))        #Decoder
        
        self.decoder = nn.Sequential(nn.Unflatten(1,(128,4,4)), 
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.ConvTranspose2d(128, 64, 3,padding = 1,stride=(2,2),output_padding=1),
                          nn.BatchNorm2d(64),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.ConvTranspose2d(64, 32, 3,padding = 1,stride=(2,2),output_padding=1),
                          nn.BatchNorm2d(32),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh),
                          nn.ConvTranspose2d(32, 1, 3,padding = 1,stride=(2,2),output_padding=1),
                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,output=True,threshold=20000) #make large so membrane can be trained
                          )
    def forward(self, x): 
        utils.reset(self.encoder) #need to reset the hidden states of LIF 
        utils.reset(self.decoder)
        utils.reset(self.linearNet) 
        
        #encode
        spk_mem=[];spk_rec=[];encoded_x=[]
        for step in range(num_steps): #for t in time
            spk_x,mem_x=self.encode(x) #Output spike trains and neuron membrane states
            spk_rec.append(spk_x) 
            spk_mem.append(mem_x)
        spk_rec=torch.stack(spk_rec,dim=2)
        spk_mem=torch.stack(spk_mem,dim=2) 
        
        #decode
        spk_mem2=[];spk_rec2=[];decoded_x=[]
        for step in range(num_steps): #for t in time
            x_recon,x_mem_recon=self.decode(spk_rec[...,step]) 
            spk_rec2.append(x_recon) 
            spk_mem2.append(x_mem_recon)
        spk_rec2=torch.stack(spk_rec2,dim=4)
        spk_mem2=torch.stack(spk_mem2,dim=4)  
        out = spk_mem2[:,:,:,:,-1] #return the membrane potential of the output neuron at t = -1 (last t)
        return out 

    def encode(self,x):
        spk_latent_x,mem_latent_x=self.encoder(x) 
        return spk_latent_x,mem_latent_x

    def decode(self,x):
        spk_x,mem_x = self.linearNet(x) #convert latent dimension back to total size of features in encoder final layer
        spk_x2,mem_x2=self.decoder(spk_x)
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
        x_recon = network(real_img) #Dimensions passed in: [Batch_size,Channels,Image_Width,Image_Length] 
        
        #Calculate loss        
        loss_val = F.mse_loss(x_recon, real_img)
                
        print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item()}')

        loss_val.backward()
        opti.step()

        #Save reconstructed images every at the end of the epoch
        if batch_idx == len(trainloader)-1:
            # NOTE: you need to create training/ and testing/ folders in your chosen path
            utls.save_image((real_img+1)/2, f'training/epoch{epoch}_finalbatch_inputs.png') 
            utls.save_image((x_recon+1)/2, f'training/epoch{epoch}_finalbatch_recon.png')
    return loss_val

#Testing 
def test(network, testloader, opti, epoch):
    network=network.eval()
    test_loss_hist=[]
    with torch.no_grad(): #no gradient this time
        for batch_idx, (real_img, labels) in enumerate(testloader):   
            real_img = real_img.to(device)#
            labels = labels.to(device)
            x_recon = network(real_img)

            loss_val = F.mse_loss(x_recon, real_img)

            print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  Loss: {loss_val.item()}')#, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')
            
            if batch_idx == len(testloader)-1:
                utls.save_image((real_img+1)/2, f'testing/epoch{epoch}_finalbatch_inputs.png')
                utls.save_image((x_recon+1)/2, f'testing/epoch{epoch}_finalbatch_recons.png')
    return loss_val

# Parameters and Run training and testing
batch_size = 250
input_size = 32 #size of input to first convolutional layer

#setup GPU
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load MNIST
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size, input_size)),transforms.Normalize((0,), (1,))]), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((input_size, input_size)),transforms.Normalize((0,), (1,))]), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# SNN parameters
spike_grad = surrogate.atan(alpha=2.0)# alternate surrogate gradient: fast_sigmoid(slope=25) 
beta = 0.5 #decay rate of neurons 
num_steps=5 #time 
latent_dim = 32 #dimension of latent layer (how compressed we want the information)
thresh=1#spiking threshold (lower = more spikes are let through)
epochs=10 #number of epochs
max_epoch=epochs

#Define Network and optimizer
net=SAE()
net = net.to(device)

optimizer = torch.optim.AdamW(net.parameters(), 
                            lr=0.0001,
                            betas=(0.9, 0.999), 
                            weight_decay=0.001)

#Run training and testing        
for e in range(epochs): 
    train_loss = train(net, train_loader, optimizer, e)
    test_loss = test(net,test_loader,optimizer,e)