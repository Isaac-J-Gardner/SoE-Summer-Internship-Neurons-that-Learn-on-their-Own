import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import utils
from snntorch import surrogate


EIG_FLOOR = 1e-12 #Used when calculating effective rank, removes zeros to prevent NaNs

batch_size  = 100

epochs      = 20

N = 20 #number of hidden neurons (neuron autoencoders)

num_steps   = 50     #5 time constants, each image is shown for 50 time steps
theta      = 2.0     #starting threshold for each hidden neuron (readout threshold are set really high, potential is used to infer recon)
beta = 0.9           #membrane decay rate, produces time constant of ~10 time steps

alpha = 5 #mutual inhibition learning rate
learning_beta = 0.05 #task loss optimiser learning rate
gamma = 0.5 #adaptive threshold learning rate
p = 0.05 #desired firing rate (per time step, neurons should fire 2.5 times per image)

GATING = True  

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

data_dir = './data'
print('data_dir =', data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

class NeuronDecoder(nn.Module):
    def __init__(self, n_neurons=N, in_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, h):                        #h: (batch, n_neurons)
        return h.unsqueeze(-1) * self.weight + self.bias   #(batch, n_neurons, in_dim)

sur_grad = surrogate.atan(alpha = 2.0)
class SpikingEncoder(nn.Module):
    def __init__(self, n_in=784, n_hidden=N, beta=beta):
        super().__init__()
        self.n_hidden = n_hidden
        self.fc  = nn.Linear(n_in, n_hidden)
        self.lif = snn.Leaky(beta=beta, spike_grad=sur_grad, threshold=theta)

        W = torch.zeros(n_hidden, n_hidden) #recurrent inhibitory weight matrix
        self.register_buffer("W_inh", W)

        self.register_buffer("theta", torch.zeros(n_hidden)) #per neuron threshold matrix, this threshold is added onto theta.

    def forward(self, x, num_steps):
        mem = self.lif.init_leaky()
        spk = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):

            cur = self.fc(x) - (spk @ self.W_inh.t()).detach() - self.theta.detach()
            spk, mem = self.lif(cur, mem)

            spk_rec.append(spk)
            mem_rec.append(mem)

        return spk_rec, mem_rec

    @torch.no_grad()
    def update_inhibition(self, activity, alpha): #SAILnet Mutual inhibition rule
        coinc = (activity.T @ activity) / activity.shape[0]   # <n_i n_m>
        dW = coinc - (p*num_steps)**2                               # n_i n_m - p^2
        dW.fill_diagonal_(0)
        self.W_inh.add_(alpha * dW)
        self.W_inh.clamp_(min=0.0)

    @torch.no_grad()
    def update_threshold(self, spk_rec, gamma): #SAILnet thresholding rule
        S_flat = torch.cat(spk_rec, dim=0)
        rate = S_flat.mean(0)
        dtheta = gamma*(rate-p)
        self.theta.add_(dtheta)

class SAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpikingEncoder(784, N)
        self.decoder = nn.Sequential(
            NeuronDecoder(N, 784),
            snn.Leaky(beta=0.9, spike_grad=sur_grad, init_hidden=True,
                      output=True, threshold=20000)   #The decoder neurons never spike, so act as integrator
        )

    def forward(self, x):
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        x = x.view(x.size(0), -1)                    #(B,784) from (B, 1, 28, 28)

        # encode
        spk_rec_list, _ = self.encode(x) #array of spiking activity for each timestep, (T, B, H), useful for later
        spk_rec = torch.stack(spk_rec_list, dim=2)        # (B, H, T) #stacked into torch tensor

        # NEW: per-(sample, neuron) activity = total spikes over time -> the gate
        activity = spk_rec.sum(dim=2)                # (B, H) #total spikes per neuron per batch

        # decode: integrate each neuron's reconstruction over time
        spk_mem2 = []
        for step in range(num_steps):
            _, x_mem_recon = self.decode(spk_rec[..., step])
            spk_mem2.append(x_mem_recon)
        out = torch.stack(spk_mem2, dim=3)[:, :, :, -1]   # membrane at last step (B,H,784)
        return spk_rec_list, x, out, activity #spk_rec_list is returned, again (T, B, H)

    def encode(self, x):
        return self.encoder(x, num_steps=num_steps)

    def decode(self, x):
        return self.decoder(x)

def recon_loss(x_recon, x, activity):
    target = x.unsqueeze(1).expand_as(x_recon)            #(B,H,784)
    se = ((x_recon - target) ** 2).mean(dim=2)            #(B,H) per neuron-sample
    if GATING:
        gate = (activity.detach() > 0).float()            #(B,H) 1 if neuron fired
        return (se * gate).sum() / gate.sum().clamp(min=1) #averaging gated squared error
    return se.mean()


def train(network, loader, opti, epoch):
    network.train()
    for i, (img, _) in enumerate(loader):
        opti.zero_grad()
        img = img.to(device)
        spk_rec_list, x, x_recon, activity = network(img)
        loss = recon_loss(x_recon, x, activity)
        loss.backward()
        opti.step()
        network.encoder.update_inhibition(activity, alpha) #actibity used for inhibition
        network.encoder.update_threshold(spk_rec_list, gamma) #T, B, H for threshold. I could have used activity, but I did this first and
                                                              # it kept breaking for some reasong when I tried, might sort it later
        if i % 50 == 0: #giving regular updates of progress as 1 epoch takes a while
            print(f'Train[{epoch+1}/{epochs}][{i}/{len(loader)}] Loss: {loss.item():.5f} ')


    return loss

def spk_effective_rank(spk_rec, n_neurons=N, eig_floor=EIG_FLOOR, jitter=1e-6):
    N = n_neurons
    spikes = spk_rec.permute(1, 0, 2).reshape(N, -1)      # (N, batch*num_steps)

    cov = torch.cov(spikes)                               # (N, N)
    
    cov = cov + jitter * torch.eye(N, device=cov.device, dtype=cov.dtype)

    eigvals = torch.linalg.eigvalsh(cov)                  
    eigvals = torch.clip(eigvals, 0.0, None)
    total = eigvals.sum()

    #If nothing spiked in the whole batch, return a graph-connected zero
    #(rather than nan)
    if total <= eig_floor:
        return spikes.sum() * 0.0

    p = eigvals / total
    p = p[p > eig_floor]
    entropy = torch.sum(-(p * torch.log(p)))
    r_eff = torch.exp(entropy)                            #effective rank
    return r_eff      

@torch.no_grad()
def test_encoder(network, loader):
    network.eval()
    losses = []
    spikes = []
    for img, _ in loader:
        img = img.to(device)
        spk_rec, x, x_recon, activity = network(img)
        spikes.append(torch.cat(spk_rec, dim=0))
    spikes = torch.cat(spikes, dim=0)
    r_eff = spk_effective_rank(spikes.unsqueeze(-1))
    avg_rate = spikes.float().mean()
    thresholds = network.encoder.theta
    inhibition = network.encoder.W_inh
    avg_thresh = thresholds.mean()
    inhibition_prop = (inhibition > 0).float().mean() #proportion of existing inhibition weights over possible inhibition weights
    inhibition_prop = inhibition_prop * 20/19 #scaling to account for zeroed diagonal
    return r_eff.item(), avg_rate.item(), avg_thresh.item(), inhibition_prop.item()

class LinearReadout(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc1 = nn.Linear(784, N)
        self.fc1.weight = nn.Parameter(weight)
        self.fc2 = nn.Linear(N, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x.detach())
        return x

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()

def train_linear(data_loader, model, criterion, optimizer):
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
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)
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
    return train_loss

def test_linear(test_loader, model, criterion):
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
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return accuracy

for seed in range(1):
  torch.manual_seed(seed)

  net = SAE().to(device)
  optimizer = torch.optim.SGD(net.parameters(), lr=learning_beta)
  test_epochs = []
  accuracy = []
  r_effs = []
  avg_rates =[]
  avg_threshs = []
  inhibition_props = []
  for e in range(epochs+1):
      if (e)%4 == 0:
          test_epochs.append(e)
          r_eff, avg_rate, avg_thresh, inhibition_prop = test_encoder(net, test_loader)
          r_effs.append(r_eff)
          avg_rates.append(avg_rate)
          avg_threshs.append(avg_thresh)
          inhibition_props.append(inhibition_prop)

          W  = net.encoder.fc.weight.detach().cpu().numpy()
          W2 = net.decoder[0].weight.detach().cpu().numpy()
          W3 = net.decoder[0].bias.detach().cpu().numpy()
          for title, mat in [("encoder", W), ("decoder", W2), ("decoder_bias", W3)]:
              fig, axes = plt.subplots(4, 5, figsize=(10, 8)); fig.suptitle(title)
              for i, ax in enumerate(axes.flat):
                  f = mat[i].reshape(28, 28); m = np.abs(f).max() + 1e-9
                  ax.imshow(f, cmap='seismic', vmin=-m, vmax=m); ax.set_title(f'neuron {i}'); ax.axis('off')
              plt.tight_layout()
              plt.show()
              plt.close()

          W4 = net.encoder.W_inh.detach().cpu().numpy()
          fig, ax = plt.subplots()
          fig.suptitle("Inhibition Weights")
          m = np.abs(W4).max() + 1e-9
          im = ax.imshow(W4, cmap='seismic', vmin=-m, vmax=m)
          fig.colorbar(im, ax=ax)
          ax.set_xlabel("source neuron j")
          ax.set_ylabel("target neuron i")
          plt.tight_layout()
          plt.show()
          plt.close(fig)

          #each readout training should start with the same seed
          torch.manual_seed(seed+100)

          readout = LinearReadout(net.encoder.fc.weight).to(device)

          criterion = nn.CrossEntropyLoss()
          optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
          previous_loss = 1
          current_loss = 0
          i = 0
          while ((i < 3) or ((previous_loss - current_loss) > 0.01)):
              previous_loss = current_loss
              print(f"Training epoch: {i+1}")
              current_loss = train_linear(train_loader, readout, criterion, optimizer)
              print(f"Loss Diff: {previous_loss - current_loss}")
              i += 1
          accuracy.append(test_linear(test_loader, readout, criterion))

      train(net, train_loader, optimizer, e)

plt.figure()
plt.plot(test_epochs, accuracy)
plt.title("Readout Accuracy across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Test Set Accuracy')
plt.show()
plt.close()

plt.figure()
plt.plot(test_epochs, r_effs)
plt.title("Encoder Activation Effective Rank across Epochs")
plt.xlabel('Epoch')
plt.ylabel('R_eff')
plt.show()
plt.close()

plt.figure()
plt.plot(test_epochs, avg_rates)
plt.title("Average Encoder Firing Rate across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Firing Rate')
plt.show()
plt.close()

plt.figure()
plt.plot(test_epochs, avg_threshs)
plt.title("Average Encoder Neuron Threshold (p=0.05) across Epochs")
plt.xlabel('Epoch')
plt.ylabel('Average Threshold')
plt.show()
plt.close()

plt.figure()
plt.plot(test_epochs, inhibition_props)
plt.title("Proportiong of W_ing > 0 across Epochs")
plt.xlabel('Epoch')
plt.ylabel('W_ing > 0')
plt.show()
plt.close()