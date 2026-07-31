import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

EIG_FLOOR = 1e-12
N_hid_neurons = 100

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
        self.encoder = nn.Linear(28*28, N_hid_neurons)
        self.decoder_weights = nn.Parameter(torch.randn(N_hid_neurons, 784) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(N_hid_neurons, 784))
        self.readout = nn.Linear(N_hid_neurons, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x #shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.sigmoid(x)
        activations = x
        decoded = None
        if self.training:
            decoded = x.unsqueeze(2) * self.decoder_weights.unsqueeze(0) + self.decoder_bias.unsqueeze(0) #shape = [batch_size, 20, 784]
        output = self.readout(x.detach())
        return output, decoded, features, activations




def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()     

def effective_rank(mat):
    cov = torch.cov(mat)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clip(eigenvalues, 0.0, None)
    total = torch.sum(eigenvalues)
    if total <= EIG_FLOOR:
        return float("nan")
    p = eigenvalues/total
    p = p[p>EIG_FLOOR]
    entropy = torch.sum(-(p*torch.log(p)))
    r_eff = torch.exp(entropy)
    return r_eff

def lat_inhib_loss(mat):
    cov = torch.cov(mat)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clip(eigenvalues, 0.0, None)
    total = torch.sum(eigenvalues)
    if total <= EIG_FLOOR:
        return float("nan")
    p = eigenvalues/total
    p = p[p>EIG_FLOOR]
    entropy = torch.sum(-(p*torch.log(p)))
    r_eff = torch.exp(entropy)
    r_spec_loss = 1-r_eff/N_hid_neurons
    return r_spec_loss  

def train_recon(data_loader, model, recon_criterion, optimizer, inhib_scaler):
    model.train()
    num_batches = len(data_loader)
    total_recon = 0
    total_inhib = 0
    total_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        _, decoded, features, activations = model(data)
        

        recon_loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        inhib_loss = lat_inhib_loss(activations.T)
        total_recon += recon_loss.item()
        total_inhib += inhib_loss
        loss = recon_loss + inhib_scaler*(inhib_loss)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/num_batches
    recon_avg = total_recon/num_batches
    inhib_avg = total_inhib/num_batches
    print(f"Average loss: {train_loss:5f}, Average Recon: {recon_avg:5f}, Average Inhibition: {inhib_avg:5f}")
    return activations
        

def train_task(data_loader, model, criterion, optimizer):
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
        output, _, _, _ = model(data)
        

        loss = criterion(output, target)
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
    return train_loss

#inhib_scalers = np.logspace(-3, -1, 10)

def train_both(data_loader, model, criterion, recon_criterion, optimiser_recon, optimiser_task, inhib_scaler):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_correct = 0
    total_recon = 0
    total_inhib = 0
    total_hidden_loss = 0
    total_task_loss = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        output, decoded, features, activations = model(data)
        

        recon_loss = recon_criterion(decoded, features.unsqueeze(1).expand_as(decoded))
        inhib_loss = lat_inhib_loss(activations.T)
        total_recon += recon_loss.item()
        total_inhib += inhib_loss
        hidden_loss = recon_loss + inhib_scaler*(inhib_loss)
        total_hidden_loss += hidden_loss.item()
        optimiser_recon.zero_grad()
        hidden_loss.backward()
        optimiser_recon.step()

        task_loss = criterion(output, target)
        total_task_loss += task_loss.item()

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        optimiser_task.zero_grad()
        task_loss.backward()
        optimiser_task.step()
    
    hidden_loss = total_hidden_loss/num_batches
    recon_avg = total_recon/num_batches
    inhib_avg = total_inhib/num_batches
    task_loss = total_task_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average Hidden Loss: {hidden_loss:5f}, Average Recon: {recon_avg:5f}, Average Inhibition: {inhib_avg:5f}\nAverage Task Loss: {task_loss:5f}, Accuracy: {accuracy:4f}")
    return activations

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
            output, _, _, _ = model(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return accuracy

seeds = [0]
#accuracy_results = np.zeros_like(inhib_scalers)
#effective_rank_vals = np.zeros_like(inhib_scalers)
epochs = 10
for seed in seeds:
    torch.manual_seed(seed)
    #for i, scaler in enumerate(inhib_scalers):
    model = SimpleMLP().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    optimizer_recon = torch.optim.SGD(model.parameters(), lr=10)
    optimizer_task = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        print(f"task epoch: {epoch+1}")
        train_recon(train_loader, model, recon_criterion, optimizer_recon, 10)
        #accuracy_results[i] += test(test_loader, model, criterion)
        #effective_rank_vals[i] += effective_rank(activations.T)
    for epoch in range(epochs):
        train_task(train_loader, model, criterion, optimizer_task)

test(test_loader, model, criterion)

#accuracy_results /= len(seeds)
#effective_rank_vals /= len(seeds)

fig, ax = plt.subplots()
#ax.semilogx(inhib_scalers, accuracy_results)
ax.set(xlabel="inhib scaler", ylabel="accuracy")
ax.grid()
plt.show()

fig, ax = plt.subplots()
#ax.semilogx(inhib_scalers, effective_rank_vals)
ax.set(xlabel="inhib scaler", ylabel="effective rank")
ax.grid()
plt.show()

fig, ax = plt.subplots()
#ax.plot(effective_rank_vals, accuracy_results)
ax.set(xlabel="effective rank", ylabel="accuracy")
ax.grid()
plt.show()