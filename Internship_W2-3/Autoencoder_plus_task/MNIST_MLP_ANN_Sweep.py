import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

batch_size = 64
epochs = 5

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
        output = self.readout(x)
        return decoded, features, output

model = SimpleMLP().to(device)
print(model)

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()          

def train(data_loader, model, task_criterion, recon_criterion, optimizer, recon_scaler):
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
        task_loss = task_criterion(output, target)
        recon_loss = recon_criterion(decoded, features)
        total_recon_loss += recon_loss.item()
        total_task_loss += task_loss.item()
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
    return train_task_loss, train_recon_loss, accuracy


def save_weight_grid(weight, path, title_prefix):
    """Save a 4x5 grid of 28x28 filters (matches the original plotting)."""
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        filt = weight[i].reshape(28, 28)
        ax.imshow(filt, cmap='seismic',
                  vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
        ax.set_title(f'{title_prefix} {i}')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_bias(bias, path):
    """Save the shared decoder bias as a single 28x28 image."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    filt = bias.reshape(28, 28)
    im = ax.imshow(filt, cmap='seismic',
                   vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
    ax.set_title('decoder bias (shared)')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


#recon scaler sweep
recon_scaler_vals = np.logspace(-2, 2.5, 2)   #0.01 to 100 (Recon Scaler is 0.1, so previous best recon Recon Scaler without task loss (1.29) would be scaler = 12.9)
final_task_losses = []
final_recon_losses = []
final_accuracy = []
enc_weight_mag = []   # mean(|encoder weight|) per rs
dec_weight_mag = []   # mean(|decoder weight|) per rs
bias_mag = []         # mean(|decoder bias|)   per rs

os.makedirs('sweep', exist_ok=True)

for recon_scaler in recon_scaler_vals:
    task_losses = []
    recon_losses = []
    accuracies = []
    scaler_name = f'scaler_{recon_scaler:.3g}'               
    scaler_dir = os.path.join('sweep', scaler_name)
    os.makedirs(scaler_dir, exist_ok=True)

    print(f'\n=== Training {scaler_name}  (recon scaler={recon_scaler:.4g}) ===')

    # Same initialisation for every run so the weights/bias are directly
    # comparable across Recon Scalers. Remove this line for random inits.
    torch.manual_seed(0)

    model = SimpleMLP().to(device)
    task_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    last_task_loss = float('nan')
    last_recon_loss = float('nan')
    last_accuracy = float('nan')
    for epoch in range(epochs):
        last_task_loss, last_recon_loss, last_accuracy = train(train_loader, model, task_criterion, recon_criterion, optimizer, recon_scaler)
        task_losses.append(last_task_loss)
        recon_losses.append(last_recon_loss)
        accuracies.append(last_accuracy)
        print(f'  epoch {epoch + 1:2d}/{epochs}  loss={last_task_loss:.6f}')

    final_task_losses.append(last_task_loss)
    final_recon_losses.append(last_recon_loss)
    final_accuracy.append(last_accuracy)

    # Pull out the trained parameters (same layout as the original code)
    W = model.encoder.weight.detach().cpu().numpy()      # (20, 784)
    W2 = model.decoder.weight.detach().cpu().numpy()     # (784, 20)
    W3 = model.decoder.bias.detach().cpu().numpy()       # (784,)
    W2 = np.transpose(W2, (1, 0))                         # (20, 784)

    save_weight_grid(W, os.path.join(scaler_dir, 'encoder_weight.png'), 'neuron')
    save_weight_grid(W2, os.path.join(scaler_dir, 'decoder_weight.png'), 'neuron')
    save_bias(W3, os.path.join(scaler_dir, 'decoder_bias.png'))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, epochs+1), task_losses, 'o-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('task training loss (Cross Entropy)')  
    ax.set_title('task loss across epochs')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(scaler_dir, 'task_loss_vs_epochs.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, epochs+1), recon_losses, 'o-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('recon training loss (MSE)')  
    ax.set_title('recon loss across epochs')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(scaler_dir, 'recon_loss_vs_epochs.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, epochs+1), accuracies, 'o-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Task Training Accuracy (%)')
    ax.set_title('Training Accuracy across epochs')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(scaler_dir, 'Accuracy_vs_epochs.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


    # Track parameter magnitudes. At high rs the bias climbs toward the data
    # mean while the encoder/decoder weights stay near their init ("learn
    # nothing"); at low rs the weights carry the reconstruction.
    enc_weight_mag.append(float(np.mean(np.abs(W))))
    dec_weight_mag.append(float(np.mean(np.abs(W2))))
    bias_mag.append(float(np.mean(np.abs(W3))))

    print(f'  final task loss {last_task_loss:.6f} final recon loss {last_recon_loss:.6f} final accuracy {last_accuracy:.2f} |  mean|W_enc|={enc_weight_mag[-1]:.4f}  '
          f'mean|W_dec|={dec_weight_mag[-1]:.4f}  mean|bias|={bias_mag[-1]:.4f}  '
          f'->  images saved to {scaler_dir}/')

# ---------------------------------------------------------------------------
# Final loss vs Recon Scaler
# ---------------------------------------------------------------------------
final_task_losses = np.array(final_task_losses)
final_recon_losses = np.array(final_recon_losses)
final_accuracy = np.array(final_accuracy)
enc_weight_mag = np.array(enc_weight_mag)
dec_weight_mag = np.array(dec_weight_mag)
bias_mag = np.array(bias_mag)

# (1) Final loss vs Recon Scaler
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recon_scaler_vals, final_task_losses, 'o-')
ax.set_xscale('log')
ax.set_xlabel('Recon Scaler')
ax.set_ylabel('Final task training loss (Cross Entropy)')
ax.set_title('Final task loss vs recon scaler')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'final_task_loss_vs_recon.png'), dpi=120, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recon_scaler_vals, final_recon_losses, 'o-')
ax.set_xscale('log')
ax.set_xlabel('Recon Scaler')
ax.set_ylabel('Final recon training loss (MSE)')
ax.set_title('Final reconstruction loss vs Recon Scaler')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'final_recon_loss_vs_recon.png'), dpi=120, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recon_scaler_vals, final_accuracy, 'o-')
ax.set_xscale('log')
ax.set_xlabel('Recon Scaler')
ax.set_ylabel('Final accuracy')
ax.set_title('Final Accuracy vs Recon Scaler')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'final_accuracy_vs_recon.png'), dpi=120, bbox_inches='tight')
plt.show()

# (2) Where the signal lives: parameter magnitudes vs Recon Scaler.
# If the weights "learn nothing" they sit near their init magnitude while the
# bias rises toward the mean image (mean |pixel| ~ 0.131 for MNIST).
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recon_scaler_vals, enc_weight_mag, 'o-', label='mean |encoder weight|')
ax.plot(recon_scaler_vals, dec_weight_mag, 's-', label='mean |decoder weight|')
ax.plot(recon_scaler_vals, bias_mag, '^-', label='mean |decoder bias|')
ax.axhline(0.1307, color='grey', ls='--', lw=1, label='MNIST pixel mean (~0.131)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Recon Scaler')
ax.set_ylabel('Mean absolute value')
ax.set_title('Where the signal lives: weights vs bias across Recon Scaler')
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'magnitude_vs_Recon_Scaler.png'), dpi=120, bbox_inches='tight')
plt.show()

print('\nSweep complete. Summary:')
print(f'  {"Recon_Scaler":>10}  {"final_task_loss":>12} {"final_recon_loss":>12} {"|W_enc|":>9}  {"|W_dec|":>9}  {"|bias|":>9}')
for recon_scaler, l, rl, we, wd, b in zip(recon_scaler_vals, final_task_losses, final_recon_losses,
                            enc_weight_mag, dec_weight_mag, bias_mag):
    print(f'  {recon_scaler:>10.4g}  {l:>12.6f} {rl:>12.6f} {we:>9.4f}  {wd:>9.4f}  {b:>9.4f}')
