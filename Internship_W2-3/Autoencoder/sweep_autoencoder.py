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
epochs = 50

data_dir = './data'
print('data_dir =', data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(28 * 28, 20)
        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        x = nn.Flatten()(x)
        features = x                      # shape = [batch_size, 784]
        x = self.encoder(x)
        x = torch.relu(x)
        decoded = self.decoder(x)         # shape = [batch_size, 784]
        return decoded, features


def train(data_loader, model, criterion, optimizer):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0.0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        decoded, features = model(data)
        loss = criterion(decoded, features)
        total_loss += loss.item()          # .item() so we don't retain the graph

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / num_batches        # average loss for this epoch


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


# ---------------------------------------------------------------------------
# Learning-rate sweep: 10 values, log-spaced from 0.001 to 10
# ---------------------------------------------------------------------------
learning_rates = np.logspace(-1, 1, 10)   # [0.001, ..., 10]
final_losses = []
enc_weight_mag = []   # mean(|encoder weight|) per lr
dec_weight_mag = []   # mean(|decoder weight|) per lr
bias_mag = []         # mean(|decoder bias|)   per lr

os.makedirs('sweep', exist_ok=True)

for lr in learning_rates:
    lr_name = f'lr_{lr:.3g}'               # e.g. lr_0.001, lr_0.167, lr_10
    lr_dir = os.path.join('sweep', lr_name)
    os.makedirs(lr_dir, exist_ok=True)

    print(f'\n=== Training {lr_name}  (lr={lr:.4g}) ===')

    # Same initialisation for every run so the weights/bias are directly
    # comparable across learning rates. Remove this line for random inits.
    torch.manual_seed(0)

    model = SimpleMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    last_loss = float('nan')
    for epoch in range(epochs):
        last_loss = train(train_loader, model, criterion, optimizer)
        print(f'  epoch {epoch + 1:2d}/{epochs}  loss={last_loss:.6f}')

    final_losses.append(last_loss)

    # Pull out the trained parameters (same layout as the original code)
    W = model.encoder.weight.detach().cpu().numpy()      # (20, 784)
    W2 = model.decoder.weight.detach().cpu().numpy()     # (784, 20)
    W3 = model.decoder.bias.detach().cpu().numpy()       # (784,)
    W2 = np.transpose(W2, (1, 0))                         # (20, 784)

    save_weight_grid(W, os.path.join(lr_dir, 'encoder_weight.png'), 'neuron')
    save_weight_grid(W2, os.path.join(lr_dir, 'decoder_weight.png'), 'neuron')
    save_bias(W3, os.path.join(lr_dir, 'decoder_bias.png'))

    # Track parameter magnitudes. At high lr the bias climbs toward the data
    # mean while the encoder/decoder weights stay near their init ("learn
    # nothing"); at low lr the weights carry the reconstruction.
    enc_weight_mag.append(float(np.mean(np.abs(W))))
    dec_weight_mag.append(float(np.mean(np.abs(W2))))
    bias_mag.append(float(np.mean(np.abs(W3))))

    print(f'  final loss {last_loss:.6f}  |  mean|W_enc|={enc_weight_mag[-1]:.4f}  '
          f'mean|W_dec|={dec_weight_mag[-1]:.4f}  mean|bias|={bias_mag[-1]:.4f}  '
          f'->  images saved to {lr_dir}/')

# ---------------------------------------------------------------------------
# Final loss vs learning rate
# ---------------------------------------------------------------------------
final_losses = np.array(final_losses)
enc_weight_mag = np.array(enc_weight_mag)
dec_weight_mag = np.array(dec_weight_mag)
bias_mag = np.array(bias_mag)

# (1) Final loss vs learning rate
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(learning_rates, final_losses, 'o-')
ax.set_xscale('log')
ax.set_xlabel('Learning rate')
ax.set_ylabel('Final training loss (MSE)')
ax.set_title('Final autoencoder loss vs learning rate')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'final_loss_vs_lr.png'), dpi=120, bbox_inches='tight')
plt.show()

# (2) Where the signal lives: parameter magnitudes vs learning rate.
# If the weights "learn nothing" they sit near their init magnitude while the
# bias rises toward the mean image (mean |pixel| ~ 0.131 for MNIST).
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(learning_rates, enc_weight_mag, 'o-', label='mean |encoder weight|')
ax.plot(learning_rates, dec_weight_mag, 's-', label='mean |decoder weight|')
ax.plot(learning_rates, bias_mag, '^-', label='mean |decoder bias|')
ax.axhline(0.1307, color='grey', ls='--', lw=1, label='MNIST pixel mean (~0.131)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Learning rate')
ax.set_ylabel('Mean absolute value')
ax.set_title('Where the signal lives: weights vs bias across learning rate')
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'magnitude_vs_lr.png'), dpi=120, bbox_inches='tight')
plt.show()

print('\nSweep complete. Summary:')
print(f'  {"lr":>10}  {"final_loss":>12}  {"|W_enc|":>9}  {"|W_dec|":>9}  {"|bias|":>9}')
for lr, l, we, wd, b in zip(learning_rates, final_losses,
                            enc_weight_mag, dec_weight_mag, bias_mag):
    print(f'  {lr:>10.4g}  {l:>12.6f}  {we:>9.4f}  {wd:>9.4f}  {b:>9.4f}')
