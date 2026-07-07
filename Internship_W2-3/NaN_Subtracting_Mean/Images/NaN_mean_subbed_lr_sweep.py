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
epochs = 25
n_neurons = 20
in_dim = 28 * 28

data_dir = './data'
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

total = torch.zeros(1, 28, 28)
n = 0
for images, _ in train_loader:
    total += images.sum(dim=0)   # sum over the batch
    n += images.size(0)
mean_image = nn.Flatten()(total / n)        # shape [1, 28, 28]
mean_image = mean_image.to(device)


class NeuronAutoencoder(nn.Module):
    """Each of the n_neurons is its own autoencoder: its scalar activation is
    the latent variable, from which it reconstructs the FULL input via its own
    decoder weight vector and bias image."""
    def __init__(self, n_neurons=20, in_dim=784):
        super().__init__()
        self.encoder = nn.Linear(in_dim, n_neurons)                       # (20, 784) weight
        self.decoder_weights = nn.Parameter(torch.randn(n_neurons, in_dim) * 0.01)
        self.decoder_bias = nn.Parameter(torch.zeros(n_neurons, in_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        x = x - mean_image
        features = x                                        # [batch, 784]
        h = torch.sigmoid(self.encoder(x))                  # [batch, 20]  one latent per neuron
        # neuron i reconstructs the input as  h_i * decoder_weights[i] + decoder_bias[i]
        decoded = (h.unsqueeze(2) * self.decoder_weights.unsqueeze(0)
                   + self.decoder_bias.unsqueeze(0))        # [batch, 20, 784]
        return decoded, features


def train(data_loader, model, criterion, optimizer):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0.0
    for data, _ in data_loader:
        data = data.to(device)
        decoded, features = model(data)
        # every neuron is scored against the same input
        target = features.unsqueeze(1).expand_as(decoded)   # [batch, 20, 784]
        loss = criterion(decoded, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def save_grid(mat, path, title_prefix='neuron'):
    """Save a 4x5 grid of 28x28 images, one per neuron. Works for the encoder
    weights, decoder weights, AND decoder biases -- all are (20, 784) here."""
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        filt = mat[i].reshape(28, 28)
        m = np.abs(filt).max()
        m = m if m > 0 else 1e-12                            # avoid vmin==vmax on a dead unit
        ax.imshow(filt, cmap='seismic', vmin=-m, vmax=m)
        ax.set_title(f'{title_prefix} {i}')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Learning-rate sweep: 20 values, log-spaced from 0.001 to 10
# ---------------------------------------------------------------------------
learning_rates = np.logspace(-3, 3, 30)   # [0.001, ..., 10]
final_losses = []
enc_weight_mag = []   # mean(|encoder weight|) per lr
dec_weight_mag = []   # mean(|decoder weight|) per lr
bias_mag = []         # mean(|decoder bias|) per lr, averaged over all 20 neurons

os.makedirs('Internship_W2-3\\Neuron_Autoencoder\\sweep', exist_ok=True)

for lr in learning_rates:
    lr_name = f'lr_{lr:.3g}'                # e.g. lr_0.001, lr_0.183, lr_10
    lr_dir = os.path.join('sweep', lr_name)
    os.makedirs(lr_dir, exist_ok=True)

    print(f'\n=== Training {lr_name}  (lr={lr:.4g}) ===')

    # Identical init AND data order every run, so lr is the only variable.
    # (Also means the same neurons stay dead across the whole sweep -- that's
    # a property of this seed's initialisation, not of the learning rate.)
    torch.manual_seed(0)

    model = NeuronAutoencoder(n_neurons, in_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    last_loss = float('nan')
    for epoch in range(epochs):
        last_loss = train(train_loader, model, criterion, optimizer)
        print(f'  epoch {epoch + 1:2d}/{epochs}  loss={last_loss:.6f}')

    final_losses.append(last_loss)

    # All three parameter blocks are (20, 784): one 28x28 image per neuron.
    W = model.encoder.weight.detach().cpu().numpy()          # (20, 784)
    W2 = model.decoder_weights.detach().cpu().numpy()        # (20, 784)  -- no transpose needed
    W3 = model.decoder_bias.detach().cpu().numpy()           # (20, 784)

    save_grid(W, os.path.join(lr_dir, 'encoder_weight.png'), 'neuron')
    save_grid(W2, os.path.join(lr_dir, 'decoder_weight.png'), 'neuron')
    save_grid(W3, os.path.join(lr_dir, 'decoder_bias.png'), 'neuron')

    # Magnitudes. For the bias there are now 20 images; the mean absolute value
    # is taken over ALL 20x784 entries, exactly as encoder/decoder are treated,
    # so the three curves stay on equal footing (one number per lr).
    enc_weight_mag.append(float(np.mean(np.abs(W))))
    dec_weight_mag.append(float(np.mean(np.abs(W2))))
    bias_mag.append(float(np.mean(np.abs(W3))))

    print(f'  final loss {last_loss:.6f}  |  mean|W_enc|={enc_weight_mag[-1]:.4f}  '
          f'mean|W_dec|={dec_weight_mag[-1]:.4f}  mean|bias|={bias_mag[-1]:.4f}  '
          f'->  images saved to {lr_dir}/')

# ---------------------------------------------------------------------------
# Summary plots
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
ax.set_title('Final neuron-autoencoder loss vs learning rate')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join('sweep', 'final_loss_vs_lr.png'), dpi=120, bbox_inches='tight')
plt.show()

# (2) Where the signal lives: parameter magnitudes vs learning rate.
# mean |pixel| ~ 0.131 for MNIST; if a unit "learns nothing" its bias drifts
# toward the mean image while its encoder weights stay near their init.
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(learning_rates, enc_weight_mag, 'o-', label='mean |encoder weight|')
ax.plot(learning_rates, dec_weight_mag, 's-', label='mean |decoder weight|')
ax.plot(learning_rates, bias_mag, '^-', label='mean |decoder bias| (all 20)')
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