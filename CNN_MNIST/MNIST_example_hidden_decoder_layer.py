import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #a convolution layer, 1 input channel, 32 output channels, aka, 32 different learned convolutions
                                            #applied to the input image. 1 way to think about this might be 28*28 image becomes 32 26*26 output images
                                            # each pixel in an output images being the output of a neuron whos inputs are 9 of the inputs and whos weights are the values from the kernel for that channel.
        self.conv2 = nn.Conv2d(32, 64, 3, 1)#another convolution layer, 32 input channels, 64 output channels, this means the 3x3 kernel is being performed on all 32
                                            #of the previous outputs simultaneously, the kernel is 3x3x32, and we have 64 different learned kernels from this.
                                            #original image is 28*28, then we have 32 26*26 convolved images, then 64 24*24 images = 36,864 outputs, maxpool gives max of 4 pixels = 9216
        self.dropout1 = nn.Dropout(0.25)#during training, randomly zeroes some of the elements of the input tensor, it forces neurons to be independant 
        self.dropout2 = nn.Dropout(0.5)#probability of zeroing value is input
        #fully connected layer: standard ANN layers
        self.fc1 = nn.Linear(9216, 128)#9216 from maxpool to 128
        self.decoder = nn.Linear(128, 9216)
        self.fc2 = nn.Linear(128, 10)#128 inputs, 10 outputs (0 through 10), 1 hidden layer of 128 neurons

    def forward(self, x):
        x = self.conv1(x) #first convolution
        x = F.relu(x) #non-linearity
        x = self.conv2(x) #second convolution
        x = F.relu(x) #non-linearity
        x = F.max_pool2d(x, 2) #split each output image into 2x2 squares, take max value from 2x2 square. goes from 24*24 to 12*12
        x = self.dropout1(x) #zero 1/4 values randomly, forces neurons to be robust, they can't rely on other neurons providing input, prevent overfitting
        x = torch.flatten(x, 1) #turn into 1 dimensional array of 9216 values to give 1 value to each neuron
        features = x
        x = self.fc1(x) #first layer
        x = F.relu(x) #non-linearity
        decoded = None
        if self.training:
            decoded = self.decoder(x)
        x = self.dropout2(x) #zero half the outputs
        x = self.fc2(x) #second layer
        output = F.log_softmax(x, dim=1) #log softmax, more negative means less confident in value
        return output, decoded, features


def train(args, model, device, train_loader, optimizer, epoch, recon_scaler):
    model.train()
    total_class_loss = 0.0
    total_recon_loss = 0.0
    total_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, decoded, features = model(data)

        recon_loss = F.mse_loss(decoded, features)
        class_loss = F.nll_loss(output, target)
        loss = class_loss + recon_scaler * recon_loss

        loss.backward()
        optimizer.step()

        total_class_loss += class_loss.item()
        total_recon_loss += recon_loss.item()
        total_batches += 1

        if batch_idx % args.log_interval == 0:
            print('  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (class: {:.6f}, recon: {:.6f})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                class_loss.item(), recon_loss.item()))
            if args.dry_run:
                break

    avg_class_loss = total_class_loss / total_batches
    avg_recon_loss = total_recon_loss / total_batches
    return avg_class_loss, avg_recon_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


def run_single_experiment(args, device, train_loader, test_loader, recon_scaler):
    """Train a fresh model with a given recon_scaler and return per-epoch results."""

    # Fresh model and optimizer for each sweep value
    torch.manual_seed(args.seed)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    epoch_results = []

    for epoch in range(1, args.epochs + 1):
        avg_class_loss, avg_recon_loss = train(
            args, model, device, train_loader, optimizer, epoch, recon_scaler
        )
        test_loss, accuracy = test(model, device, test_loader)
        scheduler.step()

        epoch_results.append({
            'recon_scaler': recon_scaler,
            'epoch': epoch,
            'train_class_loss': avg_class_loss,
            'train_recon_loss': avg_recon_loss,
            'test_loss': test_loss,
            'test_accuracy': accuracy,
        })

    return epoch_results


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST NaN Sweep')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--output-csv', type=str, default='sweep_results_ANN.csv',
                        help='Path for output CSV file (default: sweep_results_ANN.csv)')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # --- Sweep configuration ---
    # Log-spaced from 0.001 to 10, 10 values
    # This gives roughly: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
    # Equal representation per order of magnitude
    recon_scalers = np.logspace(np.log10(0.001), np.log10(10), num=10).tolist()

    print("=" * 70)
    print("RECON_SCALER SWEEP")
    print(f"Values: {[f'{v:.4f}' for v in recon_scalers]}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Total training runs: {len(recon_scalers)}")
    print("=" * 70)

    all_results = []

    for i, scaler in enumerate(recon_scalers):
        print(f"\n{'=' * 70}")
        print(f"RUN {i+1}/{len(recon_scalers)} — recon_scaler = {scaler:.6f}")
        print(f"{'=' * 70}")

        epoch_results = run_single_experiment(
            args, device, train_loader, test_loader, scaler
        )
        all_results.extend(epoch_results)

        # Write CSV after each run so partial results are saved if interrupted
        fieldnames = ['recon_scaler', 'epoch', 'train_class_loss',
                      'train_recon_loss', 'test_loss', 'test_accuracy']
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        final = epoch_results[-1]
        print(f"  FINAL — accuracy: {final['test_accuracy']:.2f}%, "
              f"test_loss: {final['test_loss']:.4f}")

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY (final epoch per scaler)")
    print(f"{'=' * 70}")
    print(f"{'recon_scaler':>14} | {'test_accuracy':>13} | {'test_loss':>9} | {'train_recon_loss':>16}")
    print("-" * 60)
    for scaler in recon_scalers:
        final = [r for r in all_results if r['recon_scaler'] == scaler and r['epoch'] == args.epochs][0]
        print(f"{final['recon_scaler']:>14.6f} | {final['test_accuracy']:>12.2f}% | {final['test_loss']:>9.4f} | {final['train_recon_loss']:>16.6f}")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == '__main__':
    main()