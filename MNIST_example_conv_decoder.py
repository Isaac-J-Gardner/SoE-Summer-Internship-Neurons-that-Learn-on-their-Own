import argparse
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
        self.fc2 = nn.Linear(128, 10)#128 inputs, 10 outputs (0 through 10), 1 hidden layer of 128 neurons

    def forward(self, x):
        x = self.conv1(x) #first convolution
        x = F.relu(x) #non-linearity
        x = self.conv2(x) #second convolution
        x = F.relu(x) #non-linearity
        x = F.max_pool2d(x, 2) #split each output image into 2x2 squares, take max value from 2x2 square. goes from 24*24 to 12*12
        x = self.dropout1(x) #zero 1/4 values randomly, forces neurons to be robust, they can't rely on other neurons providing input, prevent overfitting
        x = torch.flatten(x, 1) #turn into 1 dimensional array of 9216 values to give 1 value to each neuron
        x = self.fc1(x) #first layer
        x = F.relu(x) #non-linearity
        x = self.dropout2(x) #zero half the outputs
        x = self.fc2(x) #second layer
        output = F.log_softmax(x, dim=1) #log softmax, more negative means less confident in value
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train() #switch into training mode, dropout layers take effect, outputs scaled up by dropout percentage
    for batch_idx, (data, target) in enumerate(train_loader): #default batch size is 64, batch_idx is the id (0 through 63), data is tensor of shape [64, 1, 28, 28], and target is [64]
        data, target = data.to(device), target.to(device) #send data to same device as where the model is stored
        optimizer.zero_grad() #zero gradients
        output = model(data) #model computs the data, outputs 10 log_softmax values
        loss = F.nll_loss(output, target) #negative log-likelihood loss, takes the loss associated with the target and negates it
        loss.backward() #propagate the loss associated with target through the network, computing d_loss/d_weight for each neuron through chain rule.
        optimizer.step() #optimiser applies changes through gradient descent (weight = weight - learning_rate * gradient), negative as gradient DESCENT
        if batch_idx % args.log_interval == 0: #after log_interval batches, print progress
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
print("change")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()