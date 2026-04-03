"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks
Build, train, and save CNN for MNIST digit recognition
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# CNN model for MNIST digit recognition
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        # Conv layer with 10 5x5 layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # Max pooling layer with a 2x2 window
        self.pool1 = nn.MaxPool2d(2)

        # Conv layer with 20 5x5 filters
        self.conv2 = nn.conv2d(10, 20, kernel_size=5)

        # Dropout layer with 0.5 dropout rate
        self.dropout = nn.Dropout(p=0.5)

        # Max pooling layer with a 2x2 window
        self.pool2 = nn.MaxPool2d(2)

        # Flattening operation
        self.flatten = nn.Flatten()

        # Fully conencted layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)

        # Output layer with 10 nodes (one for each digit class)
        self.fc2 = nn.Linear(50, 10)

    # Compute forward pass through network
    def forward(self, x):
        # Conv1 -> max pool (2x2) -> ReLu
        x = F.relu(self.pool1(self.conv1(x)))

        # Conv2 -> dropout -> max pool (2x2) -> ReLu
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))

        # Flatten
        x = self.flatten(x)

        # Fully connected with ReLU
        x = F.relu(self.fc1(x))

        # Output with log softmax
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

# Load and return MNIST train and test data loaders
def load_data(batch_size=64):
    # Normalize pixel values using MNIST mean and std for faster training convergence
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the 60k training images
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Download and load the 10k test images
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Declare train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Shuffle off

    return train_loader, test_loader

# Plot first six images from test set
def plot_first_six_test(test_loader):
    # Retrieve first batch of test examples
    examples = enumerate(test_loader)
    _, (images, labels) = next(examples)

    # Create a row of six subplots
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))

    for i in range(6):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')

    plt.suptitle('First Six MNIST Test Samples')
    plt.tight_layout()
    plt.savefig('First Six MNIST Test Samples.png')
    plt.show()
    print("Saved first six MNIST test samples!")

"""
Main function:
    Used to plot first six test digits, training model, compute average loss and accuracy of model,
    and save results to a file
"""
def main(argv):
    # Handle any command line arguments in argv
    batch_size = int(argv[1]) if len(argv) > 1 else 64
    num_epochs = int(argv[2]) if len(argv) > 2 else 5

    # Load training and test data
    train_loader, test_loader = load_data(batch_size)
    print("Train and test data loaded!")

    # Plot first six test digits
    plot_first_six_test(test_loader)

    return

if __name__ == "__main__":
    main(sys.argv)