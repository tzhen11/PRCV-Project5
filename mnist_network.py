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
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

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

# Train network for one epoch and return average loss
def train_network(model, train_loader, optimizer, epoch):
    # Initialize evaluation params
    correct = 0
    total = 0
    total_loss = 0

    # Set model to train mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Clear gradients from prev step
        optimizer.zero_grad()

        # Run current batch through network
        output = model(data)

        # Compute negative log likelihood loss between predictions and true labels
        loss = F.nll_loss(output, target)

        # Backprop the loss to compute gradients
        loss.backward()

        # Update model weights using computed gradients
        optimizer.step()

        # Count up loss and correct predictions
        total_loss += loss.item()
        pred = output.argmax(dim=1) # Prediction is the index of the highest log-prob
        correct += pred.eq(target).sum().item()
        total += len(data)

        # Print progress every 200 batches
        if batch_idx % 200 == 0:
            print(f'  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f}')

    # Average loss over all batches
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# Evaluate network on data loader and return average loss and accuracy
def evaluate_network(model, loader):
    # Set model to eval mode to disable dropout so output is deterministic
    model.eval()
    total_loss = 0
    correct = 0

    # Disable gradient computation since no training
    with torch.no_grad():
        for data, target in loader:
            output = model(data)

            # Sum loss across all examples
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    # Divide by total number of examples to get average loss per sample
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    return avg_loss, accuracy


# Plot training and testing loss and accuracy curves
def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    # Build list of epoch numbers for x-axis
    epochs = range(1, len(train_losses) + 1)

    # Create two side-by-side plots with one for loss, one for accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Configure loss plot (blue for training and red for test)
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Configure accuracy plot (blue for training and red for test)
    ax2.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-o', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('MNIST CNN Training Progress')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Saved training_curves.png")

"""
Main function:
    Used to plot first six test digits, training model, compute average loss and accuracy of model,
    and save results to a file
"""
def main(argv):
    # Handle any command line arguments in argv
    batch_size = int(argv[1]) if len(argv) > 1 else 64
    num_epochs = int(argv[2]) if len(argv) > 2 else 5

    print(f"Using batch size of {batch_size} and {num_epochs} epochs")

    learning_rate = 0.01
    momentum = 0.5
    saved_model_path = 'mnist_model.pth'

    # Load training and test data
    train_loader, test_loader = load_data(batch_size)
    print("Train and test data loaded!")

    # Plot first six test digits
    plot_first_six_test(test_loader)

    # Instantiate the network and SGD optimizer
    model = MyNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Init lists to track metrics across epochs for plotting
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}")

        # Train for one full pass through training set
        train_loss, train_accuracy = train_network(model, train_loader, optimizer, epoch)

        # Evaluate on both sets after each epoch
        test_loss, test_accuracy = evaluate_network(model, test_loader)

        # Store results for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    # Plot and save the training curves
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    # Save the trained model weights to disk for use in test_network.py and custom_digits.py
    torch.save(model.state_dict(), saved_model_path)
    print(f"\nModel saved to '{saved_model_path}'")


    return

if __name__ == "__main__":
    main(sys.argv)