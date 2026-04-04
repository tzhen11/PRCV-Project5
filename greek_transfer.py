
"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Transfer learning: retrain MNIST network to recognize Greek letters (alpha, beta, gamma)
"""

import sys
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mnist_network import MyNetwork
from test_network import load_model

class GreekTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, 0, [0, 0], 36/128, [0.0])
        x = TF.center_crop(x,(28,28))
        return TF.invert(x)
    
def load_greek_data(training_set_path, batch_size = 5):
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )

    return greek_train


def prepare_transfer_model(model_path):
    # Load pretrained MNIST model
    model = load_model(model_path)

    # Freeze all layers so pretrained weights don't change
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace last layer
    model.fc2 = nn.Linear(50,3)

    print("Modified network for Greek letter classification:")
    print(model)
    print()

    return model


# Train model on Greek letters for one epoch
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# Plot training loss and accuract over epochs
def plot_training_curves(losses, accuracies):
    epochs = range(1,len(losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, losses, 'b-o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(epochs, accuracies, 'g-o')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)

    plt.suptitle('Greek Letter Transfer Learning')
    plt.tight_layout()
    plt.savefig('greek_training_curves.png', dpi=150)
    plt.show()
    print("Saved greek_training_curves.png")


def main(argv):
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    data_path = argv[2] if len(argv) > 2 else 'greek_train'
    num_epochs = int(argv[3]) if len(argv) > 3 else 50

    print(f"Training for {num_epochs} epochs")

    train_loader = load_greek_data(data_path)
    model = prepare_transfer_model(model_path)
    optimizer = torch.optim.SGD(model.fc2.parameters(), lr=0.05, momentum=0.9)

    # dataset = train_loader.dataset
    # print(f"Classes: {dataset.classes}")
    # print(f"Total images: {len(dataset)}")
    # print(f"Class to idx: {dataset.class_to_idx}")

    # fig, axes = plt.subplots(1, 9, figsize=(18, 2))
    # for i in range(9):
    #     img, label = dataset[i * 3]
    #     axes[i].imshow(img[0], cmap='gray')
    #     axes[i].set_title(f"{dataset.classes[label]}")
    #     axes[i].axis('off')
    # plt.suptitle("Transformed Greek Letters")
    # plt.tight_layout()
    # plt.show()

    losses = []
    accuracies = []

    for epoch in range(1, num_epochs + 1):
        loss, accuracy = train_epoch(model, train_loader,optimizer,epoch)
        losses.append(loss)
        accuracies.append(accuracy)

        if epoch % 10 == 0 or accuracy == 100.0:
            print(f"Epoch {epoch}: Loss = {loss:.4f} | Accuracy = {accuracy:.1f}%")
        
        # stop early
        if accuracy == 100.0:
            print(f"Reached 100% accuracy at epoch {epoch}")
            break

    plot_training_curves(losses,accuracies)

    torch.save(model.state_dict(), 'greek_model.pth')
    print("Model saved to 'greek_model.pth'")

    return

if __name__ == "__main__":
    main(sys.argv)

