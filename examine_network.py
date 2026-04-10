u"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

This file examines the network and analyzes how data is processed.
Analyzing first layer and effect of the filters.
"""

import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import network architecture and model loader
from mnist_network import MyNetwork
from test_network import load_model

# Function to print model structure and layer names
def print_model_info(model):
    print("Model Structure:")
    print(model)
    print()

# Function to retrieve first convolutional layer weights
def get_first_layer_weights(model):
    # Retrieve first conv layer weights
    weights = model.conv1.weight

    print(f"Conv1 weight shape: {weights.shape}\n")

    # Print each of the 10 filters
    for i in range(10):
        print(f"Filter {i+1}:")
        print(f"Shape: {weights[i, 0].shape}")
        print(weights[i, 0])
        print()

    return weights

# Function to visualize 10 filters of first conv layer in 3x4 grid
def visualize_filters(weights):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Conv1 Filter Weights (10 5x5 Filters)")

    for i in range(10):
        ax = plt.subplot(3, 4, i + 1)

        # Detach from computation graph and convert filter image to numpy
        filter_img = weights[i, 0].detach().numpy()

        ax.imshow(filter_img, cmap="gray")
        ax.set_title(f'Filter {i+1}', fontsize=10)

        # Remove tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('conv1_filters.png', dpi=150)
    plt.show()
    print("Saved conv1_filters.png")


# Function to apply 10 conv1 filters to first training image using OpenCV's filter2D
def show_filter_effect(model):
    # Load MNIST training set to get first training sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Retrieve MNIST train dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Get first training image
    image_tensor, label = train_dataset[0]

    # Conver tensor to numpy array for OpenCV
    image_np = image_tensor[0].numpy()

    # Inspect weight w/ no grad needed
    with torch.no_grad():
        weights = model.conv1.weight

        fig, axes = plt.subplots(5, 4, figsize=(10, 10))
        fig.suptitle(f"Conv1 Filter Effect on First Training Image (Label: {label})")

        for i in range(10):
            # Extract i-th filter as numpy array
            kernel = weights[i, 0].detach().numpy()

            # Apply filter to image using OpenCV's filter2D
            filtered = cv2.filter2D(image_np, -1, kernel)

            # Filters 1-5 go in columns 0-1 and filters 6-10 go in columns 2-3
            row = i % 5
            col_offset = 0 if i < 5 else 2

            # Left cell for raw filter weights
            axes[row, col_offset].imshow(kernel, cmap="gray")
            axes[row, col_offset].set_title(f'Filter {i+1}', fontsize=10)
            axes[row, col_offset].set_xticks([])
            axes[row, col_offset].set_yticks([])

            # Right cell for filtered image
            axes[row, col_offset + 1].imshow(filtered, cmap="gray")
            axes[row, col_offset + 1].set_title(f'Effect {i + 1}', fontsize=10)
            axes[row, col_offset + 1].set_xticks([])
            axes[row, col_offset + 1].set_yticks([])

    plt.tight_layout()
    plt.savefig('filter_effects.png', dpi=150)
    plt.show()
    print("Saved filter_effects.png")



"""
Main function:
    Loads model, prints structure, and visualizes filters and their effects
"""
def main(argv):
    # Accept optional model path from command line
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'

    print("Loading model")
    model = load_model(model_path)

    # Print model structure
    print_model_info(model)

    # Print weights of first conv layer and visualize filters
    weights = get_first_layer_weights(model)
    visualize_filters(weights)

    # Show filter effects by applying 10 filters of conv1 layer to first training image of MNIST dataset
    show_filter_effect(model)

if __name__ == "__main__":
    main(sys.argv)