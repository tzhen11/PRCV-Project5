"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Read trained network and run on test set
"""

import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import existing network
from mnist_network import MyNetwork

# Import existing MNIST data loader
from mnist_network import load_data

# Load saved model weights from pth file
def load_model(model_path):
    # Instantiate network
    model = MyNetwork()

    # Load saved weights into network
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Set to eval mode
    model.eval()

    return model

# Run model on the first 10 test samples, show results of model output values, and plot first 9 samples
def run_first_ten(model, test_loader):
    # Get first batch of samples, using batch size of 10
    examples = enumerate(test_loader)
    _, (images, labels) = next(examples)

    # Run 10 images through network w/ no gradient
    with torch.no_grad():
        outputs = model(images)

    # Print out network output values w/ 2 decimal places
    print("{:^85} {:<20} {:<6}".format("'Output Values (log softmax)'", "'Max Val Index'", "'Correct Label'\n"))


    for i in range(10):
        # Convert output tensor to numpy for indexing
        values = outputs[i].numpy()

        # Retrieve predicted digit based on highest log prob
        predicted = values.argmax()
        actual = labels[i].item()
        formatted_values = " ".join(f"{v:7.2f}" for v in values)
        print(f"[{formatted_values}]           [{predicted}]                 [{actual}]")

    # Plot first 9 as 3x3 grid with prediction labels
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        predicted = outputs[i].numpy().argmax()
        actual = labels[i].item()

        # Green title if correct and red if wrong
        color = 'green' if predicted == actual else 'red'
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Predicted: {predicted} | Actual: {actual}", color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('First 9 MNIST Test Predictions\n(green = correct, red = incorrect)')
    plt.tight_layout()
    plt.savefig('Task_E_3_by_3.png', dpi=150)
    plt.show()
    print("\nSaved Task_E_3_by_3.png")

"""
Main function:
    Loads model, evaluates first 10 test samples, and plots the first 9 samples/results
"""
def main(argv):
    # Optional model path
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    # Load with batch size 10 since we will use first batch for testing
    _, test_loader = load_data(10)

    print("Running first ten test samples\n")
    run_first_ten(model, test_loader)

    return

if __name__ == "__main__":
    main(sys.argv)