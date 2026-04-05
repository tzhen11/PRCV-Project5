"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Test trained Vision Transformer on MNIST test set
"""

import sys
import torch
import matplotlib.pyplot as plt

from mnist_network import load_data
from net_transformer import NetTransformer, NetConfig


def load_transformer_model(model_path):
    config = NetConfig()
    model = NetTransformer(config)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    return model


def run_first_ten(model, test_loader):
    examples = enumerate(test_loader)
    _, (images, labels) = next(examples)

    with torch.no_grad():
        outputs = model(images)

    print("{:^85} {:<20} {:<6}".format(
        "'Output Values (log softmax)'", "'Max Val Index'", "'Correct Label'\n"))

    for i in range(10):
        values = outputs[i].numpy()
        predicted = values.argmax()
        actual = labels[i].item()
        formatted_values = " ".join(f"{v:7.2f}" for v in values)
        print(f"[{formatted_values}]           [{predicted}]                 [{actual}]")

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        predicted = outputs[i].numpy().argmax()
        actual = labels[i].item()
        color = 'green' if predicted == actual else 'red'
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Predicted: {predicted} | Actual: {actual}", color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('Transformer: First 9 MNIST Test Predictions\n(green = correct, red = incorrect)')
    plt.tight_layout()
    plt.savefig('transformer_test_predictions.png', dpi=150)
    plt.show()
    print("\nSaved transformer_test_predictions.png")


def main(argv):
    model_path = argv[1] if len(argv) > 1 else 'transformer_model.pth'

    print(f"Loading transformer model from {model_path}")
    model = load_transformer_model(model_path)

    _, test_loader = load_data(10)

    print("Running first ten test samples\n")
    run_first_ten(model, test_loader)


if __name__ == "__main__":
    main(sys.argv)