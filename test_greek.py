"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Test transfer-learned Greek letter model on custom handwritten images
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from mnist_network import MyNetwork
from test_network import load_model


# Class label mapping
CLASS_NAMES = ['alpha', 'beta', 'gamma']


# Load Greek letter model with modified last layer
def load_greek_model(model_path):
    model = MyNetwork()

    # Replace last layer to match Greek model structure (3 classes instead of 10)
    model.fc2 = nn.Linear(50, 3)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model


# Preprocess a single image to match Greek training format
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(img).unsqueeze(0)

    return tensor, TF.rgb_to_grayscale(transforms.ToTensor()(img))


# Same transform used during training
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, 0, [0, 0], 36/128, [0.0])
        x = TF.center_crop(x, (28, 28))
        return TF.invert(x)


# Parse expected label from filename (e.g. "alpha_1.png" -> 0)
def parse_label(filename):
    name = filename.lower().split('-')[0]
    if name in CLASS_NAMES:
        return CLASS_NAMES.index(name)
    return -1


# Load all images from directory and parse labels from filenames
def load_images(directory):
    extensions = ['.png', '.jpg', '.jpeg']
    image_paths = []
    labels = []

    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            label = parse_label(filename)
            if label >= 0:
                image_paths.append(os.path.join(directory, filename))
                labels.append(label)

    return image_paths, labels


# Run model on all images and display results
def test_and_display(model, image_paths, labels, save_path):
    predictions = []
    images_for_display = []

    with torch.no_grad():
        for path in image_paths:
            tensor, display_img = preprocess_image(path)
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
            images_for_display.append(display_img)

    # Print results
    correct = 0
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        pred_name = CLASS_NAMES[predictions[i]]
        true_name = CLASS_NAMES[labels[i]]
        match = "correct" if predictions[i] == labels[i] else "WRONG"
        if predictions[i] == labels[i]:
            correct += 1
        print(f"{filename}: predicted={pred_name}, actual={true_name} [{match}]")

    print(f"\nAccuracy: {correct}/{len(labels)} = {100 * correct / len(labels):.1f}%")

    # Plot results in grid
    n = len(image_paths)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images_for_display[i][0], cmap='gray')
            pred_name = CLASS_NAMES[predictions[i]]
            true_name = CLASS_NAMES[labels[i]]
            color = 'green' if predictions[i] == labels[i] else 'red'
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Custom Greek Letter Classification\n(green = correct, red = incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")



def main(argv):
    model_path = argv[1] if len(argv) > 1 else 'greek_model.pth'
    image_dir = argv[2] if len(argv) > 2 else 'handwritten_greek'

    if not os.path.isdir(image_dir):
        print(f"Error: Directory '{image_dir}' not found")
        sys.exit(1)

    model = load_greek_model(model_path)

    image_paths, labels = load_images(image_dir)

    if not image_paths:
        print("Error: No valid images found")
        sys.exit(1)

    print(f"Found {len(image_paths)} images\n")
    test_and_display(model, image_paths, labels, 'greek_test_results.png')

    return


if __name__ == "__main__":
    main(sys.argv)