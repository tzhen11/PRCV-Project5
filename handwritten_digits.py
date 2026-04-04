"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Test network on handwritten digits
"""

import sys
import os
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import existing network
from mnist_network import MyNetwork

# Import load model function from test_network file
from test_network import load_model

# Preprocess a single handwritten digit image to match MNIST format
def preprocess_image(image_path, invert=True):
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 28x28 to match MNIST input size with downsampling
    img = img.resize((28,28), Image.LANCZOS)

    # Invert pixel values so digit is white on black to match MNIST format
    if invert:
        img = ImageOps.invert(img)

    # Apply same normalization used on MNIST training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Add batch dimension
    tensor = transform(img).unsqueeze(0)

    return tensor, img

# Load images from directory with assumption they are name custom0.png to custom9.png
def load_images_from_directory(directory):
    # Compatible file extensions
    extensions = ['.png', '.jpg', '.jpeg']

    # Store image paths and expected labels
    image_paths = []
    expected_labels = []

    # Find file names by digit and append to corresponding lists
    for digit in range(10):
        for ext in extensions:
            path = os.path.join(directory, f'custom{digit}{ext}')
            if os.path.exists(path):
                image_paths.append(path)
                expected_labels.append(digit)
                break

    return image_paths, expected_labels

# Run model on list of preprocessed image tensors and return predictions
def predict_images(model, tensors):
    predictions = []

    # Pass image tensors through model
    with torch.no_grad():
        for tensor in tensors:
            # Run single image through network
            output = model(tensor)

            # Retrieve prediction based on highest log prob
            pred = output.argmax(dim=1).item()
            predictions.append(pred)

    return predictions

# Display all custom digit images in grid with their predictions
def display_results(preprocessed_images, predictions, expected_labels, save_path):
    n = len(preprocessed_images)

    cols = 5
    rows = (n + cols - 1) // cols # Compute number of rows needed to fit images

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    # Flatten axes array for indexing
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            # Display image
            ax.imshow(preprocessed_images[i], cmap='gray')
            pred = predictions[i]
            true = expected_labels[i]
            color = 'green' if pred == true else 'red'
            ax.set_title(f'True: {true} | Pred: {pred}', color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Custom Handwritten Digits\n(green = correct, red = incorrect)', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved resulting plot to {save_path}")

# Print accuracy of results
def print_accuracy(expected_labels, predictions):
    # Count total correct predictions
    correct = sum(p == t for p, t in zip(expected_labels, predictions))
    total = len(predictions)
    print(f"Accuracy on custom digits: {correct}/{total} = {100 * correct / total:.1f}%")

"""
Main function:
    Loads images from directory, preprocesses them to be compatible with MNIST dataset, predict using loaded model, and display results
"""
def main(argv):
    # Accept model path, image directory, and invert flag from command line
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    image_dir = argv[2] if len(argv) > 2 else 'handwritten_digits'
    invert = '--no-invert' not in argv

    # Check for valid image directory
    if not os.path.isdir(image_dir):
        print("Error: Image directory does not exist")
        sys.exit(1)

    print("Loading model")
    model = load_model(model_path)

    print("Loading custom handwritten digits")
    image_paths, expected_labels = load_images_from_directory(image_dir)

    # Check for valid images
    if not image_paths:
        print("Error: No images found")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Preprocess image into normalized tensor matching MNIST format
    tensors = []
    preprocessed_images = []
    for path in image_paths:
        tensor, preprocessed_image = preprocess_image(path, invert)
        tensors.append(tensor)
        preprocessed_images.append(preprocessed_image)

    # Run images through network
    predictions = predict_images(model, tensors)

    # Print accuracy of results
    print_accuracy(expected_labels, predictions)

    # Display and save results
    display_results(preprocessed_images, predictions, expected_labels, 'custom_digits_results.png')

    return

if __name__ == "__main__":
    main(sys.argv)