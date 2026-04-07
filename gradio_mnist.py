"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Interactive MNIST digit recognition using Gradio sketchpad
"""

import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms
from PIL import Image

from mnist_network import MyNetwork
from test_network import load_model


# Load the trained MNIST model
model = load_model('mnist_model.pth')


# Preprocess sketchpad input to match MNIST format and predict
def predict_digit(image):
    if image is None:
        return {str(i): 0.0 for i in range(10)}

    if isinstance(image, dict):
        image = image.get("composite", image)

    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert to numpy to check if we need to invert
    img_array = np.array(image)

    # Sketchpad draws dark on light, MNIST is white on black, so invert
    img_array = 255 - img_array

    # Apply same normalization as MNIST training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(img_array).unsqueeze(0)

    # Run through model
    with torch.no_grad():
        output = model(tensor)
        # Convert log probabilities to probabilities
        probs = torch.exp(output)[0]

    return {str(i): float(probs[i]) for i in range(10)}


# Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="pil", label="Draw a digit"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="MNIST Digit Recognition",
    description="Draw a digit (0-9) and see the model's prediction. Use thick strokes for best results.",
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch()