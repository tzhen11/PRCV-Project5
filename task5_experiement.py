"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Examines the network model when tweaking number of filters in first convolutional layer,
number of hidden nodes in fully connected layer, and drop out rate.
"""

import sys
import csv
import time
import itertools
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


"""
Custom configurable network that follows structure of Task 1 with number of filters in first convolutional layer,
number of hidden nodes in fully connected layer, and drop out rate.
"""
class ExperimentNetwork(nn.Module):
    def __init__(self,
                 num_filters1=10,
                 num_filters2=20,
                 hidden_nodes=50,
                 dropout_rate=0.5,
                 kernel_size=5):
        super(ExperimentNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        flat_size = self._compute_flat_size()

        self.fc1 = nn.Linear(flat_size, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 10)

    def _compute_flat_size(self):
        dummy = torch.zeros(1, 1, 28, 28).to(next(self.parameters()).device)
        dummy = F.relu(self.pool1(self.conv1(dummy)))
        dummy = F.relu(self.pool2(self.dropout(self.conv2(dummy))))
        return dummy.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# Function to load fashion MNIST data loaders
def load_fashion_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

# Helper function to train one epoch
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)

    return total_loss / len(train_loader), 100.0 * correct / total

# Function to evaluate loss and accuracy
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return total_loss / len(loader.dataset), 100.0 * correct / len(loader.dataset)

"""
Function to build and train a single network with custom configurations and
returns the results of the run for examination.
"""
def run_experiment(label, num_filters1, num_filters2, hidden_nodes,
                   dropout_rate, kernel_size, batch_size, num_epochs,
                   train_loader, test_loader):

    print(f"{label}")

    #train_loader, test_loader = load_fashion_mnist(batch_size)

    model = ExperimentNetwork(num_filters1, num_filters2, hidden_nodes, dropout_rate, kernel_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    best_test_acc, best_epoch = 0.0, 0
    train_accs, test_accs = [], []

    start = time.time()

    for epoch in range(1, num_epochs + 1):
        _, train_acc = train_epoch(model, train_loader, optimizer)
        _, test_acc = evaluate(model, test_loader)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch

        print(f"Epoch {epoch}: train={train_acc:.1f}% test={test_acc:.1f}%")

    return {
        'label': label,
        'num_filters1': num_filters1,
        'num_filters2': num_filters2,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'kernel_size': kernel_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'param_count': sum(p.numel() for p in model.parameters()),
        'train_time_s': round(time.time() - start, 1),
        'train_accs': train_accs,
        'test_accs': test_accs
    }

# Function to plot each experimental run's test accuracy
def plot_summary_bar(all_results, save_path):
    labels = [r['label'] for r in all_results]
    accs = [r['best_test_acc'] for r in all_results]

    plt.figure(figsize=(max(10, int(len(labels) * 0.5)), 5))
    plt.bar(labels, accs)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to save csv for further examination
def save_csv(all_results, path):
    keys = ['label', 'num_filters1', 'num_filters2', 'hidden_nodes', 'dropout_rate', 'kernel_size', 'batch_size', 'num_epochs', 'best_test_acc', 'best_epoch', 'param_count', 'train_time_s']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in keys})


"""
Main function does one full sweep of each customizable dimension,
runs the model on random configurations, and stores results of each run for further examination
to find the optimal configuration
"""
def main(argv):
    # Default number of epochs and batch size that's configurable through command line
    num_epochs = int(argv[1]) if len(argv) > 1 else 8
    batch_size = int(argv[2]) if len(argv) > 2 else 128

    # Loads train and test loader
    train_loader, test_loader = load_fashion_mnist(batch_size)

    # Creates directory for storing results
    output_dir = "experiment"
    os.makedirs(output_dir, exist_ok=True)

    # Fixed configurations that aren't used for experimenting
    BASE_FILTERS2 = 20
    BASE_KERNEL = 5

    # List of configurable dimension values for testing
    dropout_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    hidden_values = [50, 100, 200, 400, 500, 600, 700, 800, 1000]
    filter_values = [8, 10, 16, 32, 64, 128, 256, 512, 1024]

    # Default dimensions used for sweeps
    DEFAULT_DR = 0.5
    DEFAULT_HN = 50
    DEFAULT_NF = 10

    # Store all results and configurations seen
    all_results = []
    seen_configs = set()

    # Run sweeps for number of filters individually to see how it affects the results
    for nf in filter_values:
        config_key = (nf, DEFAULT_HN, round(DEFAULT_DR, 1))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        res = run_experiment(f"nf={nf}", nf, BASE_FILTERS2, DEFAULT_HN, DEFAULT_DR, BASE_KERNEL, batch_size, num_epochs, train_loader, test_loader)
        all_results.append(res)

    best_nf = max(all_results, key=lambda r: r['best_test_acc'])['num_filters1']

    # Run sweeps for number of hidden nodes in fully connected layer individually to see how it affects the results
    for hn in hidden_values:
        config_key = (DEFAULT_NF, hn, round(DEFAULT_DR, 1))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        res = run_experiment(f"hn={hn}", DEFAULT_NF, BASE_FILTERS2, hn, DEFAULT_DR, BASE_KERNEL, batch_size, num_epochs, train_loader, test_loader)
        all_results.append(res)

    best_hn = max(all_results, key=lambda r: r['best_test_acc'])['hidden_nodes']

    # Run sweeps for drop out rate individually to see how it affects the results
    for dr in dropout_values:
        config_key = (DEFAULT_NF, DEFAULT_HN, round(dr, 1))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        res = run_experiment(f"dr={dr}", DEFAULT_NF, BASE_FILTERS2, DEFAULT_HN, dr, BASE_KERNEL, batch_size, num_epochs, train_loader, test_loader)
        all_results.append(res)

    best_dr = max(all_results, key=lambda r: r['best_test_acc'])['dropout_rate']

    print(f"Best after linear search: nf={best_nf}, hn={best_hn}, dr={best_dr}")

    # Begin randomized optimization
    current_best_nf = best_nf
    current_best_hn = best_hn
    current_best_dr = best_dr

    dimensions = ['nf', 'hn', 'dr']

    while len(all_results) < 75:
        # Check to see if all candidates have been exhausted to avoid infinite loop
        nf_candidates = [nf for nf in filter_values if
                         (nf, current_best_hn, round(current_best_dr, 1)) not in seen_configs]
        hn_candidates = [hn for hn in hidden_values if
                         (current_best_nf, hn, round(current_best_dr, 1)) not in seen_configs]
        dr_candidates = [dr for dr in dropout_values if
                         (current_best_nf, current_best_hn, round(dr, 1)) not in seen_configs]

        if not any([nf_candidates, hn_candidates, dr_candidates]):
            print("All candidates along current best axes exhausted!")
            break

        dim = random.choice(dimensions)

        if dim == 'nf':
            if not nf_candidates:
                continue
            nf = random.choice(nf_candidates)
            hn, dr = current_best_hn, current_best_dr
            label = f"R_nf={nf}|hn={hn}|dr={dr}"

        elif dim == 'hn':
            if not hn_candidates:
                continue
            hn = random.choice(hn_candidates)
            nf, dr = current_best_nf, current_best_dr
            label = f"R_nf={nf}|hn={hn}|dr={dr}"

        else:  # dr
            if not dr_candidates:
                continue
            dr = random.choice(dr_candidates)
            nf, hn = current_best_nf, current_best_hn
            label = f"R_nf={nf}|hn={hn}|dr={dr}"

        config_key = (nf, hn, round(dr, 1))
        seen_configs.add(config_key)

        # Run experiment using randomly chosen configurations
        res = run_experiment(label, nf, BASE_FILTERS2, hn, dr, BASE_KERNEL, batch_size, num_epochs, train_loader, test_loader)
        all_results.append(res)

        # Update current best configs if this result is better
        current_best_acc = max(all_results, key=lambda r: r['best_test_acc'])
        current_best_nf = current_best_acc['num_filters1']
        current_best_hn = current_best_acc['hidden_nodes']
        current_best_dr = current_best_acc['dropout_rate']

        print(f"Current bests: nf={current_best_nf}, hn={current_best_hn}, dr={current_best_dr}")

    # Plot test accuracies of each run
    plot_summary_bar(all_results, os.path.join(output_dir, "summary.png"))

    # Store all results to a csv
    save_csv(all_results, os.path.join(output_dir, "results.csv"))

    print(f"Total experiments run: {len(all_results)}")
    print(f"Saved outputs to '{output_dir} folder'")


if __name__ == "__main__":
    main(sys.argv)