"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

This file is used to examine the best performing experiments based on test accuracy.
It will print all model configuration corresponding results for examination to see which model
is the mode "optimal" based on what one considers "optimal"
"""
import csv

def load_results(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return [{**r,
                 'best_test_acc': float(r['best_test_acc']),
                 'train_time_s': float(r['train_time_s'])}
                for r in reader]

def print_best_accuracies(results, tolerance=1.0):
    max_acc = max(r['best_test_acc'] for r in results)

    # Retrieve candidate configurations within threshold
    candidates = [r for r in results if r['best_test_acc'] >= max_acc - tolerance]

    # Print range of accuracies kept
    print(f"Max acc: {max_acc:.2f}%, Lowest acc: {max_acc - tolerance:.2f}%")

    # Print candidate configs and results for analyzing
    print(f"Candidate configs ({len(candidates)}):")
    for r in sorted(candidates, key=lambda r: r['best_test_acc']):
        print(f"  {r['label']:40s} acc={r['best_test_acc']:.2f}%  time={r['train_time_s']}s  params={r['param_count']}")

# Load results of experiment from csv file
results = load_results('experiment/results.csv')

# Print the best model configurations with a tolerance drop off in test accuracy of 2.5
print_best_accuracies(results, tolerance=2.5)