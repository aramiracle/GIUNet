import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from models import *
from utils import *
import scipy.stats as stats

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval of the data.
    :param data: List of values.
    :param confidence: Desired confidence level.
    :return: Tuple of (mean, lower bound, upper bound)
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    n = len(data)
    z = stats.t.ppf((1 + confidence) / 2, df=n - 1)  # Calculate z-score for given confidence level
    margin = z * (std / np.sqrt(n))
    lower_bound = mean - margin
    upper_bound = mean + margin
    return mean, lower_bound, upper_bound

def main():
    create_results_directory()

    model_list = ['SimpleGraphUNet', 'GraphUNetTopK', 'GINModel']
    dataset_list = ['MUTAG', 'ENZYMES', 'PROTEINS']
    num_runs = 10  # Number of runs with different random seeds

    summary_results = []

    for model_name in model_list:
        model_results_dir = create_model_results_directory(model_name)

        for dataset_name in dataset_list:
            dataset, num_features, num_classes = preprocess_dataset(dataset_name)
            train_dataset, test_dataset = split_dataset(dataset)

            batch_size = 64
            epochs = 300
            max_fold_accuracies = []  # Store maximum accuracy for each run

            for run in range(num_runs):
                # Set a different random seed for each run
                random_seed = run + 1
                torch.manual_seed(random_seed)
                random.seed(random_seed)
                np.random.seed(random_seed)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                model, optimizer, criterion = create_model(model_name, num_features, num_classes)

                fold_accuracy = train_and_test_model(model, optimizer, criterion, train_loader, test_loader, model_results_dir, dataset_name, epochs)

                max_fold_accuracies.append(fold_accuracy)

                print(f"Model: {model_name}, Dataset: {dataset_name}, Run {run + 1} Accuracy: {fold_accuracy:.4f}")

            # Calculate the average of the maximum accuracies for all runs
            average_max_accuracy = sum(max_fold_accuracies) / num_runs

            # Calculate confidence interval
            mean, lower_bound, upper_bound = calculate_confidence_interval(max_fold_accuracies)

            summary_results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Average Accuracy': average_max_accuracy,
                'Confidence Interval (95%)': [lower_bound, upper_bound]
            })

    # Print summary results
    print("\nSummary Results:")
    for result in summary_results:
        print(f"Model: {result['Model']}, Dataset: {result['Dataset']}")
        print(f"Average Accuracy: {result['Average Accuracy']:.4f}")
        print(f"Confidence Interval (95%): [{result['Confidence Interval (95%)'][0]:.4f}, {result['Confidence Interval (95%)'][1]:.4f}]")
        print()

if __name__ == "__main__":
    main()
