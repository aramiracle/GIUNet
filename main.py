import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from models import *
from utils import *


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
            run_results = []  # Store results for each run

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

                run_results.append({'Run': run + 1, 'Accuracy': fold_accuracy})

                print(f"Model: {model_name}, Dataset: {dataset_name}, Run {run + 1} Accuracy: {fold_accuracy:.4f}")

            # Calculate confidence intervals for this model-dataset combination
            mean_accuracy, lower_bound, upper_bound = calculate_confidence_interval([run['Accuracy'] for run in run_results])

            summary_results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Mean Accuracy': mean_accuracy,
                'Confidence Interval (95%)': [lower_bound, upper_bound],
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Runs': run_results
            })

    # Write and save the summary
    write_and_save_summary(summary_results)

if __name__ == "__main__":
    main()