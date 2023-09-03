from torch_geometric.data import DataLoader
from models import *
from utils import *
import random

def main():
    create_results_directory()

    model_list = ['SimpleGraphUNet']
    dataset_list = ['MUTAG']
    num_runs = 10  # Number of runs with different random seeds

    for model_name in model_list:
        model_results_dir = create_model_results_directory(model_name)

        for dataset_name in dataset_list:
            dataset, num_features, num_classes = preprocess_dataset(dataset_name)
            train_dataset, test_dataset = split_dataset(dataset)

            batch_size = 64
            fold_accuracies = []

            for run in range(num_runs):
                # Set a different random seed for each run
                random_seed = run + 1
                torch.manual_seed(random_seed)
                random.seed(random_seed)
                np.random.seed(random_seed)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                model, optimizer, criterion = create_model(model_name, num_features, num_classes)

                fold_accuracy = train_and_test_model(model, optimizer, criterion, train_loader, test_loader, model_results_dir, dataset_name)

                fold_accuracies.append(fold_accuracy)

                print(f"Run {run + 1} Accuracy: {fold_accuracy:.4f}")

            average_accuracy = sum(fold_accuracies) / num_runs
            print(f"Average Accuracy for {num_runs} runs with different seeds: {average_accuracy:.4f}")

if __name__ == "__main__":
    main()
