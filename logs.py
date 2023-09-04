import matplotlib.pyplot as plt
import pandas as pd
import os

dataset_list = ['MUTAG', 'ENZYMES', 'PROTEINS']
models_list = ['GraphUNetTopK', 'SimpleGraphUNet', 'GINModel']
num_datasets = len(dataset_list)
directory = './results'

# Create a new figure
plt.figure(figsize=(16, 6 * num_datasets))

for model in models_list:
    for i, dataset in enumerate(dataset_list):
        model_results_dir = os.path.join('results', model)
        log_file_path = os.path.join(model_results_dir, f'logs_for_{dataset}.csv')
        log_df = pd.read_csv(log_file_path)

        # Extract data from the DataFrame
        epochs = log_df['Epoch']
        train_accuracies = log_df['Train Accuracy']
        test_accuracies = log_df['Test Accuracy']
        train_losses = log_df['Train Loss']
        test_losses = log_df['Test Loss']

        # Plotting
        plt.subplot(num_datasets, 2, 2 * i + 1)
        plt.plot(epochs, train_accuracies, label='Train-' + model)
        plt.plot(epochs, test_accuracies, label='Test-' + model)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset} - Train and Test Accuracies')
        plt.legend()

        plt.subplot(num_datasets, 2, 2 * i + 2)
        plt.plot(epochs, train_losses, label='Train-' + model)
        plt.plot(epochs, test_losses, label='Test-' + model)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{dataset} - Train and Test Losses')
        plt.legend()

    plt.tight_layout()
visulization_file_path = os.path.join('results', 'logs_visualization')
plt.savefig(visulization_file_path)
plt.show()