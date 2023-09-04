import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for color palettes

# Load the summary results from the CSV file
csv_filename = 'results/summary_results.csv'
summary_df = pd.read_csv(csv_filename)

# List of unique models and datasets
models = summary_df['Model'].unique()
datasets = summary_df['Dataset'].unique()

# Define a color palette with distinct colors for each dataset
colors = sns.color_palette("Set1", n_colors=len(datasets))

# Set the figure size
plt.figure(figsize=(14, 6))

# Iterate through models
for i, model in enumerate(models):
    # Create a subplot for each model
    plt.subplot(1, len(models), i + 1)
    plt.title(f'Model: {model}')
    
    # Define lists to store mean accuracies and confidence intervals for each dataset
    mean_accuracies = []
    lower_errs = []
    upper_errs = []
    
    for dataset in datasets:
        dataset_df = summary_df[(summary_df['Model'] == model) & (summary_df['Dataset'] == dataset)]
        mean_accuracy = dataset_df['Mean Accuracy'].values[0]
        lower_bound = dataset_df['Lower Bound'].values[0]
        upper_bound = dataset_df['Upper Bound'].values[0]
        err_bot = mean_accuracy - lower_bound
        err_top = upper_bound - mean_accuracy

        mean_accuracies.append(mean_accuracy)
        lower_errs.append(err_bot)
        upper_errs.append(err_top)
    
    # Create bar plots for mean accuracies with error bars (confidence intervals) and use distinct colors
    plt.bar(datasets, mean_accuracies, yerr=(lower_errs, upper_errs), capsize=10, color=colors)

    # Add labels and set y-axis limit to 0-1
    plt.xlabel('Dataset')
    plt.ylabel('Mean Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
# Adjust layout and show the plot
plt.tight_layout()
visulization_file_path = os.path.join('visualization', 'comparison_of_rapid_models')
plt.savefig(visulization_file_path)
plt.show()
