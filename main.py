import os
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import *
from utils import *


# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# List of model names]
model_list = ['GINModel', 'SimpleGraphUNet', 'GraphUNetTopK']

# Load and preprocess the MUTAG dataset
dataset_list = ['MUTAG','ENZYMES','PROTEINS']

for model_name in model_list:
    model_results_dir = os.path.join('results', model_name)
    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)

    for dataset_name in dataset_list:
        dataset_dir = os.path.join('datasets', dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset = TUDataset(root=dataset_dir, name=dataset_name)
        num_classes = dataset.num_classes
        num_features = dataset.num_features

        # Split dataset into train and test
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model, optimizer, criterion = create_model(model_name, num_features, num_classes)

        train_and_test_model(model, optimizer, criterion, train_loader, test_loader, model_results_dir, dataset_name)
