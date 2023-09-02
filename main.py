
from torch_geometric.data import DataLoader
from models import *
from utils import *


def main():
    create_results_directory()

    model_list = ['GINModel', 'SimpleGraphUNet', 'GraphUNetTopK']
    dataset_list = ['MUTAG', 'ENZYMES', 'PROTEINS']

    for model_name in model_list:
        model_results_dir = create_model_results_directory(model_name)

        for dataset_name in dataset_list:
            dataset, num_features, num_classes = preprocess_dataset(dataset_name)
            train_dataset, test_dataset = split_dataset(dataset)

            batch_size = 64
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model, optimizer, criterion = create_model(model_name, num_features, num_classes)

            train_and_test_model(model, optimizer, criterion, train_loader, test_loader, model_results_dir, dataset_name)

if __name__ == "__main__":
    main()
