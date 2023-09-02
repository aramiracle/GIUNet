import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from models import GINModel, SimpleGraphUNet, GraphUNetTopK
from tqdm import tqdm

def visualize_embeddings(model, test_loader, dataset_name, model_name, save_dir):
    # Collect graph embeddings and labels
    embeddings = []
    labels = []
    for data in test_loader:
        model.eval()
        with torch.no_grad():
            output = model(data)
        embeddings.extend(output.cpu().numpy())
        labels.extend(data.y.cpu().numpy())

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    if embeddings.shape[1] >= 3:
        # If the number of features is 3 or more, visualize in 3D
        visualize_embeddings_3d(embeddings, labels, dataset_name, model_name, save_dir)
    else:
        # Otherwise, visualize in 2D
        visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir)

def visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir):
    # Apply t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.get_cmap('tab10', num_classes)

    # Visualize the 2D embeddings with discrete labels
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=colors(i), label=f'Class {label}')

    plt.legend()
    plt.title(f't-SNE 2D Visualization of Graph Embeddings ({model_name} - {dataset_name})')
    
    # Save the figure
    save_path = os.path.join(save_dir, f'tsne_2d_{model_name}_{dataset_name}.png')
    plt.savefig(save_path)
    plt.show()

def visualize_embeddings_3d(embeddings, labels, dataset_name, model_name, save_dir):
    # Check if there are at least three features for 3D visualization
    if embeddings.shape[1] >= 3:
        # Apply PCA to reduce dimensionality to 3D
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)

        # Apply t-SNE for further dimensionality reduction to 3D
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_3d)

        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.get_cmap('tab10', num_classes)

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                       c=colors(i), label=f'Class {label}')

        ax.legend()
        ax.set_title(f't-SNE 3D Visualization of Graph Embeddings ({model_name} - {dataset_name})')
        
        # Save the figure
        save_path = os.path.join(save_dir, f'tsne_3d_{model_name}_{dataset_name}.png')
        plt.savefig(save_path)
        plt.show()
    else:
        print("The dataset has fewer than three features. Switching to 2D visualization.")
        visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir)

def main():
    # Define the list of models and datasets
    model_names = ['GINModel', 'SimpleGraphUNet', 'GraphUNetTopK']
    dataset_names = ['MUTAG', 'ENZYMES', 'PROTEINS']  # Update with your dataset names

    for model_name in model_names:
        for dataset_name in dataset_names:
            dataset_dir = os.path.join('datasets', dataset_name)
            model_results_dir = os.path.join('results', model_name)
            embedding_results_dir = os.path.join('results', 'embedding', model_name)

            # Load the dataset
            dataset = TUDataset(root=dataset_dir, name=dataset_name)
            num_classes = dataset.num_classes
            num_features = dataset.num_features

            # Use train_test_split to split dataset into train and test
            train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

            batch_size = 64
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the model, optimizer, and criterion
            if model_name == 'GINModel':
                model = GINModel(num_features, num_classes)
            elif model_name == 'SimpleGraphUNet':
                model = SimpleGraphUNet(num_features, num_classes)
            elif model_name == 'GraphUNetTopK':
                model = GraphUNetTopK(num_features, num_classes)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            model.load_state_dict(torch.load(os.path.join(model_results_dir, f'best_model_{dataset_name}.pth')))
            model.eval()

            # Create the embedding results directory if it doesn't exist
            if not os.path.exists(embedding_results_dir):
                os.makedirs(embedding_results_dir)

            # Specify the directory to save figures
            save_figure_dir = os.path.join(embedding_results_dir, 'figures')

            # Create the figure directory if it doesn't exist
            if not os.path.exists(save_figure_dir):
                os.makedirs(save_figure_dir)

            # Visualize and save graph embeddings
            visualize_embeddings(model, test_loader, dataset_name, model_name, save_figure_dir)

if __name__ == '__main__':
    main()
