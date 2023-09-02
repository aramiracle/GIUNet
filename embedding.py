import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from models import GINModel, SimpleGraphUNet, GraphUNetTopK

def create_results_directory():
    if not os.path.exists('results'):
        os.makedirs('results')

def create_embedding_results_directory(model_name):
    embedding_results_dir = os.path.join('results', 'embedding', model_name)
    if not os.path.exists(embedding_results_dir):
        os.makedirs(embedding_results_dir)
    return embedding_results_dir

def load_and_preprocess_dataset(dataset_name):
    dataset_dir = os.path.join('datasets', dataset_name)
    dataset = TUDataset(root=dataset_dir, name=dataset_name)
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    return dataset, num_features, num_classes

def create_data_loaders(dataset, batch_size=64):
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_and_evaluate_model(model_name, num_features, num_classes, model_results_dir, dataset_name):
    model = globals()[model_name](num_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(model_results_dir, f'best_model_{dataset_name}.pth')))
    model.eval()
    return model

def visualize_and_save_embeddings(model, test_loader, dataset_name, model_name, save_dir):
    embeddings, labels = get_embeddings_and_labels(model, test_loader)
    visualize_embeddings(embeddings, labels, dataset_name, model_name, save_dir)

def get_embeddings_and_labels(model, test_loader):
    embeddings = []
    labels = []
    for data in test_loader:
        model.eval()
        with torch.no_grad():
            output = model(data)
        embeddings.extend(output.cpu().numpy())
        labels.extend(data.y.cpu().numpy())
    return np.array(embeddings), np.array(labels)

def visualize_embeddings(embeddings, labels, dataset_name, model_name, save_dir):
    if embeddings.shape[1] >= 3:
        visualize_embeddings_3d(embeddings, labels, dataset_name, model_name, save_dir)
    else:
        visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir)

def visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.get_cmap('tab10', num_classes)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=colors(i), label=f'Class {label}')

    plt.legend()
    plt.title(f't-SNE 2D Visualization of Graph Embeddings ({model_name} - {dataset_name})')

    save_path = os.path.join(save_dir, f'tsne_2d_{model_name}_{dataset_name}.png')
    plt.savefig(save_path)
    plt.show()

def visualize_embeddings_3d(embeddings, labels, dataset_name, model_name, save_dir):
    if embeddings.shape[1] >= 3:
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.get_cmap('tab10', num_classes)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                       c=colors(i), label=f'Class {label}')

        ax.legend()
        ax.set_title(f't-SNE 3D Visualization of Graph Embeddings ({model_name} - {dataset_name})')

        save_path = os.path.join(save_dir, f'tsne_3d_{model_name}_{dataset_name}.png')
        plt.savefig(save_path)
        plt.show()
    else:
        print("The dataset has fewer than three features. Switching to 2D visualization.")
        visualize_embeddings_2d(embeddings, labels, dataset_name, model_name, save_dir)

def main():
    create_results_directory()
    model_names = ['GINModel', 'SimpleGraphUNet', 'GraphUNetTopK']
    dataset_names = ['MUTAG', 'ENZYMES', 'PROTEINS']

    for model_name in model_names:
        for dataset_name in dataset_names:
            embedding_results_dir = create_embedding_results_directory(model_name)
            dataset, num_features, num_classes = load_and_preprocess_dataset(dataset_name)
            test_loader = create_data_loaders(dataset)

            model_results_dir = os.path.join('results', model_name)
            model = load_and_evaluate_model(model_name, num_features, num_classes, model_results_dir, dataset_name)

            visualize_and_save_embeddings(model, test_loader, dataset_name, model_name, embedding_results_dir)

if __name__ == '__main__':
    main()
