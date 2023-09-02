# Project Name

## Description

This is up to date PyTorch implementation for Graph classification task with graph neural networks(GNN) based on GIN(graph isomorphims u-nets) mainly based on the paper "Graph isomorphism UNet" which is written by Alireza Amouzad, Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Behnam Roshanfekr from Amirkabir University of Technology.

## Table of Contents

- [Usage](#usage)
## Usage

This repository is seen as a pipeline for training and evaluating graph neural network models on graph classification tasks by using graph isomorphism networks(GIN). An overview of what this project accomplishes is provided below:

Data Loading and Preprocessing: Graph datasets (MUTAG, ENZYMES, PROTEINS) are loaded using PyTorch Geometric's TUDataset. Subsequently, the datasets are divided into training and test sets, and data loaders are created for batch processing.

Model Definition: Several graph neural network models, such as GINModel, SimpleGraphUNet, GraphUNetTopK, GIUNetSpect, GIUNetCent, etc., have been defined using PyTorch Geometric's GINConv, TopKPooling, global pooling, and custom pooling layers.

Training and Testing: The models are trained and tested using a 300 epochs. During the training phase, the loss is computed, and model parameters are updated. Additionally, training and testing accuracy and loss are monitored over the course of epochs.

Logging: The training and testing metrics, including loss and accuracy, are recorded over each epoch, and these logs are saved to CSV files.

Visualization: Scripts for visualizing training and testing metrics using matplotlib have been developed. Separate plots for accuracy and loss are generated for each dataset and model combination.

Embedding Visualization: Code for visualizing the embeddings learned by the models using t-SNE is available. The dimensionality of embeddings is reduced to either 2D or 3D, and they are visualized along with their labels.

Model Saving: The state_dict of the best-performing model is saved to disk for subsequent use or evaluation.

Overall, this code is regarded as clear and well-documented. Best practices for organizing machine learning experiments are followed.
