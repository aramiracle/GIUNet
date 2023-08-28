import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import GINModel
from utils import test

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Load and preprocess the MUTAG dataset
dataset_list = ['MUTAG', 'ENZYMES', 'PROTEINS']
for dataset_name in dataset_list:
    dataset = TUDataset(root=dataset_name+'_Dataset', name=dataset_name)
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    # Split dataset into train and test
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model, optimizer, and criterion
    model = GINModel(num_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Training loop
    # Initialize variables for tracking max accuracy
    max_test_accuracy = 0.0
    best_model_state = None

    # Create a list to store logs
    logs = []

    # Training loop
    epochs = 500
    for epoch in range(epochs):
        # Train the model
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == data.y).sum().item()
            total_samples += data.y.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples

        # Test the model
        model.eval()
        test_loss, test_accuracy = test(model, test_loader, criterion)

        # Update max test accuracy
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join('results', 'best_model_'+dataset_name+'.pth'))
            print(f'Best model saved with Test Accuracy: {test_accuracy:.4f}')
        # Append logs
        logs.append({
        'Epoch': epoch + 1,
        'Train Loss': f'{train_loss:.4f}',
        'Train Accuracy': f'{train_accuracy:.4f}',
        'Test Loss': f'{test_loss:.4f}',
        'Test Accuracy': f'{test_accuracy:.4f}'
    })

        print(f'Epoch {epoch+1}/{epochs}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

    # Convert logs to a DataFrame
    log_df = pd.DataFrame(logs)

    # Save logs to a CSV file
    log_file_path = os.path.join('results', 'logs_for_'+dataset_name+'.csv')
    log_df.to_csv(log_file_path, index=False)

    print(f'Max Test Accuracy: {max_test_accuracy:.4f}')
