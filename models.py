import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GINConv

class GINModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINModel, self).__init__()

        self.downconv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))

        self.downconv2 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))

        self.upconv1 = GINConv(nn.Sequential(
            nn.Linear(64 + num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))

        self.upconv2 = GINConv(nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        ))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Downward path
        x1 = self.downconv1(x, edge_index)
        x2 = self.downconv2(x1, edge_index)

        # Upward path
        x_up1 = torch.cat([x, x2], dim=1)
        x_up1 = self.upconv1(x_up1, edge_index)

        x_up2 = torch.cat([x_up1, x2], dim=1)
        x_up2 = self.upconv2(x_up2, edge_index)

        x_up2 = torch_geometric.nn.global_mean_pool(x_up2, batch)

        return x_up2