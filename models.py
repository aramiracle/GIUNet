import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GINConv

    
class PoolingLayer(nn.Module):
    def forward(self, x, batch):
        # Calculate mean pooling across batch dimension
        batch_size = batch.max() + 1
        sum_pool = torch.zeros(batch_size, x.size(1), device=x.device)
        count_pool = torch.zeros(batch_size, 1, device=x.device)

        for i in range(batch_size):
            mask = (batch == i)
            sum_pool[i] = torch.sum(x[mask], dim=0)
            count_pool[i] = torch.sum(mask)

        pooled_x = sum_pool / count_pool
        return pooled_x

class UnpoolingLayer(nn.Module):
    def forward(self, x_pooled, batch, num_nodes):
        # Upsample by broadcasting
        unpooled_x = x_pooled[batch]  # Broadcast pooled features to nodes
        return unpooled_x

class GIUNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNet, self).__init__()

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

        self.pool1 = PoolingLayer()  # Separate pooling layer 1

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

        self.unpool1 = UnpoolingLayer()  # Separate unpooling layer 1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Downward path
        x1 = self.downconv1(x, edge_index)
        x2 = self.downconv2(x1, edge_index)

        # First pooling layer
        x_pooled1 = self.pool1(x2, batch)

        # Upward path
        x_up1 = torch.cat([x, x2], dim=1)
        x_up1 = self.upconv1(x_up1, edge_index)

        x_up2 = torch.cat([x_up1, x2], dim=1)
        x_up2 = self.upconv2(x_up2, edge_index)

        # First unpooling layer
        x_unpooled1 = self.unpool1(x_pooled1, batch, x.size(0))

        return x_unpooled1
    
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

        # Pooling layer
        x_pooled = torch_geometric.nn.global_mean_pool(x_up2, batch)

        return x_pooled