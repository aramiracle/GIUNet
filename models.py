import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GINConv

    

class PoolingLayer(nn.Module):
    def forward(self, x, edge_index):
        pooled_x = x.mean(dim=0)
        return pooled_x


class UnpoolingLayer(nn.Module):
    def forward(self, x_pooled, num_nodes):
        return x_pooled.unsqueeze(0).repeat(num_nodes, 1)


class GIUNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNet, self).__init__()

        self.conv1 = gnn.GINConv(nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.conv2 = gnn.GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ))
        self.conv3 = gnn.GINConv(nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ))

        self.pool = PoolingLayer()
        self.unpool = UnpoolingLayer()

        self.middle_conv = gnn.GINConv(nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ))

        self.upconv1 = gnn.GINConv(nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ))
        self.upconv2 = gnn.GINConv(nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.upconv3 = gnn.GINConv(nn.Sequential(
            nn.Linear(64 + num_features, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
        ))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Encoder path
        x1 = self.conv1(x, edge_index)
        x_pooled1 = self.pool(x1, edge_index)
        x2 = self.conv2(x_pooled1, edge_index)
        x_pooled2 = self.pool(x2, edge_index)
        x3 = self.conv3(x_pooled2, edge_index)
        x_pooled3 = self.pool(x3, edge_index)

        # Middle convolution
        x_middle = self.middle_conv(x_pooled3, edge_index)

        # Decoder path
        x_up1 = self.unpool(x_middle, num_nodes=x3.size(0))
        x_up1 = self.upconv1(torch.cat([x_up1, x2], dim=1), edge_index)
        x_up2 = self.unpool(x_up1, num_nodes=x2.size(0))
        x_up2 = self.upconv2(torch.cat([x_up2, x1], dim=1), edge_index)
        x_up3 = self.unpool(x_up2, num_nodes=x1.size(0))
        x_up3 = self.upconv3(torch.cat([x_up3, x], dim=1), edge_index)

        return x_up3

    
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