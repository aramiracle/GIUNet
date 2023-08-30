import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,TopKPooling, global_max_pool, global_mean_pool
import networkx as nx


class GraphPoolingLayer(nn.Module):
    def __init__(self, in_channels, ratio):
        super(GraphPoolingLayer, self).__init__()
        self.pool = TopKPooling(in_channels, ratio=ratio)

    def forward(self, x, edge_index, batch):
        G = nx.Graph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        return x, edge_index, batch

def centrality_based(centrality_metric, graph):
    if centrality_metric in ['closeness', 'degree', 'eigenvector', 'betweenness', 'load', 'subgraph', 'harmonic']:
        centrality = nx.algorithms.centrality.__getattribute__(centrality_metric)(graph)
        return torch.tensor(list(centrality.values()))
    else:
        raise ValueError("Unknown centrality metric")

def all_centralities(graph):
    return torch.stack([
        centrality_based('closeness', graph),
        centrality_based('degree', graph),
        centrality_based('betweenness', graph),
        centrality_based('load', graph),
        centrality_based('subgraph', graph),
        centrality_based('harmonic', graph)
    ], dim=1)

class GraphUnpoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphUnpoolingLayer, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_pooled, x_previous):
        x_unpooled = F.relu(self.decoder(x_pooled + x_previous))
        return x_unpooled

class GraphUNet2(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphUNet2, self).__init__()

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        ))
        self.pool1 = GraphPoolingLayer(32, ratio=0.8)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ))
        self.pool2 = GraphPoolingLayer(64, ratio=0.8)
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ))
        self.pool3 = GraphPoolingLayer(128, ratio=0.8)

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.decoder1 = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch = self.pool1(x, edge_index, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, batch = self.pool2(x, edge_index, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, batch = self.pool3(x, edge_index, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x_d3 = F.relu(self.decoder3(x3))
        x_d2 = F.relu(self.decoder2(x_d3 + x2))
        x_d1 = F.log_softmax(self.decoder1(x_d2 + x1), dim=-1)

        return x_d1Norm1d(32),
           
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch = self.pool1(x, edge_index, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, batch = self.pool2(x, edge_index, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, batch = self.pool3(x, edge_index, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x_d3 = F.relu(self.decoder3(x3))
        x_d2 = F.relu(self.decoder2(x_d3 + x2))
        x_d1 = F.log_softmax(self.decoder1(x_d2 + x1), dim=-1)

        return x_d1


class GraphUNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphUNet, self).__init__()

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        ))
        self.pool1 = TopKPooling(32, ratio=0.8)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ))
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ))
        self.pool3 = TopKPooling(128, ratio=0.8)

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.decoder1 = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x_d3 = F.relu(self.decoder3(x3))
        x_d2 = F.relu(self.decoder2(x_d3 + x2))
        x_d1 = F.log_softmax(self.decoder1(x_d2 + x1), dim=-1)

        return x_d1
    
class SimpleGraphUNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGraphUNet, self).__init__()

        # Encoder
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ))

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.decoder1 = nn.Linear(64, num_classes)  # Adjust if needed

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        x_d3 = self.decoder3(x3)
        x_d2 = self.decoder2(x_d3 + x2)  # Skip connection
        x_d1 = self.decoder1(x_d2 + x1)  # Skip connection

        x_global_pool = global_mean_pool(x_d1, batch)  # Global mean pooling
        
        return x_global_pool



    
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
        x_pooled = global_mean_pool(x_up2, batch)

        return x_pooled