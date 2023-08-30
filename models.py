import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,TopKPooling, global_max_pool, global_mean_pool
import networkx as nx


import numpy as np
import networkx as nx
import torch
import torch.nn as nn

class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.centralities_num = 6
        self.sigmoid = nn.Sigmoid()
        self.feature_proj = nn.Linear(in_dim, 1)
        self.structure_proj = nn.Linear(self.centralities_num, 1)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        G = nx.from_numpy_array(g.detach().numpy())
        C = all_centralities(G).float()
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(C)
        weights = self.final_proj(torch.cat([feature_weights, structure_weights], dim=1)).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)

class Unpool(nn.Module):
    def forward(self, g, h, pre_h, idx):
        new_h = torch.zeros_like(h)
        new_h[idx] = h[idx]
        idx_prime = torch.tensor([i for i in idx if i >= g.shape[0]])
        for i in idx_prime:
            new_h[i] = (h * g[i, :]).sum(0) / g[i, :].sum()
        return g, new_h

def centrality_based(centrality_metric, graph):
    if centrality_metric in ['closeness', 'degree', 'eigenvector', 'betweenness', 'load', 'subgraph', 'harmonic']:
        return torch.tensor(list(nx.algorithms.centrality.__getattribute__(centrality_metric)(graph).values()))
    else:
        raise ValueError("Unknown centrality metric")

def all_centralities(graph):
    return torch.stack([centrality_based(metric, graph) for metric in ['closeness', 'degree', 'betweenness', 'load', 'subgraph', 'harmonic']], dim=1)

def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = torch.matmul(g.bool().float(), torch.matmul(g.bool().float(), g.bool().float())).bool().float()
    un_g = un_g[idx, :][:, idx]
    g = norm_g(un_g)
    return g, new_h, idx

def norm_g(g):
    return g / (g.sum(1, keepdim=True) + 1e-8)


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
        self.pool1 = Pool(32, ratio=0.8)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ))
        self.pool2 = Pool(64, ratio=0.8)
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ))
        self.pool3 = Pool(128, ratio=0.8)

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

        return x_d1


class GraphUNet(nn.Module):
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