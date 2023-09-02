import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,TopKPooling, global_max_pool, global_mean_pool
import networkx as nx
from methods import *


def make_convolution(in_channels, out_channels):
    return GINConv(nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ))

# Define a pooling layer for centrality features
class CentPool(nn.Module):
    def __init__(self, in_dim, ratio, p):
        super(CentPool, self).__init__()
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()
        self.feature_proj = nn.Linear(in_dim, 1)
        self.structure_proj = nn.Linear(4, 1)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, edge_index, h):
        Z = self.drop(h)
        G = edge_index_to_nx_graph(edge_index, h.shape[0])
        C = all_centralities(G)
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(C)
        weights = self.final_proj(torch.cat([feature_weights, structure_weights], dim=1)).squeeze()  # Combine and project weights
        scores = self.sigmoid(weights)
        g, h, idx = top_k_pool(scores, edge_index, h, self.ratio)
        edge_index = edge_index[:, idx]
        return g, h, idx, edge_index
    
# Define a pooling layer for spectral features
class SpectPool(nn.Module):
    def __init__(self, in_dim, ratio, p):
        super(SpectPool, self).__init__()
        self.ratio = ratio
        self.eigs_num = 3
        self.sigmoid = nn.Sigmoid()
        self.feature_proj = nn.Linear(in_dim, 1)
        self.structure_proj = nn.Linear(self.eigs_num, 1)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, edge_index, h):
        Z = self.drop(h)
        G = edge_index_to_nx_graph(edge_index, h.shape[0])
        L = normalized_laplacian(G)
        L_a = approximate_matrix(L, self.eigs_num)
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(L_a)
        weights = self.final_proj(torch.cat([feature_weights, structure_weights], dim=1)).squeeze()  # Combine and project weights
        scores = self.sigmoid(weights)
        g, h, idx = top_k_pool(scores, edge_index, h, self.ratio)
        edge_index = edge_index[:, idx]
        return g, h, idx, edge_index


class SimpleUnpool(nn.Module):
    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return new_h

class Unpool(nn.Module):
    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        idx_prime = torch.tensor([index for index in idx if index not in range(g.shape[0])])
        
        for i in idx_prime:
            normalized_idx = idx.float() / g[i].sum()  # Normalize indices
            weighted_mean = torch.sum(g[i][i] * normalized_idx)  # Compute weighted mean
            new_h[i] = weighted_mean * h
       
        return new_h
    
#Creating model that uses centralities
class GIUNetSpect(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNetSpect, self).__init__()

        self.conv1 = make_convolution(num_features, 32)
        self.pool1 = CentPool(32, ratio=0.8, p=0.5)  # Custom pooling layer

        self.conv2 = make_convolution(32, 64)
        self.pool2 = CentPool(64, ratio=0.8, p=0.5)  # Custom pooling layer

        self.midconv = make_convolution(64, 64)

        self.decoder2 = make_convolution(64, 32)
        self.decoder1 = nn.Linear(32, num_classes)  # Final classification layer

        self.unpool2 = SimpleUnpool()  # Unpool layer after decoder2
        self.unpool1 = SimpleUnpool()  # Unpool layer after decoder1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder
        x1 = F.relu(self.conv1(x, edge_index))
        g1, x1_pooled, idx1, edge_index1 = self.pool1(edge_index, x1)

        x2 = F.relu(self.conv2(x1_pooled, edge_index1))
        _, x2_pooled, idx2, edge_index2 = self.pool2(edge_index1, x2)

        # Middle Convolution
        x_m = F.relu(self.midconv(x2_pooled, edge_index2))

        # Decoder
        x_d2 = self.unpool2(g1, x_m, idx2)
        x_d2 = F.relu(self.decoder2(x_d2, edge_index2))

        x_d1 = self.unpool1(adjacency_matrix(edge_index), x_d2, idx1)
        x_d1 = F.relu(self.decoder1(x_d1))

        x_global_pool = global_mean_pool(x_d1, batch)

        return x_global_pool

class GIUNetCent(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIUNetCent, self).__init__()

        self.conv1 = make_convolution(num_features, 32)
        self.pool1 = CentPool(32, ratio=0.8, p=0.5)  # Custom pooling layer

        self.conv2 = make_convolution(32, 64)
        self.pool2 = CentPool(64, ratio=0.8, p=0.5)  # Custom pooling layer

        self.midconv = make_convolution(64, 64)

        self.decoder2 = make_convolution(64, 32)
        self.decoder1 = nn.Linear(32, num_classes)  # Final classification layer

        self.unpool2 = Unpool()  # Unpool layer after decoder2
        self.unpool1 = Unpool()  # Unpool layer after decoder1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x1 = F.relu(self.conv1(x, edge_index))
        g1, x1_pooled, idx1, edge_index1 = self.pool1(edge_index, x1)
        
        x2 = F.relu(self.conv2(x1_pooled, edge_index1))
        _, x2_pooled, idx2, edge_index2 = self.pool2(edge_index1, x2)
        
        x_m = F.relu(self.midconv(x2_pooled, edge_index2))
        
        x_d2 = self.unpool2(g1, x_m, idx2)
        x_d2 = F.relu(self.decoder2(x_d2, edge_index2))
        
        x_d1 = self.unpool1(adjacency_matrix(edge_index), x_d2, idx1)
        x_d1 = F.relu(self.decoder1(x_d1))

        x_global_pool = global_mean_pool(x_d1, batch)

        return x_global_pool


class GraphUNetTopK(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphUNetTopK, self).__init__()

        self.conv1 = make_convolution(num_features, 32)
        self.pool1 = TopKPooling(32, ratio=0.8)
        self.conv2 = make_convolution(32, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = make_convolution(64, 128)
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
        self.conv1 = make_convolution(num_features, 64)
        self.conv2 = make_convolution(64, 128)
        self.conv3 = make_convolution(128, 256)

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

        self.downconv1 = make_convolution(num_features, 64)
        self.downconv2 = make_convolution(64, 64)
        self.upconv1 = make_convolution(64 + num_features, 64)
        self.upconv2 = make_convolution(64 + 64, 64)
        self.final_layer = nn.Linear(64, num_classes)

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

        # Final classification layer
        x_final = self.final_layer(x_up2)

        x_global_pool = global_mean_pool(x_final, batch)

        return x_global_pool