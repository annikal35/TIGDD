
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from TGRankEncoder.TimeEncode import TimeEncoder
from TGRankEncoder.MLP import MLP
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import numpy as np
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class TSARLayer(nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_size, reduce='sum'):
        super(TSARLayer, self).__init__()
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.edge_time_emb_size = edge_time_emb_size
        self.reduce = reduce

        # Define the message function
        self.msg_function = nn.Linear(self.emb_dim + self.edge_attr_size + self.edge_time_emb_size, self.emb_dim, bias=True)
        self.linear = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.emb_dim, elementwise_affine=True, eps=1e-5)
        self.layer_relation_embedding = nn.Parameter(torch.rand(self.emb_dim), requires_grad=True)

    def forward(self, feature_view, edge_index, edge_attr, edge_time_emb, boundary_condition):
        """
        Forward pass for the TSARLayer.

        Args:
            feature_view (Tensor): Node features tensor of shape [num_nodes, emb_dim].
            edge_index (Tensor): Edge indices tensor of shape [2, num_edges].
            edge_attr (Tensor): Edge attributes tensor of shape [num_edges, edge_attr_size].
            edge_time_emb (Tensor): Edge time embeddings tensor of shape [num_edges, edge_time_emb_size].
            boundary_condition (Tensor): Boundary condition tensor of shape [num_nodes, emb_dim].

        Returns:
            Tensor: Updated node features tensor of shape [num_nodes, emb_dim].
        """
        # Extract source and target node indices
        src = edge_index[0]
        
        # Message computation
        msg = feature_view[src]
        if edge_time_emb is not None:
            msg = torch.cat((msg, edge_attr, edge_time_emb), dim=1)
        else:
            msg = torch.cat((msg, edge_attr), dim=1)
        
        msg = F.relu(self.msg_function(msg))
        
        # Augment with boundary condition
        msg_aug = torch.cat([msg, boundary_condition], dim=1)

        # Aggregate messages
        out = scatter(msg_aug, edge_index[1], dim=0, reduce=self.reduce, dim_size=feature_view.size(0))

        # Apply linear transformation, normalization, and dropout
        out = self.linear(out)
        out = F.dropout(F.relu(self.layer_norm(out)), p=0.1, training=self.training)

        return out



class TSARNet(nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_dim=128, num_layers=3, device=None):
        super(TSARNet, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.num_layers = num_layers
        self.edge_time_emb_dim = edge_time_emb_dim

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TSARLayer(self.emb_dim, self.edge_attr_size, self.edge_time_emb_dim, reduce='sum'))

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1)  # Adjust the output dimension as needed
        )

    def forward(self, feature_view, edge_index, edge_attr, edge_time_emb, boundary_condition, batch):
        """
        Forward pass for the TSARNet.

        Args:
            feature_view (Tensor): Node features tensor of shape [num_nodes, emb_dim].
            edge_index (Tensor): Edge indices tensor of shape [2, num_edges].
            edge_attr (Tensor): Edge attributes tensor of shape [num_edges, edge_attr_size].
            edge_time_emb (Tensor): Edge time embeddings tensor of shape [num_edges, edge_time_emb_size].
            boundary_condition (Tensor): Boundary condition tensor of shape [num_nodes, emb_dim].
            batch (Tensor): Batch tensor used for pooling.

        Returns:
            Tensor: Output tensor after processing through the network.
        """
        x = feature_view

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_time_emb, boundary_condition)

        # Pooling layer to get graph-level representation
        x = global_mean_pool(x, batch)

        # Final prediction
        x = self.mlp(x)

        return x


