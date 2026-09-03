"""
Simple GCN architecture for whole-graph classification (AD vs CN).

Data flow:
  input graph (19 nodes, 4 features each)
    -> GCNConv layer 1 (message passing from direct neighbours)
    -> GCNConv layer 2 (message passing from neighbours of neighbours)
    -> global mean pooling (19 nodes -> 1 vector for the whole graph)
    -> linear classifier (AD or CN)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import config


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=None, num_classes=2):
        """
        Parameters
        ---------
        num_node_features : int
            number of features per node (here: 4 - delta/theta/alpha/beta)
        hidden_dim : int
            size of the hidden vectors after each GCN layer
        num_classes : int
            number of classes to classify (here: 2 - AD and CN)
        """
        super().__init__()
        hidden_dim = hidden_dim or config.GCN_HIDDEN_DIM

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        """
        x : node features, shape (n_nodes_in_batch, num_node_features)
        edge_index : shape (2, n_edges_in_batch)
        edge_weight : shape (n_edges_in_batch,) - connectivity magnitude weights
        batch : shape (n_nodes_in_batch,) - graph assignment for each node
        """
        # Message passing (direct neighbours)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Message passing (neighbours of neighbours / second hop)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Global pooling: 19 nodes per graph -> 1 vector per graph
        x = global_mean_pool(x, batch)  # shape: (n_graphs_in_batch, hidden_dim)

        # Final classification
        out = self.classifier(x)  # shape: (n_graphs_in_batch, num_classes)
        return out
