"""
Едноставна GCN архитектура за класификација на цели графови (AD vs CN).

Тек на податоци:
  влезен граф (19 јазли, 4 features секој)
    -> GCNConv слој 1 (message passing од непосредни соседи)
    -> GCNConv слој 2 (message passing од "соседи на соседите")
    -> global mean pooling (19 јазли -> 1 вектор за целиот граф)
    -> Linear класификатор (AD или CN)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import config


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=None, num_classes=2):
        """
        Параметри
        ---------
        num_node_features : int
            колку features има секој јазол (кај нас: 4 - delta/theta/alpha/beta)
        hidden_dim : int
            големина на "скриените" вектори по секој GCN слој
        num_classes : int
            колку класи класифицираме (кај нас: 2 - AD и CN)
        """
        super().__init__()
        hidden_dim = hidden_dim or config.GCN_HIDDEN_DIM

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        """
        x : node features, облик (n_nodes_во_batch, num_node_features)
        edge_index : облик (2, n_edges_во_batch)
        edge_weight : облик (n_edges_во_batch,) - connectivity magnitude тежини
        batch : облик (n_nodes_во_batch,) - кој јазол на кој граф припаѓа
        """
        # Message passing (direct neighbours)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Message passing (neighbours of neighbours / second hop)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Global pooling: 19 јазли по граф -> 1 вектор по граф
        x = global_mean_pool(x, batch)  # облик: (n_графови_во_batch, hidden_dim)

        # Финална класификација
        out = self.classifier(x)  # облик: (n_графови_во_batch, num_classes)
        return out
