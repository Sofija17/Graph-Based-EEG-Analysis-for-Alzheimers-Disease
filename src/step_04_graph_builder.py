"""
Гради PyTorch Geometric Data објекти (графови) од:
  - node features (delta/theta/alpha/beta power по канал)
  - connectivity матрица (Pearson correlation, веќе threshold-ирана)

Секоја епоха -> еден граф.
"""

import numpy as np
import torch
from torch_geometric.data import Data

import config


def connectivity_matrix_to_edge_list(conn_matrix):
    """
    Ја конвертира квадратната connectivity матрица (n_channels x n_channels)
    во PyG формат: edge_index + edge_attr.

    Земаме само не-нула вредности (веќе threshold-ирани надвор од
    оваа функција), и ги игнорираме self-loops (дијагоналата)
    Параметри
    ---------
    conn_matrix : np.ndarray, облик (n_channels, n_channels)
                  веќе threshold-ирана (повеќето вредности се 0)

    Враќа
    -----
    edge_index : torch.Tensor, облик (2, n_edges)
    edge_attr : torch.Tensor, облик (n_edges,)
    """
    n_channels = conn_matrix.shape[0]

    sources = []
    targets = []
    signed_weights = []

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue  # прескокнуваме self-loops
            if conn_matrix[i, j] != 0:
                sources.append(i)
                targets.append(j)
                # Земаме апсолутна вредност: за јачина на функционална
                # конективност, важна е магнитудата на корелацијата, не
                # насоката. Дополнителна причина: GCNConv интерно прави
                # нормализација со квадратен корен од "степенот" на секој
                # јазол - негативни тежини можат да дадат негативен степен
                # -> sqrt(негативен број) -> NaN. Апсолутна вредност го
                # спречува ова.
                signed_weights.append(conn_matrix[i, j])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    signed_edge_attr = torch.tensor(signed_weights, dtype=torch.float)
    edge_attr = signed_edge_attr.abs()

    return edge_index, edge_attr, signed_edge_attr


def build_graph_for_epoch(node_features, conn_matrix, label):
    """
    Гради ЕДЕН PyG Data граф, за една епоха.

    Параметри
    ---------
    node_features : np.ndarray, облик (n_channels, n_bands)
                     пр. (19, 4) - delta/theta/alpha/beta по канал,
                     за оваа конкретна епоха
    conn_matrix : np.ndarray, облик (n_channels, n_channels)
                   веќе threshold-ирана connectivity матрица,
                   за оваа иста епоха
    label : int
             0 = Healthy Control, 1 = Alzheimer's Disease

    Враќа
    -----
    torch_geometric.data.Data објект
    """
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index, edge_attr, signed_edge_attr = connectivity_matrix_to_edge_list(conn_matrix)
    y = torch.tensor([label], dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        signed_edge_attr=signed_edge_attr,
        y=y,
    )
    return graph


def build_graphs_for_subject(feature_matrix, conn_matrices, label):
    """
    Гради графови за СИТЕ епохи на еден субјект.

    Параметри
    ---------
    feature_matrix : np.ndarray, облик (n_epochs, n_channels, n_bands)
                       пр. (149, 19, 4)
    conn_matrices : np.ndarray, облик (n_epochs, n_channels, n_channels)
                      веќе threshold-ирани, пр. (149, 19, 19)
    label : int
             0 = HC, 1 = AD - ИСТА лабела за сите епохи на овој субјект
             (бидејќи лабелата е на ниво на субјект, не на ниво на епоха)

    Враќа
    -----
    list од torch_geometric.data.Data објекти, должина = n_epochs
    """
    n_epochs = feature_matrix.shape[0]
    graphs = []

    for i in range(n_epochs):
        graph = build_graph_for_epoch(
            node_features=feature_matrix[i],
            conn_matrix=conn_matrices[i],
            label=label,
        )
        graphs.append(graph)

    return graphs


if __name__ == "__main__":
    # Test (combining everything up until now for 1 subject)
    from step_01_preprocessing import load_and_epoch_subject
    from step_02_feature_extraction import extract_band_powers, band_powers_to_feature_matrix, compute_relative_power
    from step_03_compute_connectivity import compute_connectivity_all_epochs, threshold_connectivity

    test_subject = "sub-001"
    epochs = load_and_epoch_subject(test_subject)
    epochs_data = epochs.get_data()

    # Feature extraction
    band_powers = extract_band_powers(epochs)
    feature_matrix = band_powers_to_feature_matrix(band_powers)
    feature_matrix = compute_relative_power(feature_matrix)

    # Connectivity + threshold for each epoch
    conn_all = compute_connectivity_all_epochs(epochs_data)
    conn_all_thresholded = np.array([
        threshold_connectivity(conn_all[i], top_k_percent=config.TOP_K_EDGES)
        for i in range(conn_all.shape[0])
    ])

    # Градење графови (пример: label=1, како да е AD субјект)
    graphs = build_graphs_for_subject(feature_matrix, conn_all_thresholded, label=1)

    print(f"Направени {len(graphs)} графови за субјект {test_subject}")
    print("\nПример - прв граф (епоха 0):")
    print(graphs[0])
    print("\nx (node features), облик:", graphs[0].x.shape)
    print("edge_index, облик:", graphs[0].edge_index.shape)
    print("edge_attr, облик:", graphs[0].edge_attr.shape)
    print("y (лабела):", graphs[0].y)
