"""
Build PyTorch Geometric Data objects (graphs) from:
  - node features (delta/theta/alpha/beta power per channel)
  - connectivity matrix (Pearson correlation, already thresholded)

Each epoch -> one graph.
"""

import numpy as np
import torch
from torch_geometric.data import Data

import config


def connectivity_matrix_to_edge_list(conn_matrix):
    """
    Convert the square connectivity matrix (n_channels x n_channels)
    into PyG format: edge_index + edge_attr.

    Use only non-zero values (already thresholded outside this function)
    and ignore self-loops (the diagonal).
    Parameters
    ----------
    conn_matrix : np.ndarray, shape (n_channels, n_channels)
                  already thresholded (most values are 0)

    Returns
    -------
    edge_index : torch.Tensor, shape (2, n_edges)
    edge_attr : torch.Tensor, shape (n_edges,)
    """
    n_channels = conn_matrix.shape[0]

    sources = []
    targets = []
    signed_weights = []

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue  # skip self-loops
            if conn_matrix[i, j] != 0:
                sources.append(i)
                targets.append(j)
                # Use the absolute value: for functional connectivity strength,
                # the correlation magnitude matters, not its direction/sign.
                # Also, GCNConv internally normalizes using the square root of
                # each node degree. Negative weights can produce a negative
                # degree -> sqrt(negative number) -> NaN. Absolute values avoid this.
                signed_weights.append(conn_matrix[i, j])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    signed_edge_attr = torch.tensor(signed_weights, dtype=torch.float)
    edge_attr = signed_edge_attr.abs()

    return edge_index, edge_attr, signed_edge_attr


def build_graph_for_epoch(node_features, conn_matrix, label):
    """
    Build ONE PyG Data graph for one epoch.

    Parameters
    ----------
    node_features : np.ndarray, shape (n_channels, n_bands)
                     e.g. (19, 4) - delta/theta/alpha/beta per channel,
                     for this specific epoch
    conn_matrix : np.ndarray, shape (n_channels, n_channels)
                   already-thresholded connectivity matrix,
                   for the same epoch
    label : int
             0 = Healthy Control, 1 = Alzheimer's Disease

    Returns
    -------
    torch_geometric.data.Data object
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
    Build graphs for ALL epochs of one subject.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_epochs, n_channels, n_bands)
                       e.g. (149, 19, 4)
    conn_matrices : np.ndarray, shape (n_epochs, n_channels, n_channels)
                      already thresholded, e.g. (149, 19, 19)
    label : int
             0 = HC, 1 = AD - SAME label for all epochs of this subject
             (because the label is subject-level, not epoch-level)

    Returns
    -------
    list of torch_geometric.data.Data objects, length = n_epochs
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

    # Build graphs (example: label=1, as if this is an AD subject)
    graphs = build_graphs_for_subject(feature_matrix, conn_all_thresholded, label=1)

    print(f"Направени {len(graphs)} графови за субјект {test_subject}")
    print("\nПример - прв граф (епоха 0):")
    print(graphs[0])
    print("\nx (node features), облик:", graphs[0].x.shape)
    print("edge_index, облик:", graphs[0].edge_index.shape)
    print("edge_attr, облик:", graphs[0].edge_attr.shape)
    print("y (лабела):", graphs[0].y)
