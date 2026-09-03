"""
Random Forest baseline for AD vs CN classification.

For a fair comparison with GCN:
    - use the same graphs
    - use the same subject-wise train/test split
    - use the same qEEG features and thresholded connectivity

Each graph (one EEG epoch) is converted into one feature vector:

    node features:
        n_channels × 4 qEEG features
        example: 19 × 4 = 76

    connectivity:
        upper triangle of the connectivity matrix
        example: 19 × 18 / 2 = 171

    total:
        76 + 171 = 247 features per epoch
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import config
from step_06_split_graph_dataset import (
    load_graphs,
    subject_wise_train_test_split,
    verify_no_leakage,
)


def graph_to_vector(graph):
    """
    Convert one PyTorch Geometric graph into a 1D feature vector.

    The vector contains:
        1. qEEG node features
        2. connectivity values from the upper triangle

    Returns
    -------
    numpy.ndarray
        1D feature vector
    """

    # ---------------------------------------------------------
    # 1. qEEG node features
    # ---------------------------------------------------------
    # graph.x shape:
    # (n_channels, n_features)
    #
    # example:
    # (19, 4) -> flatten -> (76,)
    node_features = graph.x.detach().cpu().numpy().flatten()

    num_nodes = graph.num_nodes

    # ---------------------------------------------------------
    # 2. Connectivity matrix
    # ---------------------------------------------------------
    connectivity = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32
    )

    edge_index = graph.edge_index.detach().cpu().numpy()

    src = edge_index[0]
    dst = edge_index[1]

    # edge_attr can be:
    # shape (E,) or (E, 1)
    edge_weights = (
        graph.edge_attr
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    connectivity[src, dst] = edge_weights

    # ---------------------------------------------------------
    # 3. Use only the upper triangle
    # ---------------------------------------------------------
    # Do not use the diagonal because corr(channel, channel) = 1
    # Do not use the lower half because the Pearson matrix is symmetric
    upper_indices = np.triu_indices(num_nodes, k=1)

    connectivity_features = connectivity[upper_indices]

    # ---------------------------------------------------------
    # 4. Concatenation
    # ---------------------------------------------------------
    feature_vector = np.concatenate([
        node_features,
        connectivity_features
    ])

    return feature_vector


def graphs_to_dataset(graphs):
    """
    Convert a list of PyG graphs into X and y for sklearn.

    Returns
    -------
    X : numpy.ndarray
        shape = (n_graphs, n_features)

    y : numpy.ndarray
        shape = (n_graphs,)
    """

    X = np.array([
        graph_to_vector(graph)
        for graph in graphs
    ])

    y = np.array([
        int(graph.y.item())
        for graph in graphs
    ])

    return X, y


def main():

    # ---------------------------------------------------------
    # Load graphs
    # ---------------------------------------------------------
    print("Вчитувам графови...")

    graphs = load_graphs()

    print(f"Вкупно графови: {len(graphs)}\n")

    # ---------------------------------------------------------
    # Subject-wise split
    # ---------------------------------------------------------
    train_graphs, test_graphs = subject_wise_train_test_split(
        graphs,
        test_size=0.2
    )

    verify_no_leakage(train_graphs, test_graphs)

    print()

    # ---------------------------------------------------------
    # Graph -> vector
    # ---------------------------------------------------------
    print("Ги претворам графовите во feature vectors...")

    X_train, y_train = graphs_to_dataset(train_graphs)
    X_test, y_test = graphs_to_dataset(test_graphs)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    print()

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------
    print("Почнува тренирање на Random Forest...\n")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # probability that the sample is AD (class=1)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_f1 = f1_score(
        y_test,
        test_predictions
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)

    print(f"Train accuracy : {train_accuracy:.3f}")
    print(f"Test accuracy  : {test_accuracy:.3f}")
    print(f"Test F1-score  : {test_f1:.3f}")
    print(f"Test ROC-AUC   : {test_roc_auc:.3f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["CN", "AD"],
            digits=3
        )
    )

    return model


if __name__ == "__main__":
    main()
