"""
Random Forest baseline за AD vs CN класификација.

За фер споредба со GCN:
    - ги користиме истите графови
    - го користиме истиот subject-wise train/test split
    - ги користиме истите qEEG features и thresholded connectivity

Секој граф (една EEG епоха) го претвораме во еден feature vector:

    node features:
        n_channels × 4 qEEG features
        пример: 19 × 4 = 76

    connectivity:
        upper triangle од connectivity matrix
        пример: 19 × 18 / 2 = 171

    вкупно:
        76 + 171 = 247 features по епоха
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
    Претвора еден PyTorch Geometric граф во 1D feature vector.

    Векторот содржи:
        1. qEEG node features
        2. connectivity вредности од upper triangle

    Враќа
    -----
    numpy.ndarray
        1D feature vector
    """

    # ---------------------------------------------------------
    # 1. qEEG node features
    # ---------------------------------------------------------
    # graph.x shape:
    # (n_channels, n_features)
    #
    # пример:
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

    # edge_attr може да биде:
    # shape (E,) или (E, 1)
    edge_weights = (
        graph.edge_attr
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    connectivity[src, dst] = edge_weights

    # ---------------------------------------------------------
    # 3. Земаме само upper triangle
    # ---------------------------------------------------------
    # Не ја земаме дијагоналата бидејќи corr(channel, channel) = 1
    # Не ја земаме долната половина бидејќи Pearson matrix е симетрична
    upper_indices = np.triu_indices(num_nodes, k=1)

    connectivity_features = connectivity[upper_indices]

    # ---------------------------------------------------------
    # 4. Спојување
    # ---------------------------------------------------------
    feature_vector = np.concatenate([
        node_features,
        connectivity_features
    ])

    return feature_vector


def graphs_to_dataset(graphs):
    """
    Претвора листа од PyG графови во X и y за sklearn.

    Враќа
    -----
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

    # probability дека примерот е AD (class=1)
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