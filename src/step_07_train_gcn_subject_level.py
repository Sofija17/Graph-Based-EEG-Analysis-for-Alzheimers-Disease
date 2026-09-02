"""
Epoch-level GCN со SUBJECT-LEVEL финална класификација,
5-fold subject-wise cross-validation и early stopping

Pipeline:

    EEG subject
        ↓
    повеќе 4-sec EEG сегменти
        ↓
    еден graph по сегмент
        ↓
    GCN -> P(AD) за секој сегмент
        ↓
    mean P(AD) по subject
        ↓
    една финална AD/CN prediction

Во секој fold:

    development subjects
        ↓
    train + validation
        ↓
    train GCN
        ↓
    следење validation loss
        ↓
    early stopping
        ↓
    restore best model
        ↓
    test subjects
        ↓
    subject-level prediction

ВАЖНО:
Сите сегменти од еден subject секогаш се во исто множество
"""

from collections import defaultdict
from pathlib import Path
import argparse
import copy
import csv
from datetime import datetime
import json
import platform
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from torch_geometric.loader import DataLoader

import config

from step_06_split_graph_dataset import (
    load_graphs,
    get_unique_subjects,
    subject_wise_kfold,
    verify_no_leakage,
)

from models.gcn import GCN


# ============================================================
# SETTINGS
# ============================================================

VAL_SIZE = 0.20

EARLY_STOPPING_PATIENCE = 15

# Колку најмалку треба да се подобри validation loss
# за да го сметаме за реално подобрување.
MIN_DELTA = 0.0001

# config.NUM_EPOCHS сега е MAXIMUM број на epochs.
# Early stopping може да запре порано.


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def subject_wise_train_validation_split(
    graphs,
    val_size=VAL_SIZE,
    random_state=None
):
    """
    Ги дели development графовите на train и validation
    на ниво на SUBJECT.

    Сите сегменти од еден subject остануваат заедно.
    """

    if random_state is None:
        random_state = config.RANDOM_SEED

    subject_ids, subject_labels = get_unique_subjects(graphs)

    train_subjects, val_subjects = train_test_split(
        subject_ids,
        test_size=val_size,
        stratify=subject_labels,
        random_state=random_state,
    )

    train_subjects = set(train_subjects)
    val_subjects = set(val_subjects)

    train_graphs = [
        g
        for g in graphs
        if g.subject_id in train_subjects
    ]

    val_graphs = [
        g
        for g in graphs
        if g.subject_id in val_subjects
    ]

    return train_graphs, val_graphs


# ============================================================
# SUBJECT-BALANCED GRAPH WEIGHTS
# ============================================================

def attach_subject_balanced_weights(graphs):
    """
    Додава sample_weight на секој graph така што секој subject има
    приближно еднаков вкупен придонес во loss-от.

    Без ова, subject со 300 сегменти влијае околу 4x повеќе од subject
    со 75 сегменти, иако реално имаме една label-а по subject.
    """

    subject_counts = defaultdict(int)

    for graph in graphs:
        subject_counts[graph.subject_id] += 1

    n_graphs = len(graphs)
    n_subjects = len(subject_counts)

    for graph in graphs:
        weight = (
            n_graphs
            / (
                n_subjects
                * subject_counts[graph.subject_id]
            )
        )

        graph.sample_weight = torch.tensor(
            [weight],
            dtype=torch.float
        )

    return graphs


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    """
    Еден целосен training epoch.
    """

    model.train()

    total_weighted_loss = 0.0
    total_weight = 0.0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        loss_per_graph = criterion(
            out,
            batch.y
        )

        weights = batch.sample_weight.view(-1)

        loss = (
            loss_per_graph
            * weights
        ).sum() / weights.sum()

        loss.backward()
        optimizer.step()

        total_weighted_loss += (
            loss_per_graph.detach()
            * weights
        ).sum().item()

        total_weight += (
            weights.sum().item()
        )

    return total_weighted_loss / total_weight


# ============================================================
# VALIDATION LOSS
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device
):
    """
    Го пресметува loss без ажурирање на weights.

    Ова го користиме за validation set.
    """

    model.eval()

    total_weighted_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            loss_per_graph = criterion(
                out,
                batch.y
            )

            weights = batch.sample_weight.view(-1)

            total_weighted_loss += (
                loss_per_graph
                * weights
            ).sum().item()

            total_weight += (
                weights.sum().item()
            )

    return total_weighted_loss / total_weight


# ============================================================
# SEGMENT-LEVEL EVALUATION
# ============================================================

def evaluate_segment_level(
    model,
    loader,
    device
):
    """
    Евалуација на ниво на EEG сегмент / graph.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            probabilities = torch.softmax(
                out,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_labels.extend(
                batch.y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),
        "f1": f1_score(
            labels,
            predictions
        ),
        "roc_auc": roc_auc_score(
            labels,
            probabilities
        ),
    }


# ============================================================
# SUBJECT-LEVEL EVALUATION
# ============================================================

def evaluate_subject_level(
    model,
    graphs,
    device,
    threshold=0.5
):
    """
    За секој segment graph:
        GCN -> P(AD)

    Потоа ги групираме според subject_id:

        mean P(AD) >= threshold -> AD
        mean P(AD) < threshold  -> CN
    """

    model.eval()

    loader = DataLoader(
        graphs,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    subjects = defaultdict(
        lambda: {
            "probabilities": [],
            "label": None
        }
    )

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            probabilities = torch.softmax(
                out,
                dim=1
            )

            ad_probabilities = (
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            labels = (
                batch.y
                .cpu()
                .numpy()
            )

            subject_ids = batch.subject_id

            if isinstance(subject_ids, str):
                subject_ids = [subject_ids]
            else:
                subject_ids = list(subject_ids)

            for subject_id, probability, label in zip(
                subject_ids,
                ad_probabilities,
                labels
            ):

                probability = float(probability)
                label = int(label)

                subjects[
                    subject_id
                ]["probabilities"].append(
                    probability
                )

                if (
                    subjects[
                        subject_id
                    ]["label"] is None
                ):

                    subjects[
                        subject_id
                    ]["label"] = label

                elif (
                    subjects[
                        subject_id
                    ]["label"] != label
                ):

                    raise ValueError(
                        f"Subject {subject_id} "
                        f"има различни labels!"
                    )

    # --------------------------------------------------------
    # Aggregation по subject
    # --------------------------------------------------------

    subject_ids = []
    subject_labels = []
    subject_predictions = []
    subject_probabilities = []

    for subject_id, data in subjects.items():

        mean_ad_probability = float(
            np.mean(
                data["probabilities"]
            )
        )

        prediction = (
            1
            if mean_ad_probability >= threshold
            else 0
        )

        subject_ids.append(
            subject_id
        )

        subject_labels.append(
            data["label"]
        )

        subject_predictions.append(
            prediction
        )

        subject_probabilities.append(
            mean_ad_probability
        )

    subject_labels = np.array(
        subject_labels
    )

    subject_predictions = np.array(
        subject_predictions
    )

    subject_probabilities = np.array(
        subject_probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        subject_labels,
        subject_predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(
            subject_labels,
            subject_predictions
        ),

        "f1": f1_score(
            subject_labels,
            subject_predictions,
            zero_division=0
        ),

        "precision": precision_score(
            subject_labels,
            subject_predictions,
            zero_division=0
        ),

        "recall": recall_score(
            subject_labels,
            subject_predictions,
            zero_division=0
        ),

        "specificity": specificity,

        "roc_auc": roc_auc_score(
            subject_labels,
            subject_probabilities
        ),

        "subject_ids": subject_ids,
        "labels": subject_labels,
        "predictions": subject_predictions,
        "probabilities": subject_probabilities,
        "threshold": threshold,
        "subjects": subjects,
    }


# ============================================================
# VALIDATION THRESHOLD SELECTION
# ============================================================

def choose_threshold_by_youdens_j(labels, probabilities):
    """
    Избира subject-level threshold користејќи Youden's J:

        J = sensitivity + specificity - 1

    Threshold-от се избира само од validation subjects, а потоа се
    применува еднаш на test subjects.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    candidate_thresholds = np.unique(
        probabilities
    )

    candidate_thresholds = np.concatenate(
        [
            np.array([0.0]),
            candidate_thresholds,
            np.array([1.0]),
        ]
    )

    best_threshold = 0.5
    best_j = -float("inf")
    best_sensitivity = 0.0
    best_specificity = 0.0

    for threshold in candidate_thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        true_positives = np.sum(
            (labels == 1)
            & (predictions == 1)
        )

        false_negatives = np.sum(
            (labels == 1)
            & (predictions == 0)
        )

        true_negatives = np.sum(
            (labels == 0)
            & (predictions == 0)
        )

        false_positives = np.sum(
            (labels == 0)
            & (predictions == 1)
        )

        sensitivity = (
            true_positives
            / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        specificity = (
            true_negatives
            / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0.0
        )

        youdens_j = (
            sensitivity
            + specificity
            - 1.0
        )

        is_better = (
            youdens_j
            > best_j + 1e-12
        )

        is_tie_closer_to_default = (
            abs(youdens_j - best_j) <= 1e-12
            and abs(threshold - 0.5)
            < abs(best_threshold - 0.5)
        )

        if (
            is_better
            or is_tie_closer_to_default
        ):

            best_threshold = float(threshold)
            best_j = float(youdens_j)
            best_sensitivity = float(sensitivity)
            best_specificity = float(specificity)

    return {
        "threshold": best_threshold,
        "youdens_j": best_j,
        "sensitivity": best_sensitivity,
        "specificity": best_specificity,
    }


# ============================================================
# LOSS PLOT
# ============================================================

def plot_losses(
    train_losses,
    val_losses,
    best_epoch,
    fold,
    output_dir=None
):
    """
    Црта train loss и validation loss за секој training epoch.

    Вертикалната линија го покажува epoch-от со
    најмал validation loss.
    """

    epochs = range(
        1,
        len(train_losses) + 1
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation loss"
    )

    plt.axvline(
        x=best_epoch,
        linestyle="--",
        label=f"Best epoch = {best_epoch}"
    )

    plt.xlabel("Training epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Fold {fold} - Train vs Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    # results/training_curves/
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "results"
            / "training_curves"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"fold_{fold}_loss.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Loss график зачуван: "
        f"{output_path}"
    )


# ============================================================
# PRINT SUBJECT PREDICTIONS
# ============================================================

def print_subject_predictions(metrics):

    print("\nПредикции по субјект:")
    print("-" * 75)

    for (
        subject_id,
        true_label,
        prediction,
        probability
    ) in zip(
        metrics["subject_ids"],
        metrics["labels"],
        metrics["predictions"],
        metrics["probabilities"]
    ):

        true_name = (
            "AD"
            if true_label == 1
            else "CN"
        )

        predicted_name = (
            "AD"
            if prediction == 1
            else "CN"
        )

        num_segments = len(
            metrics["subjects"][
                subject_id
            ]["probabilities"]
        )

        print(
            f"{subject_id:15s} | "
            f"segments={num_segments:3d} | "
            f"true={true_name:2s} | "
            f"P(AD)={probability:.3f} | "
            f"pred={predicted_name}"
        )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold,
    development_graphs,
    test_graphs,
    device,
    output_dir=None
):
    """
    Еден fold:

        development
            ↓
        train + validation
            ↓
        train со early stopping
            ↓
        restore best model
            ↓
        test
    """

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"FOLD {fold}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Development / test leakage check
    # --------------------------------------------------------

    verify_no_leakage(
        development_graphs,
        test_graphs
    )

    # --------------------------------------------------------
    # DEVELOPMENT -> TRAIN + VALIDATION
    # --------------------------------------------------------

    train_graphs, val_graphs = (
        subject_wise_train_validation_split(
            development_graphs,
            val_size=VAL_SIZE,
            random_state=(
                config.RANDOM_SEED + fold
            )
        )
    )

    print(
        "\nTrain / validation split:"
    )

    verify_no_leakage(
        train_graphs,
        val_graphs
    )

    train_subjects = set(
        g.subject_id
        for g in train_graphs
    )

    val_subjects = set(
        g.subject_id
        for g in val_graphs
    )

    test_subjects = set(
        g.subject_id
        for g in test_graphs
    )

    # Дополнителна безбедносна проверка
    if train_subjects & test_subjects:
        raise ValueError(
            "Leakage меѓу train и test!"
        )

    if val_subjects & test_subjects:
        raise ValueError(
            "Leakage меѓу validation и test!"
        )

    print(
        f"Final split за Fold {fold}:"
    )

    print(
        f"Train subjects: "
        f"{len(train_subjects)}"
    )

    print(
        f"Validation subjects: "
        f"{len(val_subjects)}"
    )

    print(
        f"Test subjects: "
        f"{len(test_subjects)}"
    )

    print(
        f"Train graphs: "
        f"{len(train_graphs)}"
    )

    print(
        f"Validation graphs: "
        f"{len(val_graphs)}"
    )

    print(
        f"Test graphs: "
        f"{len(test_graphs)}"
    )

    attach_subject_balanced_weights(
        train_graphs
    )

    attach_subject_balanced_weights(
        val_graphs
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_graphs,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # NEW MODEL FOR THIS FOLD
    # --------------------------------------------------------

    set_seed(
        config.RANDOM_SEED + fold
    )

    num_node_features = (
        train_graphs[0].x.shape[1]
    )

    model = GCN(
        num_node_features=num_node_features
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            reduction="none"
        )
    )

    # --------------------------------------------------------
    # EARLY STOPPING VARIABLES
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_model_state = None

    best_epoch = 0

    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print(
        "\nПочнува тренирање...\n"
    )

    for epoch in range(
        1,
        config.NUM_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        # ----------------------------------------------------
        # Проверка за подобрување
        # ----------------------------------------------------

        if (
            val_loss
            < best_val_loss - MIN_DELTA
        ):

            best_val_loss = val_loss

            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Епоха {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"best_epoch={best_epoch:3d}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly stopping!"
            )

            print(
                f"Нема подобрување "
                f"{EARLY_STOPPING_PATIENCE} "
                f"епохи по ред."
            )

            print(
                f"Training запрен на epoch "
                f"{epoch}."
            )

            print(
                f"Најдобар epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_model_state is None:
        raise RuntimeError(
            "Не е зачуван best model state."
        )

    model.load_state_dict(
        best_model_state
    )

    model.to(device)

    print(
        f"\nВратен е моделот од "
        f"epoch {best_epoch}."
    )

    # --------------------------------------------------------
    # LOSS GRAPH
    # --------------------------------------------------------

    plot_losses(
        train_losses,
        val_losses,
        best_epoch,
        fold,
        output_dir=(
            output_dir / "training_curves"
            if output_dir is not None
            else None
        )
    )

    # --------------------------------------------------------
    # VALIDATION THRESHOLD
    # --------------------------------------------------------

    val_subject_metrics = (
        evaluate_subject_level(
            model,
            val_graphs,
            device
        )
    )

    threshold_info = (
        choose_threshold_by_youdens_j(
            val_subject_metrics["labels"],
            val_subject_metrics["probabilities"]
        )
    )

    subject_threshold = (
        threshold_info["threshold"]
    )

    print(
        "\nValidation threshold "
        "chosen by Youden's J:"
    )

    print(
        f"threshold={subject_threshold:.3f} | "
        f"J={threshold_info['youdens_j']:.3f} | "
        f"sensitivity={threshold_info['sensitivity']:.3f} | "
        f"specificity={threshold_info['specificity']:.3f}"
    )

    # --------------------------------------------------------
    # TEST - дури сега
    # --------------------------------------------------------

    segment_metrics = (
        evaluate_segment_level(
            model,
            test_loader,
            device
        )
    )

    subject_metrics = (
        evaluate_subject_level(
            model,
            test_graphs,
            device,
            threshold=subject_threshold
        )
    )

    print(
        "\nTEST Segment-level:"
    )

    print(
        f"Accuracy = "
        f"{segment_metrics['accuracy']:.3f}"
    )

    print(
        f"F1       = "
        f"{segment_metrics['f1']:.3f}"
    )

    print(
        f"ROC-AUC  = "
        f"{segment_metrics['roc_auc']:.3f}"
    )

    print(
        "\nTEST Subject-level "
        f"(threshold={subject_threshold:.3f}):"
    )

    print(
        f"Accuracy = "
        f"{subject_metrics['accuracy']:.3f}"
    )

    print(
        f"F1       = "
        f"{subject_metrics['f1']:.3f}"
    )

    print(
        f"ROC-AUC  = "
        f"{subject_metrics['roc_auc']:.3f}"
    )

    print_subject_predictions(
        subject_metrics
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "subject_threshold": subject_threshold,
        "threshold_info": threshold_info,
        "segment_metrics": segment_metrics,
        "subject_metrics": subject_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }


# ============================================================
# MAIN
# ============================================================

def _json_value(value):
    """Convert Path/numpy values to JSON-safe Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def get_experiment_config(device, connectivity_method=None, graphs_file="all_graphs.pt"):
    """Snapshot parameters needed to reproduce the reference run."""
    connectivity_method = connectivity_method or config.CONNECTIVITY_METHOD
    return {
        "experiment": f"{connectivity_method}_reference",
        "connectivity_method": connectivity_method,
        "graphs_file": graphs_file,
        "top_k_edges": config.TOP_K_EDGES,
        "frequency_bands": config.FREQ_BANDS,
        "epoch_duration_seconds": config.EPOCH_DURATION,
        "epoch_overlap_seconds": config.EPOCH_OVERLAP,
        "gcn_hidden_dim": config.GCN_HIDDEN_DIM,
        "gcn_num_layers_config": config.GCN_NUM_LAYERS,
        "learning_rate": config.LEARNING_RATE,
        "maximum_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "random_seed": config.RANDOM_SEED,
        "outer_folds": 5,
        "validation_size": VAL_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": MIN_DELTA,
        "subject_aggregation": "mean_ad_probability",
        "threshold_selection": "validation_youdens_j",
        "subject_balanced_loss": True,
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_oof_results(labels, predictions, probabilities, output_dir):
    """Save the final subject-level confusion matrix, ROC and scores."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    ax.set_xticks([0, 1], labels=["CN", "AD"])
    ax.set_yticks([0, 1], labels=["CN", "AD"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Out-of-fold subject confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_confusion_matrix.png", dpi=150)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Out-of-fold subject ROC curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    cn_probabilities = probabilities[labels == 0]
    ad_probabilities = probabilities[labels == 1]
    rng = np.random.default_rng(config.RANDOM_SEED)
    ax.scatter(
        rng.normal(0, 0.04, size=len(cn_probabilities)),
        cn_probabilities,
        alpha=0.8,
        label="CN"
    )
    ax.scatter(
        rng.normal(1, 0.04, size=len(ad_probabilities)),
        ad_probabilities,
        alpha=0.8,
        label="AD"
    )
    ax.set_xticks([0, 1], labels=["CN", "AD"])
    ax.set_ylabel("Out-of-fold P(AD)")
    ax.set_title("Subject-level predicted probabilities")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "oof_probability_distribution.png", dpi=150)
    plt.close(fig)


def save_experiment_results(output_dir, experiment_config, fold_rows,
                            prediction_rows, history_rows, split_rows,
                            overall_metrics):
    """Persist every result required for later fair ablations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(_json_value(experiment_config), file, indent=2, ensure_ascii=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_value(overall_metrics), file, indent=2, ensure_ascii=False)

    write_csv(output_dir / "fold_metrics.csv", fold_rows, list(fold_rows[0]))
    write_csv(
        output_dir / "oof_subject_predictions.csv",
        prediction_rows,
        list(prediction_rows[0])
    )
    write_csv(output_dir / "training_history.csv", history_rows, list(history_rows[0]))
    write_csv(output_dir / "fold_subject_splits.csv", split_rows, list(split_rows[0]))

    labels = np.array([row["true_label"] for row in prediction_rows])
    predictions = np.array([row["prediction"] for row in prediction_rows])
    probabilities = np.array([row["ad_probability"] for row in prediction_rows])
    plot_oof_results(labels, predictions, probabilities, output_dir / "figures")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the subject-level GCN.")
    parser.add_argument(
        "--graphs-file",
        default="all_graphs.pt",
        help="Graph dataset filename inside data/graphs.",
    )
    parser.add_argument(
        "--connectivity-method",
        choices=("pearson", "spearman", "coherence"),
        default=None,
        help="Method label saved in experiment metadata.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Results subdirectory name (default: <method>_reference).",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    connectivity_method = args.connectivity_method or config.CONNECTIVITY_METHOD
    experiment_name = args.experiment_name or f"{connectivity_method}_reference"

    set_seed(
        config.RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Користиме device: {device}\n"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.RESULTS_DIR / experiment_name / run_id

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    graphs = load_graphs(args.graphs_file)

    print(
        f"Вкупно графови: "
        f"{len(graphs)}"
    )

    unique_subjects = set(
        g.subject_id
        for g in graphs
    )

    print(
        f"Вкупно субјекти: "
        f"{len(unique_subjects)}"
    )

    print(
        "\nПочнува 5-fold "
        "subject-wise cross-validation "
        "со early stopping..."
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    fold_accuracies = []
    fold_f1_scores = []
    fold_roc_auc_scores = []
    fold_best_epochs = []
    fold_thresholds = []

    all_subject_labels = []
    all_subject_predictions = []
    all_subject_probabilities = []
    fold_rows = []
    prediction_rows = []
    history_rows = []
    split_rows = []

    # --------------------------------------------------------
    # 5 folds
    # --------------------------------------------------------

    folds = subject_wise_kfold(
        graphs,
        n_splits=5
    )

    for fold, (
        development_graphs,
        test_graphs
    ) in enumerate(
        folds,
        start=1
    ):

        result = train_fold(
            fold,
            development_graphs,
            test_graphs,
            device,
            output_dir=output_dir
        )

        subject_metrics = (
            result["subject_metrics"]
        )

        fold_best_epochs.append(
            result["best_epoch"]
        )

        fold_thresholds.append(
            result["subject_threshold"]
        )

        fold_accuracies.append(
            subject_metrics["accuracy"]
        )

        fold_f1_scores.append(
            subject_metrics["f1"]
        )

        fold_roc_auc_scores.append(
            subject_metrics["roc_auc"]
        )

        # Out-of-fold predictions
        all_subject_labels.extend(
            subject_metrics["labels"]
        )

        all_subject_predictions.extend(
            subject_metrics["predictions"]
        )

        all_subject_probabilities.extend(
            subject_metrics["probabilities"]
        )

        fold_rows.append({
            "fold": fold,
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_val_loss"],
            "threshold": result["subject_threshold"],
            "validation_youdens_j": result["threshold_info"]["youdens_j"],
            "validation_sensitivity": result["threshold_info"]["sensitivity"],
            "validation_specificity": result["threshold_info"]["specificity"],
            "accuracy": subject_metrics["accuracy"],
            "precision": subject_metrics["precision"],
            "recall_sensitivity": subject_metrics["recall"],
            "specificity": subject_metrics["specificity"],
            "f1": subject_metrics["f1"],
            "roc_auc": subject_metrics["roc_auc"],
            "segment_accuracy": result["segment_metrics"]["accuracy"],
            "segment_f1": result["segment_metrics"]["f1"],
            "segment_roc_auc": result["segment_metrics"]["roc_auc"],
            "n_train_subjects": len(result["train_subjects"]),
            "n_validation_subjects": len(result["validation_subjects"]),
            "n_test_subjects": len(result["test_subjects"]),
        })

        for epoch, (train_loss, val_loss) in enumerate(
            zip(result["train_losses"], result["val_losses"]), start=1
        ):
            history_rows.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "is_best_epoch": int(epoch == result["best_epoch"]),
            })

        for split_name in ("train", "validation", "test"):
            for subject_id in result[f"{split_name}_subjects"]:
                split_rows.append({
                    "fold": fold,
                    "split": split_name,
                    "subject_id": subject_id,
                })

        for subject_id, label, prediction, probability in zip(
            subject_metrics["subject_ids"],
            subject_metrics["labels"],
            subject_metrics["predictions"],
            subject_metrics["probabilities"],
        ):
            prediction_rows.append({
                "fold": fold,
                "subject_id": subject_id,
                "true_label": int(label),
                "prediction": int(prediction),
                "ad_probability": float(probability),
                "threshold": result["subject_threshold"],
                "n_segments": len(subject_metrics["subjects"][subject_id]["probabilities"]),
            })

    # ========================================================
    # CV SUMMARY
    # ========================================================

    fold_accuracies = np.array(
        fold_accuracies
    )

    fold_f1_scores = np.array(
        fold_f1_scores
    )

    fold_roc_auc_scores = np.array(
        fold_roc_auc_scores
    )

    fold_best_epochs = np.array(
        fold_best_epochs
    )

    fold_thresholds = np.array(
        fold_thresholds
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        "5-FOLD SUBJECT-LEVEL CROSS-VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    for i in range(5):

        print(
            f"Fold {i + 1}: "
            f"best_epoch="
            f"{fold_best_epochs[i]:3d} | "
            f"threshold="
            f"{fold_thresholds[i]:.3f} | "
            f"Accuracy="
            f"{fold_accuracies[i]:.3f} | "
            f"F1="
            f"{fold_f1_scores[i]:.3f} | "
            f"ROC-AUC="
            f"{fold_roc_auc_scores[i]:.3f}"
        )

    print(
        "\nПросечни резултати:"
    )

    print(
        f"Accuracy = "
        f"{fold_accuracies.mean():.3f} "
        f"± {fold_accuracies.std():.3f}"
    )

    print(
        f"F1-score = "
        f"{fold_f1_scores.mean():.3f} "
        f"± {fold_f1_scores.std():.3f}"
    )

    print(
        f"ROC-AUC  = "
        f"{fold_roc_auc_scores.mean():.3f} "
        f"± {fold_roc_auc_scores.std():.3f}"
    )

    print(
        f"\nBest epochs: "
        f"{fold_best_epochs.tolist()}"
    )

    print(
        f"Average best epoch: "
        f"{fold_best_epochs.mean():.1f}"
    )

    print(
        f"Thresholds: "
        f"{fold_thresholds.round(3).tolist()}"
    )

    print(
        f"Average threshold: "
        f"{fold_thresholds.mean():.3f}"
    )

    # ========================================================
    # OUT-OF-FOLD RESULTS
    # ========================================================

    all_subject_labels = np.array(
        all_subject_labels
    )

    all_subject_predictions = np.array(
        all_subject_predictions
    )

    all_subject_probabilities = np.array(
        all_subject_probabilities
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        "OUT-OF-FOLD SUBJECT RESULTS"
    )

    print(
        "=" * 70
    )

    overall_accuracy = accuracy_score(
        all_subject_labels,
        all_subject_predictions
    )

    overall_f1 = f1_score(
        all_subject_labels,
        all_subject_predictions
    )

    overall_roc_auc = roc_auc_score(
        all_subject_labels,
        all_subject_probabilities
    )

    overall_precision = precision_score(
        all_subject_labels,
        all_subject_predictions,
        zero_division=0
    )
    overall_recall = recall_score(
        all_subject_labels,
        all_subject_predictions,
        zero_division=0
    )
    overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(
        all_subject_labels,
        all_subject_predictions,
        labels=[0, 1]
    ).ravel()
    overall_specificity = overall_tn / (overall_tn + overall_fp)

    print(
        f"Subjects evaluated: "
        f"{len(all_subject_labels)}"
    )

    print(
        f"Accuracy : "
        f"{overall_accuracy:.3f}"
    )

    print(
        f"F1-score : "
        f"{overall_f1:.3f}"
    )

    print(
        f"ROC-AUC  : "
        f"{overall_roc_auc:.3f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_matrix(
            all_subject_labels,
            all_subject_predictions
        )
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            all_subject_labels,
            all_subject_predictions,
            target_names=["CN", "AD"],
            digits=3,
            zero_division=0
        )
    )

    overall_metrics = {
        "subjects_evaluated": len(all_subject_labels),
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall_sensitivity": overall_recall,
        "specificity": overall_specificity,
        "f1": overall_f1,
        "roc_auc": overall_roc_auc,
        "confusion_matrix": [[int(overall_tn), int(overall_fp)],
                             [int(overall_fn), int(overall_tp)]],
        "fold_mean": {
            metric: float(np.mean([row[metric] for row in fold_rows]))
            for metric in ("accuracy", "precision", "recall_sensitivity",
                           "specificity", "f1", "roc_auc")
        },
        "fold_standard_deviation": {
            metric: float(np.std([row[metric] for row in fold_rows]))
            for metric in ("accuracy", "precision", "recall_sensitivity",
                           "specificity", "f1", "roc_auc")
        },
    }

    save_experiment_results(
        output_dir=output_dir,
        experiment_config=get_experiment_config(
            device,
            connectivity_method=connectivity_method,
            graphs_file=args.graphs_file,
        ),
        fold_rows=fold_rows,
        prediction_rows=prediction_rows,
        history_rows=history_rows,
        split_rows=split_rows,
        overall_metrics=overall_metrics,
    )

    print(f"\nКомплетните {connectivity_method} резултати се зачувани во: {output_dir}")


if __name__ == "__main__":
    main()
