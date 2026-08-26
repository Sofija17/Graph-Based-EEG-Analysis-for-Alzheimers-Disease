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
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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

    return {
        "accuracy": accuracy_score(
            subject_labels,
            subject_predictions
        ),

        "f1": f1_score(
            subject_labels,
            subject_predictions
        ),

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
    fold
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
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    output_dir = (
        project_root
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
    device
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
        fold
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
    }


# ============================================================
# MAIN
# ============================================================

def main():

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

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    graphs = load_graphs()

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
            device
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


if __name__ == "__main__":
    main()
