"""
Subject-wise train/test split ensures that epochs (graphs) from the
SAME subject never appear in train and test at the same time.
"""

import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

import config


def load_graphs(filename="all_graphs.pt"):
    path = config.DATA_GRAPHS / filename
    graphs = torch.load(path, weights_only=False)
    return graphs


def get_unique_subjects(graphs):
    """
    Extract a list of unique subjects and their labels.

    Returns
    -------
    subject_ids : list[str]
    subject_labels : list[int]  (parallel list, same length)
    """
    seen = {}
    for g in graphs:
        seen[g.subject_id] = int(g.y.item())

    subject_ids = list(seen.keys())
    subject_labels = list(seen.values())
    return subject_ids, subject_labels


def subject_wise_train_test_split(graphs, test_size=0.2, random_state=None):
    """
    Split graphs into train/test at SUBJECT level (not epoch level),
    using stratification to preserve the same AD/CN ratio in both groups.

    Returns
    -------
    train_graphs : list
    test_graphs : list
    """
    random_state = random_state or config.RANDOM_SEED
    subject_ids, subject_labels = get_unique_subjects(graphs)

    train_subjects, test_subjects = train_test_split(
        subject_ids,
        test_size=test_size,
        stratify=subject_labels,
        random_state=random_state,
    )

    train_subjects = set(train_subjects)
    test_subjects = set(test_subjects)

    train_graphs = [g for g in graphs if g.subject_id in train_subjects]
    test_graphs = [g for g in graphs if g.subject_id in test_subjects]

    return train_graphs, test_graphs


def verify_no_leakage(train_graphs, test_graphs):
    """
    Check that NO subject appears in both train and test.
    If leakage exists, raise an error.
    """
    train_subjects = set(g.subject_id for g in train_graphs)
    test_subjects = set(g.subject_id for g in test_graphs)

    overlap = train_subjects & test_subjects

    if overlap:
        raise ValueError(f"DATA LEAKAGE! Овие субјекти се и во train и во test: {overlap}")

    print("Проверка success: нема преклопување субјекти меѓу train и test.")
    print(f"Train субјекти: {len(train_subjects)}, Test субјекти: {len(test_subjects)}")
    print(f"Train графови: {len(train_graphs)}, Test графови: {len(test_graphs)}")


def subject_wise_kfold(graphs, n_splits=5, random_state=None):
    """
    Subject-level K-fold cross-validation instead of a single
    train/test split. Useful for a better and more robust evaluation
    on a small dataset (65 subjects).

    Returns
    -------
    generator - each element is (train_graphs, test_graphs) for one fold
    """
    random_state = random_state or config.RANDOM_SEED
    subject_ids, subject_labels = get_unique_subjects(graphs)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # StratifiedGroupKFold requires X, y, and groups. Here each subject is
    # its own group, not each graph individually.
    dummy_X = subject_ids
    for train_idx, test_idx in sgkf.split(dummy_X, subject_labels, groups=subject_ids):
        train_subjects = set(subject_ids[i] for i in train_idx)
        test_subjects = set(subject_ids[i] for i in test_idx)

        train_graphs = [g for g in graphs if g.subject_id in train_subjects]
        test_graphs = [g for g in graphs if g.subject_id in test_subjects]

        yield train_graphs, test_graphs


if __name__ == "__main__":
    graphs = load_graphs()
    print(f"Вчитани {len(graphs)} графови вкупно.\n")

    subject_ids, subject_labels = get_unique_subjects(graphs)
    n_ad = sum(subject_labels)
    n_cn = len(subject_labels) - n_ad
    print(f"Уникатни субјекти: {len(subject_ids)} (AD={n_ad}, CN={n_cn})\n")

    # train/test split
    train_graphs, test_graphs = subject_wise_train_test_split(graphs, test_size=0.2)
    verify_no_leakage(train_graphs, test_graphs)
