"""
Subject-wise train/test split осигурува дека епохи (графови) од
ИСТ субјект никогаш не се појавуваат и во train и во test истовремено
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
    Извлекува листа од уникатни субјекти + нивните лабели

    Враќа
    -----
    subject_ids : list[str]
    subject_labels : list[int]  (паралелна листа, иста должина)
    """
    seen = {}
    for g in graphs:
        seen[g.subject_id] = int(g.y.item())

    subject_ids = list(seen.keys())
    subject_labels = list(seen.values())
    return subject_ids, subject_labels


def subject_wise_train_test_split(graphs, test_size=0.2, random_state=None):
    """
    Ги дели графовите во train/test, на ниво на СУБЈЕКТ (не на ниво
    на епоха), со stratify за да се задржи истиот AD/CN сооднос во
    двете групи

    Враќа
    -----
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
    Проверка: осигурува дека НИТУ ЕДЕН субјект не се појавува
    и во train и во test. Ако има leakage, ќе фрли грешка
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
    K-fold cross-validation на ниво на субјект (наместо еднократен
    train/test split). Корисно за подобра и поробустна евалуација
    на мал датасет (65 субјекти)

    Враќа
    -----
    generator - секој елемент е (train_graphs, test_graphs) за еден fold
    """
    random_state = random_state or config.RANDOM_SEED
    subject_ids, subject_labels = get_unique_subjects(graphs)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # StratifiedGroupKFold бара X, y, groups - овде секој субјект е
    # своја група (group), не секој граф поединечно
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