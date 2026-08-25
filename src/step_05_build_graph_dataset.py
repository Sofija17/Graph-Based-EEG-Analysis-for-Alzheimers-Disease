"""
Batch скрипта: за секој AD/CN субјект во датасетот, го извршува целиот
pipeline (load -> epoch -> features -> connectivity -> graphs)
ги зачувува СИТЕ графови (од сите субјекти) во еден фајл на диск

"""

import time
import numpy as np
import torch

import config
from helpers.load_participants import get_ad_cn_subjects
from step_01_preprocessing import load_and_epoch_subject
from step_02_feature_extraction import (
    extract_band_powers,
    band_powers_to_feature_matrix,
    compute_relative_power,
)
from step_03_compute_connectivity import compute_connectivity_all_epochs, threshold_connectivity
from step_04_graph_builder import build_graphs_for_subject


def process_subject(subject_id, label):
    """
    Го извршува целиот pipeline за ЕДЕН субјект.

    Враќа
    -----
    list од PyG Data графови (еден по епоха), или None ако субјектот
    не можел да се обработи (пр. фајл не постои)
    """
    epochs = load_and_epoch_subject(subject_id)
    epochs_data = epochs.get_data()

    # Feature extraction
    band_powers = extract_band_powers(epochs)
    feature_matrix = band_powers_to_feature_matrix(band_powers)
    feature_matrix = compute_relative_power(feature_matrix)

    # Connectivity + threshold  per epoch
    conn_all = compute_connectivity_all_epochs(epochs_data)
    conn_all_thresholded = np.array([
        threshold_connectivity(conn_all[i], top_k_percent=config.TOP_K_EDGES)
        for i in range(conn_all.shape[0])
    ])

    # Градење графови
    graphs = build_graphs_for_subject(feature_matrix, conn_all_thresholded, label)

    # Важно: чуваме subject_id во секој граф, ќе ни треба подоцна
    # за subject-wise train/test split (за да не мешаме епохи од ист
    # субјект во train и test истовремено)
    for g in graphs:
        g.subject_id = subject_id

    return graphs


def build_full_dataset():
    """
    Го извршува process_subject() за СИТЕ AD/CN субјекти со логови
    за прогрес и враќа еден голем список од сите графови.
    """
    subjects = get_ad_cn_subjects()
    print(f"Вкупно субјекти за обработка: {len(subjects)}\n")

    all_graphs = []
    failed_subjects = []

    start_time = time.time()

    for idx, (subject_id, label) in enumerate(subjects, start=1):
        label_name = "AD" if label == 1 else "CN"
        print(f"[{idx}/{len(subjects)}] Обработувам {subject_id} ({label_name})...", end=" ")

        try:
            graphs = process_subject(subject_id, label)
            all_graphs.extend(graphs)
            print(f"OK - {len(graphs)} графови")
        except Exception as e:
            print(f"ГРЕШКА - {e}")
            failed_subjects.append((subject_id, str(e)))

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Завршено за {elapsed:.1f} сек")
    print(f"Вкупно графови: {len(all_graphs)}")
    print(f"Успешни субјекти: {len(subjects) - len(failed_subjects)}/{len(subjects)}")

    if failed_subjects:
        print(f"\nНеуспешни субјекти ({len(failed_subjects)}):")
        for subject_id, error in failed_subjects:
            print(f"  {subject_id}: {error}")

    return all_graphs


def save_graphs(graphs, filename="all_graphs.pt"):
    config.DATA_GRAPHS.mkdir(parents=True, exist_ok=True)
    save_path = config.DATA_GRAPHS / filename
    torch.save(graphs, save_path)
    print(f"\nЗачувано во: {save_path}")

if __name__ == "__main__":
    graphs = build_full_dataset()
    save_graphs(graphs)