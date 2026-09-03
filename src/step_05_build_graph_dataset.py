"""
Batch script: for each AD/CN subject in the dataset, run the full
pipeline (load -> epoch -> features -> connectivity -> graphs)
and save ALL graphs from all subjects into one file on disk.
"""

import argparse
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


def process_subject(subject_id, label, connectivity_method=None):
    """
    Run the full pipeline for ONE subject.

    Returns
    -------
    list of PyG Data graphs (one per epoch), or None if the subject
    could not be processed (e.g. file does not exist)
    """
    epochs = load_and_epoch_subject(subject_id)
    epochs_data = epochs.get_data()
    connectivity_method = connectivity_method or config.CONNECTIVITY_METHOD

    # Feature extraction
    band_powers = extract_band_powers(epochs)
    feature_matrix = band_powers_to_feature_matrix(band_powers)
    feature_matrix = compute_relative_power(feature_matrix)

    # Connectivity + threshold  per epoch
    conn_all = compute_connectivity_all_epochs(
        epochs_data,
        method=connectivity_method,
        sfreq=float(epochs.info["sfreq"]),
    )
    conn_all_thresholded = np.array([
        threshold_connectivity(conn_all[i], top_k_percent=config.TOP_K_EDGES)
        for i in range(conn_all.shape[0])
    ])

    # Build graphs
    graphs = build_graphs_for_subject(feature_matrix, conn_all_thresholded, label)

    # Important: store subject_id in every graph; it is needed later
    # for subject-wise train/test splitting so epochs from the same
    # subject are not mixed into train and test at the same time.
    for g in graphs:
        g.subject_id = subject_id
        g.connectivity_method = connectivity_method

    return graphs


def build_full_dataset(connectivity_method=None):
    """
    Run process_subject() for ALL AD/CN subjects with progress logs
    and return one large list of all graphs.
    """
    subjects = get_ad_cn_subjects()
    connectivity_method = connectivity_method or config.CONNECTIVITY_METHOD
    print(f"Connectivity method: {connectivity_method}")
    print(f"Вкупно субјекти за обработка: {len(subjects)}\n")

    all_graphs = []
    failed_subjects = []

    start_time = time.time()

    for idx, (subject_id, label) in enumerate(subjects, start=1):
        label_name = "AD" if label == 1 else "CN"
        print(f"[{idx}/{len(subjects)}] Обработувам {subject_id} ({label_name})...", end=" ")

        try:
            graphs = process_subject(subject_id, label, connectivity_method)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Build an EEG graph dataset.")
    parser.add_argument(
        "--connectivity-method",
        choices=("pearson", "spearman", "coherence"),
        default=config.CONNECTIVITY_METHOD,
    )
    parser.add_argument("--output", default=None, help="Filename inside data/graphs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or f"all_graphs_{args.connectivity_method}.pt"
    graphs = build_full_dataset(args.connectivity_method)
    save_graphs(graphs, output)
