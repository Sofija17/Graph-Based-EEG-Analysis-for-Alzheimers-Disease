"""Compare completed Pearson, Spearman and coherence GCN experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

import config


METHODS = ("pearson", "spearman", "coherence")


def latest_run(method):
    root = config.RESULTS_DIR / f"{method}_reference"
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No completed run in {root}")
    return runs[-1]


def bootstrap_auc_difference(reference, candidate, repetitions=5000):
    """Subject bootstrap CI for candidate AUC minus Pearson AUC."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    labels = reference["true_label"].to_numpy()
    reference_scores = reference["ad_probability"].to_numpy()
    candidate_scores = candidate["ad_probability"].to_numpy()
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(
            roc_auc_score(sampled_labels, candidate_scores[indices])
            - roc_auc_score(sampled_labels, reference_scores[indices])
        )
    return np.quantile(differences, [0.025, 0.975])


def main():
    summaries = []
    predictions = {}
    split_tables = {}

    for method in METHODS:
        run = latest_run(method)
        with (run / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        table = pd.read_csv(run / "oof_subject_predictions.csv").sort_values("subject_id")
        split_tables[method] = pd.read_csv(run / "fold_subject_splits.csv").sort_values(
            ["fold", "split", "subject_id"]
        ).reset_index(drop=True)
        predictions[method] = table.reset_index(drop=True)
        summaries.append({
            "method": method,
            "run_directory": str(run),
            "accuracy": summary["accuracy"],
            "precision": summary["precision"],
            "sensitivity": summary["recall_sensitivity"],
            "specificity": summary["specificity"],
            "f1": summary["f1"],
            "pooled_oof_roc_auc": summary["roc_auc"],
            "fold_roc_auc_mean": summary["fold_mean"]["roc_auc"],
            "fold_roc_auc_sd": summary["fold_standard_deviation"]["roc_auc"],
        })

    reference_splits = split_tables["pearson"]
    if not all(reference_splits.equals(split_tables[method]) for method in METHODS[1:]):
        raise ValueError("Connectivity experiments do not use identical subject folds")
    reference = predictions["pearson"]
    if not all(reference["subject_id"].equals(predictions[method]["subject_id"]) for method in METHODS[1:]):
        raise ValueError("OOF subject ordering differs between methods")

    paired_rows = []
    reference_correct = reference["prediction"].to_numpy() == reference["true_label"].to_numpy()
    for method in METHODS[1:]:
        candidate = predictions[method]
        candidate_correct = candidate["prediction"].to_numpy() == candidate["true_label"].to_numpy()
        reference_only = int(np.sum(reference_correct & ~candidate_correct))
        candidate_only = int(np.sum(~reference_correct & candidate_correct))
        discordant = reference_only + candidate_only
        mcnemar_p = (
            binomtest(min(reference_only, candidate_only), discordant, 0.5).pvalue
            if discordant else 1.0
        )
        low, high = bootstrap_auc_difference(reference, candidate)
        paired_rows.append({
            "comparison": f"{method}_minus_pearson",
            "prediction_agreement": float(np.mean(
                candidate["prediction"].to_numpy() == reference["prediction"].to_numpy()
            )),
            "pearson_only_correct": reference_only,
            "candidate_only_correct": candidate_only,
            "exact_mcnemar_p_value": float(mcnemar_p),
            "auc_difference": float(
                roc_auc_score(candidate["true_label"], candidate["ad_probability"])
                - roc_auc_score(reference["true_label"], reference["ad_probability"])
            ),
            "auc_difference_bootstrap_ci_low": float(low),
            "auc_difference_bootstrap_ci_high": float(high),
        })

    output_dir = config.RESULTS_DIR / "connectivity_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "method_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_comparisons.csv", index=False)

    metrics = ["accuracy", "sensitivity", "specificity", "f1", "pooled_oof_roc_auc"]
    plot_frame = summary_frame.melt(
        id_vars="method", value_vars=metrics, var_name="metric", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, method in enumerate(METHODS):
        subset = plot_frame[plot_frame["method"] == method]
        positions = np.arange(len(metrics)) + (index - 1) * 0.24
        ax.bar(positions, subset["score"], width=0.24, label=method.title())
    ax.set_xticks(np.arange(len(metrics)), [name.replace("_", "\n") for name in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Subject-level GCN comparison using identical folds")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_method_comparison.png", dpi=180)
    plt.close(fig)

    print(summary_frame.to_string(index=False))
    print("\nPaired comparisons versus Pearson:")
    print(pd.DataFrame(paired_rows).to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
