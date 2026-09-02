"""Subject-level qEEG biomarker analysis for AD versus CN.

The graph dataset contains one graph per EEG epoch.  Statistical tests must not
treat those epochs as independent samples, so this script first averages every
feature within each subject and only then compares the AD and CN groups.
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu

import config
from step_06_split_graph_dataset import load_graphs


CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
BAND_NAMES = list(config.FREQ_BANDS)
RATIO_DEFINITIONS = {
    "theta_alpha_ratio": ("theta", "alpha"),
    "delta_alpha_ratio": ("delta", "alpha"),
    "slow_fast_ratio": (("delta", "theta"), ("alpha", "beta")),
}
EPSILON = 1e-8


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ratio_values(features):
    """Return epoch x channel arrays for clinically interpretable ratios."""
    band_index = {name: index for index, name in enumerate(BAND_NAMES)}
    values = {}
    for ratio_name, (numerator, denominator) in RATIO_DEFINITIONS.items():
        numerator_names = (numerator,) if isinstance(numerator, str) else numerator
        denominator_names = (
            (denominator,) if isinstance(denominator, str) else denominator
        )
        numerator_value = sum(features[..., band_index[name]] for name in numerator_names)
        denominator_value = sum(
            features[..., band_index[name]] for name in denominator_names
        )
        values[ratio_name] = numerator_value / (denominator_value + EPSILON)
    return values


def aggregate_subject_features(graphs):
    """Build one channel-wise feature vector per subject."""
    grouped = {}
    for graph in graphs:
        subject_id = str(graph.subject_id)
        label = int(graph.y.item())
        if subject_id not in grouped:
            grouped[subject_id] = {"label": label, "features": []}
        elif grouped[subject_id]["label"] != label:
            raise ValueError(f"Inconsistent labels for {subject_id}")
        grouped[subject_id]["features"].append(graph.x.detach().cpu().numpy())

    subjects = []
    for subject_id, data in sorted(grouped.items()):
        epoch_features = np.stack(data["features"])
        if epoch_features.shape[1:] != (len(CHANNEL_NAMES), len(BAND_NAMES)):
            raise ValueError(
                f"Unexpected feature shape for {subject_id}: {epoch_features.shape}"
            )
        ratios = _ratio_values(epoch_features)
        subjects.append({
            "subject_id": subject_id,
            "label": data["label"],
            "group": "AD" if data["label"] == 1 else "CN",
            "n_segments": len(epoch_features),
            "band_values": epoch_features.mean(axis=0),
            "ratio_values": {
                name: values.mean(axis=0) for name, values in ratios.items()
            },
        })
    return subjects


def make_long_rows(subjects):
    rows = []
    for subject in subjects:
        common = {
            "subject_id": subject["subject_id"],
            "label": subject["label"],
            "group": subject["group"],
            "n_segments": subject["n_segments"],
        }
        for channel_index, channel in enumerate(CHANNEL_NAMES):
            for band_index, band in enumerate(BAND_NAMES):
                rows.append({
                    **common,
                    "channel": channel,
                    "biomarker": f"relative_{band}_power",
                    "value": float(subject["band_values"][channel_index, band_index]),
                })
            for ratio_name, values in subject["ratio_values"].items():
                rows.append({
                    **common,
                    "channel": channel,
                    "biomarker": ratio_name,
                    "value": float(values[channel_index]),
                })
    return rows


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction with monotone adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def compute_statistics(rows):
    grouped = {}
    for row in rows:
        key = (row["channel"], row["biomarker"])
        grouped.setdefault(key, {"AD": [], "CN": []})[row["group"]].append(row["value"])

    results = []
    for (channel, biomarker), values in sorted(grouped.items()):
        ad = np.asarray(values["AD"])
        cn = np.asarray(values["CN"])
        statistic, p_value = mannwhitneyu(ad, cn, alternative="two-sided")
        # Positive rank-biserial correlation means higher values in AD.
        effect_size = 2.0 * statistic / (len(ad) * len(cn)) - 1.0
        results.append({
            "channel": channel,
            "biomarker": biomarker,
            "n_ad": len(ad),
            "n_cn": len(cn),
            "ad_mean": float(ad.mean()),
            "ad_median": float(np.median(ad)),
            "cn_mean": float(cn.mean()),
            "cn_median": float(np.median(cn)),
            "mean_difference_ad_minus_cn": float(ad.mean() - cn.mean()),
            "mann_whitney_u": float(statistic),
            "p_value": float(p_value),
            "rank_biserial_effect": float(effect_size),
        })

    q_values = benjamini_hochberg([result["p_value"] for result in results])
    for result, q_value in zip(results, q_values):
        result["fdr_q_value"] = float(q_value)
        result["significant_fdr_0_05"] = bool(q_value < 0.05)
    return results


def compute_group_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["group"], row["channel"], row["biomarker"])
        grouped.setdefault(key, []).append(row["value"])
    summary = []
    for (group, channel, biomarker), values in sorted(grouped.items()):
        array = np.asarray(values)
        summary.append({
            "group": group,
            "channel": channel,
            "biomarker": biomarker,
            "n_subjects": len(array),
            "mean": float(array.mean()),
            "standard_deviation": float(array.std(ddof=1)),
            "median": float(np.median(array)),
            "q1": float(np.quantile(array, 0.25)),
            "q3": float(np.quantile(array, 0.75)),
        })
    return summary


def plot_heatmaps(statistics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    biomarkers = [f"relative_{band}_power" for band in BAND_NAMES] + list(RATIO_DEFINITIONS)
    for biomarker in biomarkers:
        selected = {row["channel"]: row for row in statistics if row["biomarker"] == biomarker}
        effects = np.array([[selected[channel]["rank_biserial_effect"] for channel in CHANNEL_NAMES]])
        q_values = np.array([[selected[channel]["fdr_q_value"] for channel in CHANNEL_NAMES]])
        annotations = np.array([[
            f"{effect:.2f}{'*' if q < 0.05 else ''}"
            for effect, q in zip(effects[0], q_values[0])
        ]])
        fig, ax = plt.subplots(figsize=(14, 2.6))
        sns.heatmap(
            effects, annot=annotations, fmt="", cmap="coolwarm", center=0,
            vmin=-1, vmax=1, xticklabels=CHANNEL_NAMES, yticklabels=[biomarker], ax=ax,
            cbar_kws={"label": "Rank-biserial effect (AD higher → positive)"},
        )
        ax.set_title(f"AD vs CN: {biomarker} (* FDR q < 0.05)")
        fig.tight_layout()
        fig.savefig(output_dir / f"effect_{biomarker}.png", dpi=180)
        plt.close(fig)


def _channel_positions_2d():
    """Return standard 10-20 positions, including legacy T3/T4/T5/T6 names."""
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    xyz = np.array([positions[aliases.get(channel, channel)] for channel in CHANNEL_NAMES])
    return xyz[:, :2]


def plot_topomaps(statistics, output_dir):
    """Plot spatial distributions of AD-versus-CN rank-biserial effects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = _channel_positions_2d()
    biomarkers = [f"relative_{band}_power" for band in BAND_NAMES] + list(RATIO_DEFINITIONS)
    for biomarker in biomarkers:
        selected = {row["channel"]: row for row in statistics if row["biomarker"] == biomarker}
        effects = np.array([selected[channel]["rank_biserial_effect"] for channel in CHANNEL_NAMES])
        significant = np.array([selected[channel]["fdr_q_value"] < 0.05 for channel in CHANNEL_NAMES])
        fig, ax = plt.subplots(figsize=(6, 5.5))
        image, _ = mne.viz.plot_topomap(
            effects,
            positions,
            axes=ax,
            show=False,
            cmap="coolwarm",
            vlim=(-1, 1),
            names=CHANNEL_NAMES,
            mask=significant,
            mask_params={
                "marker": "o", "markerfacecolor": "none", "markeredgecolor": "black",
                "linewidth": 1.5, "markersize": 8,
            },
        )
        fig.colorbar(image, ax=ax, label="Rank-biserial effect (AD higher → positive)")
        ax.set_title(f"AD vs CN: {biomarker}\noutlined electrodes: FDR q < 0.05")
        fig.tight_layout()
        fig.savefig(output_dir / f"topomap_{biomarker}.png", dpi=180)
        plt.close(fig)


def plot_top_features(rows, statistics, output_dir, count=8):
    output_dir.mkdir(parents=True, exist_ok=True)
    top = sorted(statistics, key=lambda row: (row["fdr_q_value"], -abs(row["rank_biserial_effect"])))[:count]
    for rank, result in enumerate(top, start=1):
        selected = [
            row for row in rows
            if row["channel"] == result["channel"] and row["biomarker"] == result["biomarker"]
        ]
        selected_frame = pd.DataFrame(selected)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.boxplot(data=selected_frame, x="group", y="value", order=["CN", "AD"], ax=ax)
        sns.stripplot(
            data=selected_frame, x="group", y="value", order=["CN", "AD"],
            color="black", alpha=0.65, jitter=0.18, ax=ax,
        )
        ax.set_title(
            f"{result['channel']} – {result['biomarker']}\n"
            f"effect={result['rank_biserial_effect']:.2f}, FDR q={result['fdr_q_value']:.3g}"
        )
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(output_dir / f"top_{rank:02d}_{result['channel']}_{result['biomarker']}.png", dpi=180)
        plt.close(fig)


def main():
    graphs = load_graphs()
    subjects = aggregate_subject_features(graphs)
    rows = make_long_rows(subjects)
    statistics = compute_statistics(rows)
    summary = compute_group_summary(rows)

    output_dir = config.RESULTS_DIR / "biomarker_analysis" / "relative_power"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "subject_channel_biomarkers.csv", rows)
    _write_csv(output_dir / "group_summary.csv", summary)
    _write_csv(output_dir / "statistical_results.csv", statistics)
    _write_csv(
        output_dir / "significant_fdr_results.csv",
        [row for row in statistics if row["significant_fdr_0_05"]],
    )
    plot_heatmaps(statistics, output_dir / "figures" / "effect_heatmaps")
    plot_topomaps(statistics, output_dir / "figures" / "topomaps")
    plot_top_features(rows, statistics, output_dir / "figures" / "top_features")

    metadata = {
        "analysis_unit": "subject",
        "n_subjects": len(subjects),
        "n_ad": sum(subject["label"] for subject in subjects),
        "n_cn": sum(subject["label"] == 0 for subject in subjects),
        "bands": BAND_NAMES,
        "ratios": RATIO_DEFINITIONS,
        "statistical_test": "two-sided Mann-Whitney U",
        "effect_size": "rank-biserial correlation; positive means AD higher",
        "multiple_testing": "Benjamini-Hochberg FDR across all channel-biomarker tests",
        "fdr_alpha": 0.05,
        "significant_tests": sum(row["significant_fdr_0_05"] for row in statistics),
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Subjects: {metadata['n_subjects']} (AD={metadata['n_ad']}, CN={metadata['n_cn']})")
    print(f"Tests: {len(statistics)}")
    print(f"Significant after FDR: {metadata['significant_tests']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
