"""Subject-level biomarkers from the existing thresholded Pearson graphs.

Important limitation: ``step_04_graph_builder.py`` stores absolute edge weights.
Consequently this analysis measures connectivity magnitude and graph topology,
not whether the original Pearson correlation was positive or negative.
"""

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.csgraph import shortest_path

import config
from step_06_split_graph_dataset import load_graphs
from step_09_biomarker_analysis import (
    CHANNEL_NAMES,
    _write_csv,
    compute_group_summary,
    compute_statistics,
)


def graph_metrics(graph):
    """Compute weighted topology metrics for one epoch graph."""
    n_channels = len(CHANNEL_NAMES)
    adjacency = np.zeros((n_channels, n_channels), dtype=float)
    edges = graph.edge_index.detach().cpu().numpy()
    weights = graph.edge_attr.detach().cpu().numpy().reshape(-1)
    adjacency[edges[0], edges[1]] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)

    upper = adjacency[np.triu_indices(n_channels, 1)]
    retained = upper[upper > 0]
    node_strength = adjacency.sum(axis=1)

    network = nx.from_numpy_array(adjacency)
    clustering_by_node = nx.clustering(
        network, nodes=range(n_channels), weight="weight"
    )
    clustering = np.array([
        clustering_by_node[index] for index in range(n_channels)
    ])

    distances = np.full_like(adjacency, np.inf)
    positive = adjacency > 0
    distances[positive] = 1.0 / adjacency[positive]
    np.fill_diagonal(distances, 0.0)
    paths = shortest_path(distances, directed=False)
    off_diagonal = ~np.eye(n_channels, dtype=bool)
    finite_paths = paths[off_diagonal & np.isfinite(paths)]
    weighted_efficiency = float(np.mean(1.0 / finite_paths)) if len(finite_paths) else 0.0

    return {
        "node_strength": node_strength,
        "node_clustering": clustering,
        "mean_edge_magnitude": float(retained.mean()) if len(retained) else 0.0,
        "density": float(np.count_nonzero(upper) / len(upper)),
        "mean_node_strength": float(node_strength.mean()),
        "mean_weighted_clustering": float(clustering.mean()),
        "weighted_global_efficiency": weighted_efficiency,
    }


def aggregate_subject_metrics(graphs):
    grouped = {}
    for index, graph in enumerate(graphs, start=1):
        subject_id = str(graph.subject_id)
        label = int(graph.y.item())
        if subject_id not in grouped:
            grouped[subject_id] = {"label": label, "metrics": []}
        elif grouped[subject_id]["label"] != label:
            raise ValueError(f"Inconsistent labels for {subject_id}")
        grouped[subject_id]["metrics"].append(graph_metrics(graph))
        if index % 1000 == 0:
            print(f"Processed {index}/{len(graphs)} graphs")

    subjects = []
    for subject_id, data in sorted(grouped.items()):
        metrics = data["metrics"]
        subjects.append({
            "subject_id": subject_id,
            "label": data["label"],
            "group": "AD" if data["label"] else "CN",
            "n_segments": len(metrics),
            "node_strength": np.mean([item["node_strength"] for item in metrics], axis=0),
            "node_clustering": np.mean([item["node_clustering"] for item in metrics], axis=0),
            **{
                name: float(np.mean([item[name] for item in metrics]))
                for name in (
                    "mean_edge_magnitude", "density", "mean_node_strength",
                    "mean_weighted_clustering", "weighted_global_efficiency",
                )
            },
        })
    return subjects


def make_rows(subjects):
    rows = []
    for subject in subjects:
        common = {
            "subject_id": subject["subject_id"],
            "label": subject["label"],
            "group": subject["group"],
            "n_segments": subject["n_segments"],
        }
        for channel_index, channel in enumerate(CHANNEL_NAMES):
            rows.append({
                **common, "channel": channel, "biomarker": "node_strength",
                "value": float(subject["node_strength"][channel_index]),
            })
            rows.append({
                **common, "channel": channel, "biomarker": "weighted_clustering",
                "value": float(subject["node_clustering"][channel_index]),
            })
        for biomarker in (
            "mean_edge_magnitude", "density", "mean_node_strength",
            "mean_weighted_clustering", "weighted_global_efficiency",
        ):
            rows.append({
                **common, "channel": "GLOBAL", "biomarker": biomarker,
                "value": subject[biomarker],
            })
    return rows


def plot_channel_effects(statistics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for biomarker in ("node_strength", "weighted_clustering"):
        selected = {row["channel"]: row for row in statistics if row["biomarker"] == biomarker}
        effects = np.array([[selected[channel]["rank_biserial_effect"] for channel in CHANNEL_NAMES]])
        annotations = np.array([[
            f"{selected[channel]['rank_biserial_effect']:.2f}"
            f"{'*' if selected[channel]['fdr_q_value'] < 0.05 else ''}"
            for channel in CHANNEL_NAMES
        ]])
        fig, ax = plt.subplots(figsize=(14, 2.6))
        sns.heatmap(
            effects, annot=annotations, fmt="", cmap="coolwarm", center=0,
            vmin=-1, vmax=1, xticklabels=CHANNEL_NAMES, yticklabels=[biomarker], ax=ax,
            cbar_kws={"label": "Rank-biserial effect (AD higher → positive)"},
        )
        ax.set_title(f"Pearson magnitude: {biomarker} (* FDR q < 0.05)")
        fig.tight_layout()
        fig.savefig(output_dir / f"effect_{biomarker}.png", dpi=180)
        plt.close(fig)


def plot_global_metrics(rows, statistics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for row in rows if row["channel"] == "GLOBAL"])
    lookup = {
        row["biomarker"]: row for row in statistics if row["channel"] == "GLOBAL"
    }
    for biomarker, subset in frame.groupby("biomarker"):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.boxplot(data=subset, x="group", y="value", order=["CN", "AD"], ax=ax)
        sns.stripplot(
            data=subset, x="group", y="value", order=["CN", "AD"],
            color="black", alpha=0.65, jitter=0.18, ax=ax,
        )
        result = lookup[biomarker]
        ax.set_title(
            f"{biomarker}\neffect={result['rank_biserial_effect']:.2f}, "
            f"FDR q={result['fdr_q_value']:.3g}"
        )
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(output_dir / f"global_{biomarker}.png", dpi=180)
        plt.close(fig)


def main():
    graphs = load_graphs()
    subjects = aggregate_subject_metrics(graphs)
    rows = make_rows(subjects)
    statistics = compute_statistics(rows)
    summary = compute_group_summary(rows)

    output_dir = config.RESULTS_DIR / "biomarker_analysis" / "pearson_connectivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "subject_connectivity_biomarkers.csv", rows)
    _write_csv(output_dir / "group_summary.csv", summary)
    _write_csv(output_dir / "statistical_results.csv", statistics)
    _write_csv(
        output_dir / "significant_fdr_results.csv",
        [row for row in statistics if row["significant_fdr_0_05"]],
    )
    plot_channel_effects(statistics, output_dir / "figures" / "channel_effects")
    plot_global_metrics(rows, statistics, output_dir / "figures" / "global_metrics")

    metadata = {
        "analysis_unit": "subject",
        "epoch_aggregation": "compute metric per epoch, then average within subject",
        "connectivity": "absolute thresholded Pearson weights",
        "top_k_edges": config.TOP_K_EDGES,
        "important_limitation": "correlation sign was discarded during graph construction",
        "n_subjects": len(subjects),
        "n_ad": sum(subject["label"] for subject in subjects),
        "n_cn": sum(subject["label"] == 0 for subject in subjects),
        "statistical_test": "two-sided Mann-Whitney U",
        "multiple_testing": "Benjamini-Hochberg FDR across all connectivity tests",
        "fdr_alpha": 0.05,
        "tests": len(statistics),
        "significant_tests": sum(row["significant_fdr_0_05"] for row in statistics),
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Subjects: {metadata['n_subjects']} (AD={metadata['n_ad']}, CN={metadata['n_cn']})")
    print(f"Tests: {metadata['tests']}")
    print(f"Significant after FDR: {metadata['significant_tests']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
