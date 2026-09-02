# Graph-Based EEG Analysis for Alzheimer's Disease

This project explores whether EEG-based graph representations can help distinguish Alzheimer's Disease (AD) subjects from cognitively normal (CN) controls.

The dataset contains resting-state eyes-closed EEG recordings. Only AD and CN subjects are used for binary classification,while FTD subjects are excluded.

## Goal

The goal is to classify subjects as:

- `AD` - Alzheimer's Disease
- `CN` - Cognitively Normal / Healthy Control

## Pipeline

1. Load EEG recordings and subject labels.
2. Split each EEG recording into fixed-length epochs.
3. Remove epochs that overlap with `boundary` discontinuities.
4. Extract qEEG band power features for each channel:
   - delta
   - theta
   - alpha
   - beta
5. Compute Pearson connectivity between EEG channels.
6. Convert each epoch into a graph:
   - nodes = EEG channels
   - node features = relative band powers
   - edges = functional connectivity between channels
7. Train a GCN on epoch-level graphs.
8. Aggregate epoch-level AD probabilities per subject.
9. Evaluate using subject-wise cross-validation.

## Model

The model is a Graph Convolutional Network (GCN). Each epoch is represented as one graph. The GCN predicts an AD probability for each graph, and the subject-level prediction is obtained by averaging probabilities across all epochs from the same subject.

A subject-balanced loss is used so that subjects with more epochs do not dominate training.

## Evaluation

Evaluation is performed using 5-fold subject-wise cross-validation. All epochs from one subject are kept in the same fold to avoid data leakage.

The subject-level decision threshold is selected on the validation set using Youden's J statistic and then applied to the held-out test subjects.

The original `step_07_train_gcn.py` is retained as the initial epoch-level
implementation. The main experiment is `step_07_train_gcn_subject_level.py`.
Each run creates a timestamped directory under
`results/pearson_reference/` containing:

- the complete experiment configuration;
- fold membership for every subject;
- fold-level and pooled subject-level metrics;
- out-of-fold probabilities and predictions for every subject;
- training history and loss curves;
- an out-of-fold ROC curve, confusion matrix, and probability plot.

Build the graph dataset first, then run the reference experiment:

```powershell
python src/step_05_build_graph_dataset.py
python src/step_07_train_gcn_subject_level.py
```

Alternative connectivity datasets can be built and evaluated without
overwriting the Pearson reference dataset:

```powershell
python src/step_05_build_graph_dataset.py --connectivity-method spearman --output all_graphs_spearman.pt
python src/step_07_train_gcn_subject_level.py --graphs-file all_graphs_spearman.pt --connectivity-method spearman

python src/step_05_build_graph_dataset.py --connectivity-method coherence --output all_graphs_coherence.pt
python src/step_07_train_gcn_subject_level.py --graphs-file all_graphs_coherence.pt --connectivity-method coherence
```

Newly built datasets retain signed connectivity values in `signed_edge_attr`;
the GCN continues to use non-negative magnitudes in `edge_attr` because signed
weights are not safe for the normalization performed by `GCNConv`.

## Notes

This project is intended as an exploratory analysis of potential EEG biomarkers for Alzheimer's Disease. Due to the small number of subjects, results should be interpreted carefully and not as a clinical diagnostic system.
