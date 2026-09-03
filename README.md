# Graph-Based EEG Analysis for Alzheimer's Disease

This project investigates whether resting-state EEG recordings can be represented as graphs and used to distinguish subjects with Alzheimer's Disease (AD) from cognitively normal controls (CN).

The main idea is to model each EEG epoch as a brain connectivity graph. EEG channels are represented as graph nodes, qEEG spectral features are used as node features, and functional connectivity between channels is represented as weighted graph edges.

## Objective

The objective is to build an EEG graph-learning pipeline for binary subject classification:

- `AD` - Alzheimer's Disease
- `CN` - Cognitively Normal control

Only AD and CN subjects are used. FTD subjects are excluded because the project is framed as a binary AD vs CN classification task.

## Methodology

The pipeline consists of the following steps:

1. Load EEG recordings and subject labels from `participants.tsv`.
2. Split each EEG recording into fixed-length 4-second epochs with no overlap.
3. Remove epochs that overlap with `boundary` discontinuities.
4. Extract qEEG spectral features using Welch Power Spectral Density (PSD).
5. Compute relative band power for delta, theta, alpha, and beta bands.
6. Compute functional connectivity between EEG channels.
7. Keep only the strongest 30% of connections to reduce graph density and noise.
8. Convert each EEG epoch into a PyTorch Geometric graph.
9. Train and evaluate a subject-level GCN model using 5-fold subject-wise cross-validation.
10. Analyze spectral and connectivity biomarkers at subject level.

## EEG Features

For each EEG epoch and each channel, Power Spectral Density is computed in the 0.5-30 Hz range using Welch's method. The signal power is then summarized into four clinically meaningful frequency bands:

- delta: 0.5-4 Hz
- theta: 4-8 Hz
- alpha: 8-13 Hz
- beta: 13-30 Hz

The absolute band powers are converted into relative power values. This makes the features more comparable across subjects, because absolute EEG power can vary due to technical and biological factors.

Each graph node therefore contains four features:

```text
relative delta power
relative theta power
relative alpha power
relative beta power
```

## Functional Connectivity

The reference model uses Pearson correlation as the functional connectivity method. For each EEG epoch, Pearson correlation is computed between all pairs of EEG channels.

The project also supports Spearman correlation and coherence for comparison, but the main reference experiment is based on Pearson connectivity.

To avoid overly dense graphs, only the strongest 30% of connections are retained. With 19 EEG channels, there are 171 unique undirected channel pairs, so keeping 30% corresponds to approximately 51 strongest connections per epoch.

The strongest edges are selected based on the absolute value of the connectivity score. This means that both strong positive and strong negative correlations can be treated as strong functional relationships. The GCN uses non-negative edge magnitudes in `edge_attr`, while the original signed values are retained separately in `signed_edge_attr`.

## Graph Construction

Each EEG epoch is converted into one graph:

- nodes: EEG channels
- node features: relative delta, theta, alpha, and beta power
- edges: thresholded functional connectivity values
- label: subject label, either AD or CN

Because diagnosis is defined at subject level, all epoch graphs from the same subject receive the same label. Each graph also stores `subject_id`, which is essential for preventing data leakage during evaluation.

## Model

The main model is implemented in:

```text
src/step_07_train_gcn_subject_level.py
```

The older file:

```text
src/step_07_train_gcn.py
```

is kept as an initial epoch-level implementation, but it is not used as the final model.

The final model is a Graph Convolutional Network (GCN) with two `GCNConv` layers. The first layer learns from directly connected EEG channels, while the second layer allows information to propagate through neighbors of neighbors. After the graph convolution layers, `global_mean_pool` aggregates the 19 node representations into one vector representation for the whole EEG graph. A final linear classifier converts this graph-level representation into two output scores: CN and AD.

## Validation

Evaluation is performed using 5-fold subject-wise cross-validation. In each fold, a new GCN model is trained from scratch.

Within each fold:

1. The training set is used to update the model weights.
2. The validation set is used to monitor validation loss, apply early stopping, select the best epoch, and choose the decision threshold.
3. The test set is used only at the end, after the model and threshold have already been selected.

All epochs from the same subject always stay in the same split. This prevents data leakage, because the model never sees epochs from the same subject in both training and testing.

Early stopping is based on validation loss. Training can run for up to 100 epochs, but it stops earlier if validation loss does not improve for 15 consecutive epochs. The model from the epoch with the lowest validation loss is restored and used for testing.

The final AD/CN decision is made at subject level. The GCN first predicts `P(AD)` for each EEG epoch. Then all epoch probabilities from the same subject are averaged:

```text
subject_score = mean P(AD) across all epochs of that subject
```

The decision threshold is not fixed at 0.5. Instead, it is selected dynamically from the validation subjects using Youden's J statistic:

```text
J = sensitivity + specificity - 1
```

This selects a threshold that balances AD detection and CN detection. The selected threshold is then applied once to the held-out test subjects.

## Results

The Pearson reference GCN model achieves the following subject-level results:

- accuracy: 75.4%
- precision: 77.8%
- sensitivity: 77.8%
- specificity: 72.4%
- F1-score: 77.8%
- pooled out-of-fold ROC-AUC: 0.749

Out of 65 subjects, the model correctly classifies 49 subjects and misclassifies 16 subjects.

The confusion matrix is:

| | Predicted CN | Predicted AD |
| --- | ---: | ---: |
| True CN | 21 | 8 |
| True AD | 8 | 28 |

Fold-level results for the Pearson reference model:

| Fold | Best Epoch | Threshold | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 0.582 | 61.5% | 0.667 | 0.667 |
| 2 | 1 | 0.534 | 92.3% | 0.933 | 0.975 |
| 3 | 4 | 0.658 | 76.9% | 0.769 | 0.929 |
| 4 | 15 | 0.564 | 76.9% | 0.727 | 0.714 |
| 5 | 50 | 0.503 | 69.2% | 0.778 | 0.762 |

Average fold results:

- accuracy: 0.754 +/- 0.102
- F1-score: 0.775 +/- 0.088
- ROC-AUC: 0.809 +/- 0.121

## Connectivity Method Comparison

Pearson, Spearman, and coherence connectivity were compared using the same subject-wise folds, node features, random seeds, and GCN hyperparameters.

| Method | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pearson | 75.4% | 77.8% | 72.4% | 77.8% | 0.749 |
| Spearman | 76.9% | 77.8% | 75.9% | 78.9% | 0.747 |
| Coherence | 78.5% | 69.4% | 89.7% | 78.1% | 0.748 |

Coherence achieves the highest accuracy and specificity, but lower sensitivity. Pearson and Spearman produce more balanced results. Since ROC-AUC is very similar across all three methods, no connectivity method can be considered clearly superior based on these results alone.

## Biomarker Analysis

Biomarker analysis is performed at subject level. First, biomarker values are averaged across all EEG epochs for each subject. Then the AD and CN groups are compared statistically.

The analysis uses:

- Mann-Whitney U test
- rank-biserial effect size
- Benjamini-Hochberg FDR correction

Mann-Whitney U is used because it does not assume normally distributed data. FDR correction is applied because many biomarkers and channels are tested, which increases the chance of false positives.

The spectral qEEG analysis shows that AD subjects tend to have:

- higher theta/alpha ratio
- lower relative alpha power

These effects are especially visible in posterior EEG channels such as O2 and T5, suggesting EEG slowing in AD.

The connectivity biomarker analysis shows regional changes in network organization. Some channels, such as O2 and T6, show stronger connectivity in AD, while other channels, such as F7 and F4, show weaker connectivity. This suggests that Alzheimer's Disease is associated with regional changes in functional connectivity rather than a simple global increase or decrease.

## Bias And Leakage Prevention

The project includes several measures to reduce bias and avoid leakage:

- all epochs from one subject remain in the same split;
- subject-wise cross-validation is used;
- subject-balanced loss prevents subjects with more epochs from dominating training;
- early stopping is based on validation loss;
- the decision threshold is selected without using test data;
- each subject receives exactly one out-of-fold test prediction;
- FDR correction is applied in biomarker analysis.

## Limitations

The results should be interpreted as exploratory, not clinical. The main limitations are:

- limited sample size: 65 subjects;
- no external validation on an independent cohort;
- the graph sparsification threshold is fixed at the strongest 30% of edges;
- the model is not a clinical diagnostic system.

## How To Run

Build the graph dataset:

```powershell
python src/step_05_build_graph_dataset.py
```

Train the main subject-level GCN model:

```powershell
python src/step_07_train_gcn_subject_level.py
```

Build and evaluate Spearman connectivity:

```powershell
python src/step_05_build_graph_dataset.py --connectivity-method spearman --output all_graphs_spearman.pt
python src/step_07_train_gcn_subject_level.py --graphs-file all_graphs_spearman.pt --connectivity-method spearman
```

Build and evaluate coherence connectivity:

```powershell
python src/step_05_build_graph_dataset.py --connectivity-method coherence --output all_graphs_coherence.pt
python src/step_07_train_gcn_subject_level.py --graphs-file all_graphs_coherence.pt --connectivity-method coherence
```

## Conclusion

This project shows that resting-state EEG can be represented as a graph and used for graph-based AD vs CN classification. The GCN model performs above chance level and captures both spectral and connectivity-based information. The clearest biomarker findings are reduced alpha power and increased theta/alpha ratio in AD, especially in posterior EEG channels. Connectivity changes are also present, but they appear to be regional rather than global. The results are promising, but further validation on larger and independent datasets is needed.
