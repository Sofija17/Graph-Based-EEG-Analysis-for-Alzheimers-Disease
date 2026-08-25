# import torch
# from dataset_split import load_graphs
#
# graphs = load_graphs()
#
# nan_graphs = []
# for g in graphs:
#     has_nan = torch.isnan(g.x).any() or torch.isnan(g.edge_attr).any()
#     has_inf = torch.isinf(g.x).any() or torch.isinf(g.edge_attr).any()
#     if has_nan or has_inf:
#         nan_graphs.append(g.subject_id)
#
# print(f"Вкупно проблематични графови: {len(nan_graphs)} од {len(graphs)}")
# print(f"Засегнати субјекти: {set(nan_graphs)}")

import torch
from step_06_split_graph_dataset import load_graphs

graphs = load_graphs()

min_weight = min(g.edge_attr.min().item() for g in graphs)
max_weight = max(g.edge_attr.max().item() for g in graphs)
n_negative = sum((g.edge_attr < 0).sum().item() for g in graphs)

print(f"Мин edge weight: {min_weight}")
print(f"Макс edge weight: {max_weight}")
print(f"Вкупно негативни тежини: {n_negative}")