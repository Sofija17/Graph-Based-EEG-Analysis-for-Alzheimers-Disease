"""
Training loop за GCN моделот - вчитува графови, ги дели subject-wise,
тренира и печати прогрес на секои неколку епохи.
"""

import torch
from torch_geometric.loader import DataLoader

import config
from step_06_split_graph_dataset import load_graphs, subject_wise_train_test_split, verify_no_leakage
from models.gcn import GCN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Еден целосен помин низ train податоците (сите batch-ови)."""
    model.train()  # ставаме модел во "training режим" (важно за некои слоеви)
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()  # бришеме стари градиенти од претходниот batch
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)

        loss.backward()   # пресметува градиенти
        optimizer.step()  # ги ажурира тежините на моделот

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate_accuracy(model, loader, device):
    """Пресметува accuracy на моделот"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # не ни требаат градиенти при евалуација
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = out.argmax(dim=1)  # избери ја класата со најголема веројатност
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Користиме device: {device}\n")

    # --- Load & split ---
    graphs = load_graphs()
    train_graphs, test_graphs = subject_wise_train_test_split(graphs, test_size=0.2)
    verify_no_leakage(train_graphs, test_graphs)
    print()

    train_loader = DataLoader(train_graphs, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- Model ---
    num_node_features = train_graphs[0].x.shape[1]  # 4 (delta/theta/alpha/beta)
    model = GCN(num_node_features=num_node_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()  # стандарден loss за класификација

    # --- Training loop ---
    print("Почнува тренирање...\n")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % 10 == 0 or epoch == 1:
            train_acc = evaluate_accuracy(model, train_loader, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"Епоха {epoch:3d} | loss={train_loss:.4f} | "
                  f"train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")

    return model, train_loader, test_loader, device


if __name__ == "__main__":
    main()