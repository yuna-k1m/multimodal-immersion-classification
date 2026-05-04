from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from Dataset import BiosignalDataset
from model import FusionModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    dataset: BiosignalDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for eeg, gsr, ppg, label in loader:
        eeg = eeg.to(device)
        gsr = gsr.to(device)
        ppg = ppg.to(device)
        label = label.to(device).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(eeg, gsr, ppg)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        total_correct += ((torch.sigmoid(logits) >= 0.5).float() == label).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for eeg, gsr, ppg, label in loader:
        eeg = eeg.to(device)
        gsr = gsr.to(device)
        ppg = ppg.to(device)
        label = label.to(device).view(-1, 1)

        logits, _ = model(eeg, gsr, ppg)
        loss = criterion(logits, label)

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        total_correct += ((torch.sigmoid(logits) >= 0.5).float() == label).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    y_true = []
    y_prob = []

    for eeg, gsr, ppg, label in loader:
        eeg = eeg.to(device)
        gsr = gsr.to(device)
        ppg = ppg.to(device)

        logits, _ = model(eeg, gsr, ppg)
        prob = torch.sigmoid(logits).view(-1)

        y_prob.extend(prob.cpu().numpy())
        y_true.extend(label.numpy())

    return np.asarray(y_true), np.asarray(y_prob)


def save_learning_curves(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multimodal EEG-GSR-PPG immersion classifier."
    )
    parser.add_argument("--data-dir", type=str, default="./data/DATA_SLICED")
    parser.add_argument("--split-dir", type=str, default="./data/data_splits")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "best_model.pt"
    curve_path = output_dir / "learning_curves.png"

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BiosignalDataset(args.data_dir, split="train", split_path=args.split_dir)
    val_dataset = BiosignalDataset(args.data_dir, split="val", split_path=args.split_dir)
    test_dataset = BiosignalDataset(args.data_dir, split="test", split_path=args.split_dir)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, device=device)
    test_loader = make_loader(test_dataset, args.test_batch_size, shuffle=False, device=device)

    print(f"Device: {device}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = FusionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break

    save_learning_curves(history, curve_path)

    model.load_state_dict(load_checkpoint(checkpoint_path, device))
    y_true, y_prob = predict(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nTest Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report\n", classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
