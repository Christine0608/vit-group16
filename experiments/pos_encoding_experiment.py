"""
experiments/pos_encoding_experiment.py
Author: Zhou Danding
Desc:   Compare learnable vs sinusoidal positional encoding on CIFAR-10.
        Runs two short training sessions and saves results to outputs/.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.vit import ViT
from utils.seed import set_seed

EXPERIMENT_CONFIG = {
    "dataset_name": "cifar10",
    "batch_size": 32,
    "img_size": 224,
    "num_workers": 2,
    "data_fraction": 0.2,   # 用20%数据，快速跑完
    "epochs": 5,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "seed": 42,
}

def train_and_eval(pos_type: str, cfg: dict) -> dict:
    set_seed(cfg["seed"])

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{'='*50}")
    print(f"pos_encoding_type = {pos_type}  |  device = {device}")
    print('='*50)

    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=cfg["dataset_name"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        num_workers=cfg["num_workers"],
        data_fraction=cfg["data_fraction"],
    )

    model = ViT(
        img_size=cfg["img_size"],
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        hidden_size=192,
        num_layers=4,
        mlp_dim=768,
        num_heads=3,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        classifier="token",
        pos_encoding_type=pos_type,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Eval
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples

        print(f"Epoch {epoch+1} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


if __name__ == "__main__":
    results = {}
    for pos_type in ["learnable", "sinusoidal"]:
        results[pos_type] = train_and_eval(pos_type, EXPERIMENT_CONFIG)

    # Save results
    save_dir = "outputs/pos_encoding_experiment"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Print summary
    print("\n===== SUMMARY =====")
    for pos_type, history in results.items():
        best_val = max(history["val_acc"])
        print(f"{pos_type:12s} | best val acc = {best_val:.4f}")

    print(f"\nResults saved to {save_dir}/results.json")