# train.py
import os
import json
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm

from data.dataset import get_dataloaders
from configs.finetune_config import CONFIG
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[device] using {device}")

    # 1. Load data
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=cfg["dataset_name"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        num_workers=cfg["num_workers"],
        data_fraction=cfg["data_fraction"],
    )

    # 2. Build model
    model = timm.create_model(
        cfg["model_name"],
        pretrained=cfg["pretrained"]
    )

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    model = model.to(device)

    print(f"[model] loaded: {cfg['model_name']}")
    print(f"[model] head changed to {num_classes} classes")

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    best_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 4. Training loop
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch [{epoch + 1}/{cfg['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg,
                },
                save_dir=cfg["save_dir"],
                filename="best_model.pth"
            )

        # Save history
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Save JSON
    json_path = os.path.join(cfg["save_dir"], "history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"[history] saved json to {json_path}")

    # Save CSV
    csv_path = os.path.join(cfg["save_dir"], "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])

    print(f"[history] saved csv to {csv_path}")
    
    print(f"\n[done] best val acc = {best_acc:.4f}")

if __name__ == "__main__":
    main()