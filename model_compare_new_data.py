import os
import json
import csv
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm

# Import updated data pipeline and utilities
from data.extension_dataset import get_dataloaders
from configs.new_data_finetune_config import CONFIG
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from models.ResNet_CNN import resnet18

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for a single epoch."""
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates the model on the validation/test set."""
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

    return total_loss / total_samples, total_correct / total_samples

def print_and_save_tables(global_val_history, best_acc_summary, epochs, save_path):
    """
    Generates and prints formatted ASCII tables for epoch progression and best accuracies 
    across Path, Blood, and Tissue datasets, and exports to CSV.
    """
    print("\n" + "="*120)
    print(f" TABLE 1: VALIDATION ACCURACY PROGRESSION ACROSS {epochs} EPOCHS")
    print("="*120)
    
    # Define table headers for 3 datasets x 2 models
    header = f"{'Epoch':<10} | {'Path-ViT':<12} | {'Path-CNN':<12} | {'Blood-ViT':<12} | {'Blood-CNN':<12} | {'Tissu-ViT':<12} | {'Tissu-CNN':<12}"
    print(header)
    print("-" * len(header))
    
    # Prepare CSV export
    csv_data = [["Epoch", "PathMNIST_ViT", "PathMNIST_CNN", "BloodMNIST_ViT", "BloodMNIST_CNN", "TissueMNIST_ViT", "TissueMNIST_CNN"]]

    for epoch in range(epochs):
        ep_str = f"Epoch {epoch+1}"
        
        # Extract accuracy values safely
        p_vit = global_val_history['pathmnist']['vit_pretrained'][epoch] if epoch < len(global_val_history['pathmnist']['vit_pretrained']) else 0.0
        p_cnn = global_val_history['pathmnist']['cnn'][epoch] if epoch < len(global_val_history['pathmnist']['cnn']) else 0.0
        b_vit = global_val_history['bloodmnist']['vit_pretrained'][epoch] if epoch < len(global_val_history['bloodmnist']['vit_pretrained']) else 0.0
        b_cnn = global_val_history['bloodmnist']['cnn'][epoch] if epoch < len(global_val_history['bloodmnist']['cnn']) else 0.0
        t_vit = global_val_history['tissuemnist']['vit_pretrained'][epoch] if epoch < len(global_val_history['tissuemnist']['vit_pretrained']) else 0.0
        t_cnn = global_val_history['tissuemnist']['cnn'][epoch] if epoch < len(global_val_history['tissuemnist']['cnn']) else 0.0
        
        row_str = (f"{ep_str:<10} | {p_vit*100:>10.2f}% | {p_cnn*100:>10.2f}% | "
                   f"{b_vit*100:>10.2f}% | {b_cnn*100:>10.2f}% | "
                   f"{t_vit*100:>10.2f}% | {t_cnn*100:>10.2f}%")
        print(row_str)
        csv_data.append([epoch+1, f"{p_vit:.4f}", f"{p_cnn:.4f}", f"{b_vit:.4f}", f"{b_cnn:.4f}", f"{t_vit:.4f}", f"{t_cnn:.4f}"])

    print("\n" + "="*100)
    print(" TABLE 2: BEST VALIDATION ACCURACY SUMMARY")
    print("="*100)
    best_header = f"{'Model':<20} | {'PathMNIST':<15} | {'BloodMNIST':<15} | {'TissueMNIST':<15}"
    print(best_header)
    print("-" * len(best_header))
    
    models = ['vit_pretrained', 'cnn']
    for model in models:
        p_best = best_acc_summary['pathmnist'].get(model, 0.0)
        b_best = best_acc_summary['bloodmnist'].get(model, 0.0)
        t_best = best_acc_summary['tissuemnist'].get(model, 0.0)
        print(f"{model:<20} | {p_best*100:>13.2f}% | {b_best*100:>13.2f}% | {t_best*100:>13.2f}%")
    print("="*100 + "\n")

    # Save to CSV
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, "epoch_tracking_summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print(f"[Export] Summary tables saved to {csv_file}")


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[System] Using device: {device}")

    # Triple datasets: Path, Blood, and the new Tissue
    datasets_to_test = ['pathmnist', 'bloodmnist', 'tissuemnist']
    models_to_test = ['vit_pretrained', 'cnn']
    
    # Trackers for the final tabular reports
    best_acc_summary = {ds: {} for ds in datasets_to_test}
    global_val_history = {ds: {model: [] for model in models_to_test} for ds in datasets_to_test}

    for dataset_name in datasets_to_test:
        print(f"\n{'='*60}")
        print(f" Commencing Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        for model_name in models_to_test:
            print(f"\n Initializing Model: {model_name}")

            # 1. Load Data
            # Note: Explicit fractions (0.6, 1.0, 0.3) are now handled internally by get_dataloaders
            train_loader, test_loader, num_classes = get_dataloaders(
                dataset_name=dataset_name,
                batch_size=cfg["batch_size"],
                img_size=cfg["img_size"],
                num_workers=cfg["num_workers"]
            )

            # Model creation
            if model_name == "vit_pretrained":
                model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)
            elif model_name == "cnn":
                model = resnet18(num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            model = model.to(device)

            save_dir = os.path.join("outputs", "extension_new_data_model_compare", dataset_name, model_name)
            os.makedirs(save_dir, exist_ok=True)

            # Save config for reproducibility
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=4)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            
            # --- Added Learning Rate Scheduler ---
            # Smoothes out the learning process in later epochs to prevent validation fluctuations
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

            best_acc = 0.0
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            for epoch in range(cfg["epochs"]):
                # Retrieve current learning rate for logging
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch + 1}/{cfg['epochs']}] - {model_name} (LR: {current_lr:.6f})")

                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, test_loader, criterion, device)
                
                # Step the scheduler at the end of each epoch
                scheduler.step()

                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Append to global tracker for final tables
                global_val_history[dataset_name][model_name].append(val_acc)

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
                        save_dir=save_dir, filename="best_model.pth"
                    )

            # Export history
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)

            csv_path = os.path.join(save_dir, "history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                for i in range(len(history["train_loss"])):
                    writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i], history["val_loss"][i], history["val_acc"][i]])
            
            best_acc_summary[dataset_name][model_name] = best_acc
            
            # Strict memory cleanup to avoid crashes on high-res datasets
            del model, train_loader, test_loader, optimizer, scheduler
            gc.collect()
            if device == "mps": torch.mps.empty_cache()

    # Final summary reports
    output_base_dir = os.path.join("outputs", "extension_new_data_model_compare")
    print_and_save_tables(global_val_history, best_acc_summary, cfg["epochs"], output_base_dir)


if __name__ == "__main__":
    main()