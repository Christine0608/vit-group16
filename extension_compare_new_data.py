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

# Model imports (matching your required structure)
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

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


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

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


def print_and_save_tables(global_val_history, best_acc_summary, epochs, save_path):
    """
    Generates and prints formatted ASCII tables for epoch progression and best accuracies 
    across Path and Blood datasets, and exports the summary to CSV.
    """
    print("\n" + "="*100)
    print(f" TABLE 1: VALIDATION ACCURACY PROGRESSION ACROSS {epochs} EPOCHS")
    print("="*100)
    
    # Define table headers for 2 datasets x 2 models
    header = (f"{'Epoch':<8} | {'Path-ViT-Pre':<14} | {'Path-CNN':<10} | "
              f"{'Blood-ViT-Pre':<14} | {'Blood-CNN':<10}")
    print(header)
    print("-" * len(header))
    
    # Prepare CSV export data
    csv_data = [["Epoch", "Path_ViT_Pre", "Path_CNN", "Blood_ViT_Pre", "Blood_CNN"]]

    for epoch in range(epochs):
        ep_str = f"Epoch {epoch+1}"
        
        # Extract accuracy values safely, defaulting to 0.0 if incomplete
        def get_acc(ds, mod, ep):
            hist = global_val_history[ds][mod]
            return hist[ep] if ep < len(hist) else 0.0

        p_v_pre = get_acc('pathmnist', 'vit_pretrained', epoch)
        p_cnn   = get_acc('pathmnist', 'cnn', epoch)
        
        b_v_pre = get_acc('bloodmnist', 'vit_pretrained', epoch)
        b_cnn   = get_acc('bloodmnist', 'cnn', epoch)
        
        row_str = (f"{ep_str:<8} | {p_v_pre*100:>13.2f}% | {p_cnn*100:>9.2f}% | "
                   f"{b_v_pre*100:>13.2f}% | {b_cnn*100:>9.2f}%")
        print(row_str)
        csv_data.append([epoch+1, f"{p_v_pre:.4f}", f"{p_cnn:.4f}", f"{b_v_pre:.4f}", f"{b_cnn:.4f}"])

    print("\n" + "="*85)
    print(" TABLE 2: BEST VALIDATION ACCURACY SUMMARY")
    print("="*85)
    best_header = f"{'Model':<20} | {'PathMNIST':<20} | {'BloodMNIST':<20}"
    print(best_header)
    print("-" * len(best_header))
    
    models = ['vit_pretrained', 'cnn']
    for model in models:
        p_best = best_acc_summary['pathmnist'].get(model, 0.0)
        b_best = best_acc_summary['bloodmnist'].get(model, 0.0)
        print(f"{model:<20} | {p_best*100:>19.2f}% | {b_best*100:>19.2f}%")
    print("="*85 + "\n")

    # Save tracking summary to CSV
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, "extension_model_compare_summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print(f"[Export] Summary tables successfully saved to {csv_file}")


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Hardware device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[System] Using device: {device}")

    # Scope of the experiment: Path and Blood datasets only
    datasets_to_test = ['pathmnist', 'bloodmnist']
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
            # Note: Dataset fractions are handled intrinsically by the extension_dataset pipeline
            train_loader, test_loader, num_classes = get_dataloaders(
                dataset_name=dataset_name,
                batch_size=cfg["batch_size"],
                img_size=cfg["img_size"],
                num_workers=cfg.get("num_workers", 0)
            )

            # 2. Build Model
            if model_name == "vit_pretrained":
                # Included the drop_rate injection to combat overfitting as discussed
                model = timm.create_model(
                    "vit_tiny_patch16_224", 
                    pretrained=True, 
                    num_classes=num_classes,
                    drop_rate=cfg.get("drop_rate", 0.3)
                )

            elif model_name == "cnn":
                model = resnet18(num_classes=num_classes)

            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            model = model.to(device)

            # 3. Create save directory specific to this dataset and model
            save_dir = os.path.join("outputs", "extension_compare", dataset_name, model_name)
            os.makedirs(save_dir, exist_ok=True)

            # 4. Save configuration file for reproducibility
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=4)

            # 5. Initialize Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=cfg["lr"], 
                weight_decay=cfg["weight_decay"]
            )
            
            # Retained CosineAnnealingLR to ensure stable validation curves
            # eta_min prevents the learning rate from hitting absolute 0, keeping it at 1e-6 at the end.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=cfg["epochs"], 
                eta_min=1e-6
            )

            best_acc = 0.0
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            # 6. Training Loop
            for epoch in range(cfg["epochs"]):
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch + 1}/{cfg['epochs']}] - {model_name} (LR: {current_lr:.6f})")

                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, test_loader, criterion, device)
                
                # Step the learning rate scheduler
                scheduler.step()

                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

                # Log metrics for this epoch
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                global_val_history[dataset_name][model_name].append(val_acc)

                # Save the model if it achieves a new best validation accuracy
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

            # 7. Save detailed history to JSON
            json_path = os.path.join(save_dir, "history.json")
            with open(json_path, "w") as f:
                json.dump(history, f, indent=4)

            # 8. Save detailed history to CSV
            csv_path = os.path.join(save_dir, "history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                for i in range(len(history["train_loss"])):
                    writer.writerow([
                        i + 1, 
                        history["train_loss"][i], 
                        history["train_acc"][i], 
                        history["val_loss"][i], 
                        history["val_acc"][i]
                    ])
            
            # Store the best accuracy for the final summary table
            best_acc_summary[dataset_name][model_name] = best_acc
            
            # Strict memory cleanup to avoid Apple Silicon (MPS) OOM errors
            del model, train_loader, test_loader, optimizer, scheduler
            gc.collect()
            if device == "mps": 
                torch.mps.empty_cache()

    # Final summary reports execution
    output_base_dir = os.path.join("outputs", "extension_compare")
    print_and_save_tables(global_val_history, best_acc_summary, cfg["epochs"], output_base_dir)


if __name__ == "__main__":
    main()