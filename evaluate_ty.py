import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data.dataset import get_dataloaders
from configs.finetune_config import CONFIG
from models.ResNet_CNN import resnet18
from models.vit import ViT
import timm

def accuracy_topk(output, target, topk=(1, 5)):
    """Computes Top-1 and Top-5 accuracy as used in the original paper."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def interpolate_pos_embed(pos_embed, new_size, patch_size=16):
    """Bicubic 2D interpolation for positional embeddings (Paper Section 3.2)."""
    cls_token_embed = pos_embed[:, :1]
    patch_embeds = pos_embed[:, 1:]
    B, N, E = patch_embeds.shape
    old_grid_size = int(np.sqrt(N))
    new_grid_size = new_size // patch_size
    if old_grid_size == new_grid_size:
        return pos_embed
    patch_embeds = patch_embeds.reshape(B, old_grid_size, old_grid_size, E).permute(0, 3, 1, 2)
    patch_embeds = torch.nn.functional.interpolate(
        patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False
    )
    patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(B, -1, E)
    return torch.cat((cls_token_embed, patch_embeds), dim=1)

def build_model(model_name, num_classes, img_size):
    """Path and model logic copied directly from provided evaluate.py."""
    if model_name == "cnn":
        model = resnet18(num_classes=num_classes)
        experiment_name = "cnn_resnet18"
    elif model_name == "vit_scratch":
        model = ViT(
            img_size=img_size, patch_size=16, in_channels=3,
            num_classes=num_classes, hidden_size=192, num_layers=4,
            mlp_dim=768, num_heads=3, dropout_rate=0.1,
            attention_dropout_rate=0.1, representation_size=None,
            classifier="token",
        )
        experiment_name = "vit_scratch"
    elif model_name == "vit_pretrained":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        experiment_name = "vit_pretrained"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model, experiment_name

def evaluate_run(checkpoint_path, model, test_loader, device, img_size):
    """Evaluates a single model run to collect metrics."""
    print(f"[load] checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if hasattr(model, 'pos_embed') and img_size != 224:
        model.pos_embed = nn.Parameter(interpolate_pos_embed(model.pos_embed, img_size))

    all_labels, all_preds = [], []
    top1_accs, top5_accs = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            t1, t5 = accuracy_topk(outputs, labels, topk=(1, 5))
            top1_accs.append(t1)
            top5_accs.append(t5)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return {
        "top1": np.mean(top1_accs),
        "top5": np.mean(top5_accs),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIG
    model_name = cfg["model_name"]
    img_size = cfg["img_size"]

    _, test_loader, num_classes = get_dataloaders(
        dataset_name='cifar100', batch_size=cfg["batch_size"], img_size=img_size
    )

    temp_model, experiment_name = build_model(model_name, num_classes, img_size)
    base_dir = os.path.join("outputs", experiment_name)
    runs = ["run1", "run2", "run3"]
    run_metrics = []

    for run in runs:
        checkpoint_path = os.path.join(base_dir, run, "best_model.pth")
        if os.path.exists(checkpoint_path):
            model, _ = build_model(model_name, num_classes, img_size)
            metrics = evaluate_run(checkpoint_path, model, test_loader, device, img_size)
            run_metrics.append(metrics)
        else:
            print(f"[warning] missing checkpoint: {checkpoint_path}")

    if not run_metrics:
        print("[error] no checkpoints found to evaluate.")
        return

    # Aggregate results for reporting Mean and SD
    final_results = {}
    metric_keys = ["top1", "top5", "precision", "recall", "f1"]
    
    for key in metric_keys:
        values = [m[key] for m in run_metrics]
        final_results[f"{key}_mean"] = np.mean(values)
        final_results[f"{key}_std"] = np.std(values)

    print("\n===== Evaluation Result =====")
    print(f"runs_evaluated: {len(run_metrics)}")
    for k, v in final_results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()