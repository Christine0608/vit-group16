import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Pipeline Imports
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
        return [correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size).item() for k in topk]

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
    """Initializes architecture based on the project's build logic."""
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = CONFIG["model_name"]
    img_size = CONFIG["img_size"]

    # 1. Determine Path based on build_model logic
    _, experiment_name = build_model(model_name, 10, img_size)
    save_dir = os.path.join("outputs", experiment_name)
    checkpoint_path = os.path.join(save_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[error] No checkpoint found at {checkpoint_path}")
        return

    # 2. Extract Metadata from Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Extract the training config
    train_cfg = checkpoint.get("config", CONFIG)
    dataset_name = train_cfg.get("dataset_name", CONFIG.get("dataset_name"))
    
    # FIX: Dynamically detect num_classes by checking layer names
    if "head.weight" in state_dict:
        num_classes = state_dict["head.weight"].shape[0]
    elif "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
    else:
        # Fallback to config if layer names don't match standard patterns
        num_classes = train_cfg.get("num_classes", 100)

    print(f"[info] Loading context: {dataset_name} | {num_classes} classes")

    # 3. Setup Data & Model using detected metadata
    _, test_loader, _ = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=CONFIG["batch_size"], 
        img_size=img_size
    )

    model, _ = build_model(model_name, num_classes, img_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Handle resolution changes via interpolation if necessary
    if hasattr(model, 'pos_embed') and img_size != 224:
        model.pos_embed = nn.Parameter(interpolate_pos_embed(model.pos_embed, img_size))

    # 4. Evaluation Loop
    all_labels, all_preds = [], []
    top1_accs, top5_accs = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            t1, t5 = accuracy_topk(outputs, labels)
            top1_accs.append(t1)
            top5_accs.append(t5)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    # 5. Aggregate Final Metrics
    results = {
        "top1_accuracy": float(np.mean(top1_accs)),
        "top5_accuracy": float(np.mean(top5_accs)),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "dataset_evaluated": dataset_name,
        "num_classes": int(num_classes)
    }

    # 6. Save to evaluation.json and Print
    print("\n===== Evaluation Result =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    json_path = os.path.join(save_dir, "evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[metrics] saved to {json_path}")

if __name__ == "__main__":
    main()