import os
import json

import torch
import timm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data.dataset import get_dataloaders
from configs.finetune_config import CONFIG
from models.ResNet_CNN import resnet18
from models.vit import ViT


def build_model(model_name, num_classes, img_size):
    if model_name == "cnn":
        model = resnet18(num_classes=num_classes)
        experiment_name = "cnn_resnet18"

    elif model_name == "vit_scratch":
        model = ViT(
            img_size=img_size,
            patch_size=16,
            in_channels=3,
            num_classes=num_classes,
            hidden_size=192,
            num_layers=4,
            mlp_dim=768,
            num_heads=3,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            representation_size=None,
            classifier="token",
        )
        experiment_name = "vit_scratch"

    elif model_name == "vit_pretrained":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )
        experiment_name = "vit_pretrained"

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, experiment_name


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_probs


def main():
    cfg = CONFIG

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[device] using {device}")

    model_name = cfg["model_name"]

    # load data
    _, test_loader, num_classes = get_dataloaders(
        dataset_name=cfg["dataset_name"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        num_workers=cfg["num_workers"],
        data_fraction=cfg["data_fraction"],
    )

    # build model
    model, experiment_name = build_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=cfg["img_size"],
    )

    save_dir = os.path.join("outputs", experiment_name)
    checkpoint_path = os.path.join(save_dir, "best_model.pth")

    print(f"[load] checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    labels, preds, probs = evaluate_model(model, test_loader, device)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

    try:
        metrics["auc_macro_ovr"] = roc_auc_score(
            labels,
            probs,
            multi_class="ovr",
            average="macro",
        )
    except Exception as e:
        print(f"[warning] failed to compute AUC: {e}")
        metrics["auc_macro_ovr"] = None

    print("\n===== Evaluation Result =====")
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n[metrics] saved to {metrics_path}")


if __name__ == "__main__":
    main()