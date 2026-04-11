"""
finetune.py
Author: Zhou Danding
Desc:   Fine-tune ViT on CIFAR-10 / CIFAR-100 from a pretrain checkpoint
        (produced by pretrain.py). Style follows train.py by Zhang Tianyue.

Usage:
    # Fine-tune from our own pretrain checkpoint
    python finetune.py --pretrain-ckpt ./checkpoints/pretrain/vit_pretrained.pth

    # Fine-tune from timm ImageNet weights (baseline comparison)
    python finetune.py --pretrain-ckpt none

    # Change dataset / epochs
    python finetune.py --pretrain-ckpt ./checkpoints/pretrain/vit_pretrained.pth \
                       --dataset cifar100 --epochs 30
"""

import os
import json
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm

from data.dataset import get_dataloaders
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


# ── Default config (mirrors finetune_config.py style) ─────────────────────
DEFAULT_CONFIG = {
    "dataset_name":  "cifar10",
    "batch_size":    128,
    "img_size":      224,
    "num_workers":   2,
    "data_fraction": 1.0,
    "epochs":        20,
    "lr":            1e-4,        # lower lr for fine-tuning
    "weight_decay":  0.05,
    "seed":          42,
}


# ── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune ViT on CIFAR-10/100')
    parser.add_argument('--pretrain-ckpt', type=str, required=True,
                        help='Path to pretrain checkpoint, or "none" to use timm ImageNet weights')
    parser.add_argument('--dataset',      type=str,   default=DEFAULT_CONFIG['dataset_name'],
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs',       type=int,   default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch-size',   type=int,   default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr',           type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_CONFIG['weight_decay'])
    parser.add_argument('--img-size',     type=int,   default=DEFAULT_CONFIG['img_size'])
    parser.add_argument('--num-workers',  type=int,   default=DEFAULT_CONFIG['num_workers'])
    parser.add_argument('--data-fraction',type=float, default=DEFAULT_CONFIG['data_fraction'])
    parser.add_argument('--seed',         type=int,   default=DEFAULT_CONFIG['seed'])
    parser.add_argument('--output-dir',   type=str,   default='outputs')
    return parser.parse_args()


# ── Train one epoch ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_samples, total_correct / total_samples


# ── Evaluate ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[device] using {device}")

    # ── Data: CIFAR-10 or CIFAR-100 ────────────────────────────────────────
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        data_fraction=args.data_fraction,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    use_our_pretrain = args.pretrain_ckpt.lower() != 'none'

    if use_our_pretrain:
        # Load ViT pretrained on Tiny-ImageNet by us
        print(f"[finetune] Loading our pretrain checkpoint: {args.pretrain_ckpt}")
        ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')

        # Build model with pretrain num_classes first, then swap head
        pretrain_num_classes = ckpt.get('num_classes', 200)
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=pretrain_num_classes,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  ✓ Loaded (pretrain val_acc={ckpt.get('val_acc', '?'):.2f}%)")

        # Replace classification head for target dataset
        model.head = nn.Linear(model.head.in_features, num_classes)
        experiment_name = f"vit_our_pretrain_{args.dataset}"

    else:
        # Baseline: timm ImageNet pretrained weights (same as train.py vit_pretrained)
        print("[finetune] Using timm ImageNet pretrained weights (baseline)")
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes,
        )
        experiment_name = f"vit_imagenet_pretrain_{args.dataset}"

    model = model.to(device)

    # ── Save dir & config ──────────────────────────────────────────────────
    save_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    cfg = vars(args)
    cfg['experiment_name'] = experiment_name
    cfg['num_classes'] = num_classes
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"[config] saved to {save_dir}/config.json")

    # ── Loss & optimiser ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ──────────────────────────────────────────────────────
    best_acc = 0.0
    history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device)

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
                    "epoch":              epoch + 1,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc":           best_acc,
                    "config":             cfg,
                },
                save_dir=save_dir,
                filename="best_model.pth"
            )

    # ── Save history ───────────────────────────────────────────────────────
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(save_dir, 'history.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([i+1, history["train_loss"][i], history["train_acc"][i],
                             history["val_loss"][i], history["val_acc"][i]])

    print(f"\n[done] best val acc = {best_acc:.4f}")
    print(f"[done] results saved to {save_dir}/")


if __name__ == '__main__':
    main()
