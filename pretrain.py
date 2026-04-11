"""
pretrain.py
Author: Zhou Danding
Desc:   Pre-train ViT-B/16 from scratch on Tiny-ImageNet (200 classes).
        Saves a checkpoint that can be loaded for fine-tuning on CIFAR-10/100.

Usage:
    python pretrain.py                          # default settings
    python pretrain.py --epochs 30 --batch-size 64
    python pretrain.py --output-dir ./checkpoints/pretrain

Checkpoint:
    ./checkpoints/pretrain/vit_pretrained.pth
    → load with finetune.py --pretrain-ckpt ./checkpoints/pretrain/vit_pretrained.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

# reuse our existing data pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import get_dataloaders


# ── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Pre-train ViT on Tiny-ImageNet')
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--img-size',    type=int,   default=224)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight-decay',type=float, default=0.05)
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--data-root',   type=str,   default='./data/raw')
    parser.add_argument('--output-dir',  type=str,   default='./checkpoints/pretrain')
    parser.add_argument('--seed',        type=int,   default=42)
    return parser.parse_args()


# ── Train one epoch ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch}/{total_epochs}] step {i+1}/{len(loader)} "
                  f"loss={loss.item():.4f}")

    return total_loss / total, 100. * correct / total


# ── Evaluate ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100. * correct / total


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[pretrain] device={device}  epochs={args.epochs}  lr={args.lr}")

    # ── Data: Tiny-ImageNet (200 classes) ──────────────────────────────────
    trainloader, valloader, num_classes = get_dataloaders(
        dataset_name='tiny-imagenet',
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    # ── Model: ViT-B/16 from scratch (no pretrained weights) ───────────────
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,           # <-- scratch, this is the point
        num_classes=num_classes,    # 200 for Tiny-ImageNet
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[pretrain] ViT-B/16 | params={total_params:.1f}M | num_classes={num_classes}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"val loss={val_loss:.4f} acc={val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, 'vit_pretrained.pth')
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'num_classes': num_classes,
                'args':       vars(args),
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved → {ckpt_path}  (val_acc={val_acc:.2f}%)")

    print(f"\n[pretrain] Done. Best val acc = {best_val_acc:.2f}%")
    print(f"[pretrain] Checkpoint → {os.path.join(args.output_dir, 'vit_pretrained.pth')}")
    print(f"[pretrain] Next step  → python finetune.py "
          f"--pretrain-ckpt {os.path.join(args.output_dir, 'vit_pretrained.pth')}")


if __name__ == '__main__':
    main()
