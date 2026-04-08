"""
data/dataset.py
Author: Zhou Danding
Desc:   CIFAR-10 / CIFAR-100 / Tiny-ImageNet data pipeline for ViT reproduction.
        Resizes images to 224x224 to match ViT patch-size=16 (196 patches).

Tiny-ImageNet setup (one-time):
    bash scripts/download_tiny_imagenet.sh
    # or manually:
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip -d ./data/raw/
    python data/dataset.py --prepare-tiny   # fixes val folder structure
"""

import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np


# ── Mean / Std (ImageNet stats, standard for ViT fine-tuning) ──────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

DATASET_NUM_CLASSES = {
    'cifar10':        10,
    'cifar100':       100,
    'tiny-imagenet':  200,
}


def get_transforms(img_size: int = 224, train: bool = True):
    """Return train or test transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),   # slightly larger before crop
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def _prepare_tiny_imagenet_val(data_root: str):
    """
    Tiny-ImageNet val/ folder ships as a flat directory with an annotation file.
    This function reorganises it into ImageFolder format: val/<class_id>/<img>.
    Safe to call multiple times (skips if already done).
    """
    val_dir        = os.path.join(data_root, 'tiny-imagenet-200', 'val')
    annotation_file = os.path.join(val_dir, 'val_annotations.txt')
    img_dir        = os.path.join(val_dir, 'images')

    if not os.path.exists(annotation_file):
        print("[dataset] Tiny-ImageNet val already prepared, skipping.")
        return

    print("[dataset] Preparing Tiny-ImageNet val folder structure...")
    with open(annotation_file) as f:
        for line in f:
            parts    = line.strip().split('\t')
            img_name = parts[0]          # e.g. val_0.JPEG
            class_id = parts[1]          # e.g. n01443537
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(img_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)

    # Clean up flat images/ dir and annotation file
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)
    os.remove(annotation_file)
    print("[dataset] Tiny-ImageNet val folder ready.")


def _get_tiny_imagenet_loaders(
    batch_size: int,
    img_size: int,
    num_workers: int,
    data_fraction: float,
    data_root: str,
):
    """Build train/val DataLoaders for Tiny-ImageNet."""
    tiny_root = os.path.join(data_root, 'tiny-imagenet-200')
    assert os.path.exists(tiny_root), (
        f"Tiny-ImageNet not found at {tiny_root}.\n"
        "Download: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
        "Then: unzip tiny-imagenet-200.zip -d ./data/raw/\n"
        "Then: python data/dataset.py --prepare-tiny"
    )

    _prepare_tiny_imagenet_val(data_root)

    train_transform = get_transforms(img_size, train=True)
    val_transform   = get_transforms(img_size, train=False)

    trainset = ImageFolder(os.path.join(tiny_root, 'train'), transform=train_transform)
    valset   = ImageFolder(os.path.join(tiny_root, 'val'),   transform=val_transform)

    if data_fraction < 1.0:
        n_total  = len(trainset)
        n_subset = int(n_total * data_fraction)
        indices  = np.random.choice(n_total, n_subset, replace=False)
        trainset = Subset(trainset, indices)
        print(f"[dataset] Using {data_fraction*100:.0f}% of Tiny-ImageNet train "
              f"({n_subset}/{n_total} samples)")

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True,  num_workers=num_workers, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader


def get_dataloaders(
    dataset_name: str = 'cifar10',
    batch_size: int = 128,
    img_size: int = 224,
    num_workers: int = 2,
    data_fraction: float = 1.0,
    data_root: str = './data/raw',
):
    """
    Build train / test DataLoaders for CIFAR-10, CIFAR-100, or Tiny-ImageNet.

    Args:
        dataset_name:   'cifar10' | 'cifar100' | 'tiny-imagenet'
        batch_size:     mini-batch size
        img_size:       resize target (224 for ViT-B/16)
        num_workers:    DataLoader workers
        data_fraction:  fraction of training data to use (for low-data experiments)
        data_root:      where to download / cache datasets

    Returns:
        trainloader, testloader, num_classes
    """
    assert dataset_name in DATASET_NUM_CLASSES, \
        f"dataset_name must be one of {list(DATASET_NUM_CLASSES)}, got '{dataset_name}'"
    assert 0 < data_fraction <= 1.0, \
        f"data_fraction must be in (0, 1], got {data_fraction}"

    num_classes = DATASET_NUM_CLASSES[dataset_name]

    # ── Tiny-ImageNet ──────────────────────────────────────────────────────
    if dataset_name == 'tiny-imagenet':
        trainloader, testloader = _get_tiny_imagenet_loaders(
            batch_size, img_size, num_workers, data_fraction, data_root)
        print(f"[dataset] Tiny-ImageNet loaded | "
              f"train={len(trainloader.dataset)} val={len(testloader.dataset)} | "
              f"img_size={img_size} batch={batch_size} classes={num_classes}")
        return trainloader, testloader, num_classes

    # ── CIFAR-10 / CIFAR-100 ───────────────────────────────────────────────
    DatasetClass = (torchvision.datasets.CIFAR10
                    if dataset_name == 'cifar10'
                    else torchvision.datasets.CIFAR100)

    train_transform = get_transforms(img_size, train=True)
    test_transform  = get_transforms(img_size, train=False)

    trainset = DatasetClass(root=data_root, train=True,  download=True, transform=train_transform)
    testset  = DatasetClass(root=data_root, train=False, download=True, transform=test_transform)

    if data_fraction < 1.0:
        n_total  = len(trainset)
        n_subset = int(n_total * data_fraction)
        indices  = np.random.choice(n_total, n_subset, replace=False)
        trainset = Subset(trainset, indices)
        print(f"[dataset] Using {data_fraction*100:.0f}% of training data "
              f"({n_subset}/{n_total} samples)")

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True,  num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[dataset] {dataset_name.upper()} loaded | "
          f"train={len(trainset)} test={len(testset)} | "
          f"img_size={img_size} batch={batch_size} classes={num_classes}")

    return trainloader, testloader, num_classes


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-tiny', action='store_true',
                        help='Reorganise Tiny-ImageNet val folder and exit')
    parser.add_argument('--data-root', default='./data/raw')
    args = parser.parse_args()

    if args.prepare_tiny:
        _prepare_tiny_imagenet_val(args.data_root)
        print("Done.")
    else:
        for ds in ('cifar10', 'cifar100'):
            train_loader, test_loader, nc = get_dataloaders(
                dataset_name=ds, batch_size=64, img_size=224)
            images, labels = next(iter(train_loader))
            print(f"  batch shape : {images.shape}")
            print(f"  label range : {labels.min()}–{labels.max()}")
            print(f"  num_classes : {nc}\n")
