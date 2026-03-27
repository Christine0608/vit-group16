"""
data/dataset.py
Author: Zhou Danding
Desc:   CIFAR-10 / CIFAR-100 data pipeline for ViT reproduction.
        Resizes images to 224x224 to match ViT patch-size=16 (196 patches).
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


# ── Mean / Std (ImageNet stats, standard for ViT fine-tuning) ──────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def get_transforms(img_size: int = 224, train: bool = True):
    """Return train or test transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
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


def get_dataloaders(
    dataset_name: str = 'cifar10',
    batch_size: int = 128,
    img_size: int = 224,
    num_workers: int = 2,
    data_fraction: float = 1.0,   # 低数据量实验用：0.1 / 0.3 / 1.0
    data_root: str = './data/raw',
):
    """
    Build train / test DataLoaders for CIFAR-10 or CIFAR-100.

    Args:
        dataset_name:   'cifar10' or 'cifar100'
        batch_size:     mini-batch size
        img_size:       resize target (224 for ViT-B/16)
        num_workers:    DataLoader workers
        data_fraction:  fraction of training data to use (for low-data experiments)
        data_root:      where to download / cache the dataset

    Returns:
        trainloader, testloader, num_classes
    """
    assert dataset_name in ('cifar10', 'cifar100'), \
        f"dataset_name must be 'cifar10' or 'cifar100', got '{dataset_name}'"
    assert 0 < data_fraction <= 1.0, \
        f"data_fraction must be in (0, 1], got {data_fraction}"

    DatasetClass = (torchvision.datasets.CIFAR10
                    if dataset_name == 'cifar10'
                    else torchvision.datasets.CIFAR100)
    num_classes = 10 if dataset_name == 'cifar10' else 100

    train_transform = get_transforms(img_size, train=True)
    test_transform  = get_transforms(img_size, train=False)

    trainset = DatasetClass(root=data_root, train=True,  download=True, transform=train_transform)
    testset  = DatasetClass(root=data_root, train=False, download=True, transform=test_transform)

    # ── Low-data regime: randomly subsample training set ──────────────────
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
    for ds in ('cifar10', 'cifar100'):
        train_loader, test_loader, nc = get_dataloaders(
            dataset_name=ds, batch_size=64, img_size=224)
        images, labels = next(iter(train_loader))
        print(f"  batch shape : {images.shape}")   # expect [64, 3, 224, 224]
        print(f"  label range : {labels.min()}–{labels.max()}")
        print(f"  num_classes : {nc}\n")
