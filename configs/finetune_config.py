# configs/finetune_config.py

CONFIG = {
    # ===== Data =====
    "dataset_name": "cifar10",     # choose from: "cifar10", "cifar100"
    "batch_size": 32,
    "img_size": 224,
    "num_workers": 2,
    "data_fraction": 1.0,          # use 1.0 for full data, 0.1 / 0.3 for low-data experiments

    # ===== Model =====
    "model_name": "vit_tiny_patch16_224",
    "pretrained": True,

    # ===== Training =====
    "epochs": 3,
    "lr": 1e-4,
    "weight_decay": 1e-4,

    # ===== System =====
    "device": "cuda",
    "seed": 42,

    # ===== Saving =====
    "save_dir": "./checkpoints",
}