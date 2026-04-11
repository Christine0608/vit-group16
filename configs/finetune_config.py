CONFIG = {
    # ===== Data =====
    "dataset_name": "cifar10",
    "batch_size": 32,
    "img_size": 224,
    "num_workers": 2,
    "data_fraction": 1.0,

    # ===== Experiment =====
    # options: "cnn", "vit_pretrained", "vit_scratch"
    "model_name": "vit_scratch",

    # ===== Training =====
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,

    # ===== System =====
    "seed": 42,
}