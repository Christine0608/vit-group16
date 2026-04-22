"""
configs/new_data_finetune_config.py
Configuration for fine-tuning models on MedMNIST (Path, Blood).
"""

CONFIG = {
    # ===== Data =====
    "batch_size": 32,     
    "img_size": 224,
    "num_workers": 2,        
    "data_fraction": 1.0,        

    # ===== Training =====
    "epochs": 10,        
    "lr": 1e-4,             
    "weight_decay": 0.05, 

    # ===== System =====
    "seed": 42,
}