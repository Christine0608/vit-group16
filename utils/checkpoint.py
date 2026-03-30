# utils/checkpoint.py

import os
import torch


def save_checkpoint(state, save_dir, filename="best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    print(f"[checkpoint] saved to {save_path}")