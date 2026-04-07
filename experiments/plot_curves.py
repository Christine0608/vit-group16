import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_curve(x, y1, y2, label1, label2, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, marker='o', label=label1)
    plt.plot(x, y2, marker='o', label=label2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[saved] {save_path}")


def main():
    pretrained_path = "./checkpoints/pretrained/history.csv"
    scratch_path = "./checkpoints/scratch/history.csv"
    save_dir = "./experiments"

    os.makedirs(save_dir, exist_ok=True)

    pre = pd.read_csv(pretrained_path)
    scr = pd.read_csv(scratch_path)

    epochs_pre = pre["epoch"]
    epochs_scr = scr["epoch"]

    # Accuracy curves
    plot_curve(
        epochs_pre,
        pre["train_acc"],
        scr["train_acc"],
        "Pretrained Train Acc",
        "Scratch Train Acc",
        "Train Accuracy Curve",
        "Accuracy",
        os.path.join(save_dir, "train_accuracy_curve.png")
    )

    plot_curve(
        epochs_pre,
        pre["val_acc"],
        scr["val_acc"],
        "Pretrained Val Acc",
        "Scratch Val Acc",
        "Validation Accuracy Curve",
        "Accuracy",
        os.path.join(save_dir, "val_accuracy_curve.png")
    )

    # Loss curves
    plot_curve(
        epochs_pre,
        pre["train_loss"],
        scr["train_loss"],
        "Pretrained Train Loss",
        "Scratch Train Loss",
        "Train Loss Curve",
        "Loss",
        os.path.join(save_dir, "train_loss_curve.png")
    )

    plot_curve(
        epochs_pre,
        pre["val_loss"],
        scr["val_loss"],
        "Pretrained Val Loss",
        "Scratch Val Loss",
        "Validation Loss Curve",
        "Loss",
        os.path.join(save_dir, "val_loss_curve.png")
    )


if __name__ == "__main__":
    main()