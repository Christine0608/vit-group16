import os
import pandas as pd
import matplotlib.pyplot as plt


def load_history(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_summary(pre_df, scratch_df, save_dir):
    pre_best_acc = pre_df["val_acc"].max()
    scratch_best_acc = scratch_df["val_acc"].max()

    pre_final_acc = pre_df["val_acc"].iloc[-1]
    scratch_final_acc = scratch_df["val_acc"].iloc[-1]

    pre_final_loss = pre_df["val_loss"].iloc[-1]
    scratch_final_loss = scratch_df["val_loss"].iloc[-1]

    summary_path = os.path.join(save_dir, "pretrain_vs_scratch_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Pretrained vs Scratch Comparison\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Pretrained best val acc : {pre_best_acc:.4f}\n")
        f.write(f"Scratch best val acc    : {scratch_best_acc:.4f}\n\n")
        f.write(f"Pretrained final val acc  : {pre_final_acc:.4f}\n")
        f.write(f"Scratch final val acc     : {scratch_final_acc:.4f}\n\n")
        f.write(f"Pretrained final val loss : {pre_final_loss:.4f}\n")
        f.write(f"Scratch final val loss    : {scratch_final_loss:.4f}\n")

    print(f"[summary] saved to {summary_path}")


def plot_val_accuracy(pre_df, scratch_df, save_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(pre_df["epoch"], pre_df["val_acc"], label="ViT Pretrained")
    plt.plot(scratch_df["epoch"], scratch_df["val_acc"], label="ViT Scratch")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Pretrained vs Scratch - Validation Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "pretrain_vs_scratch_val_accuracy.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[plot] saved to {save_path}")


def plot_val_loss(pre_df, scratch_df, save_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(pre_df["epoch"], pre_df["val_loss"], label="ViT Pretrained")
    plt.plot(scratch_df["epoch"], scratch_df["val_loss"], label="ViT Scratch")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Pretrained vs Scratch - Validation Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "pretrain_vs_scratch_val_loss.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[plot] saved to {save_path}")


def plot_train_accuracy(pre_df, scratch_df, save_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(pre_df["epoch"], pre_df["train_acc"], label="ViT Pretrained")
    plt.plot(scratch_df["epoch"], scratch_df["train_acc"], label="ViT Scratch")

    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Pretrained vs Scratch - Train Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "pretrain_vs_scratch_train_accuracy.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[plot] saved to {save_path}")


def plot_train_loss(pre_df, scratch_df, save_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(pre_df["epoch"], pre_df["train_loss"], label="ViT Pretrained")
    plt.plot(scratch_df["epoch"], scratch_df["train_loss"], label="ViT Scratch")

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Pretrained vs Scratch - Train Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "pretrain_vs_scratch_train_loss.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[plot] saved to {save_path}")


def main():
    pretrained_csv = "outputs/vit_pretrained/history.csv"
    scratch_csv = "outputs/vit_scratch/history.csv"
    save_dir = "outputs/comparison"

    ensure_dir(save_dir)

    pre_df = load_history(pretrained_csv)
    scratch_df = load_history(scratch_csv)

    plot_val_accuracy(pre_df, scratch_df, save_dir)
    plot_val_loss(pre_df, scratch_df, save_dir)
    plot_train_accuracy(pre_df, scratch_df, save_dir)
    plot_train_loss(pre_df, scratch_df, save_dir)
    save_summary(pre_df, scratch_df, save_dir)

    print("\n[done] comparison plots generated successfully.")


if __name__ == "__main__":
    main()