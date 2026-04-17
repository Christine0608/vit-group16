"""
plot_results.py
Author: Extension
Desc:   Reads training history CSV files and generates publication-quality 
        learning curve plots (Loss and Accuracy) comparing ViT vs CNN 
        across multiple datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_learning_curves():
    # Updated to match the folders in your 'extension_new_data_model_compare' directory
    datasets = ['bloodmnist', 'pathmnist', 'tissuemnist']
    models = ['vit_pretrained', 'cnn']
    
    # Updated base directory based on your screenshot
    base_dir = os.path.join("outputs", "extension_new_data_model_compare")
    
    # Visualization settings: specific colors and line styles
    color_map = {
        'vit_pretrained': '#1f77b4',  # Blue for ViT
        'cnn': '#d62728'              # Red for CNN
    }
    
    if not os.path.exists(base_dir):
        print(f"[Error] Directory not found: {base_dir}")
        print("Please ensure you are running this script from the project root.")
        return
    
    for dataset in datasets:
        # Initialize a figure with 1 row and 2 columns
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        fig.suptitle(f"Learning Curves: {dataset.upper()}", fontsize=16, fontweight='bold')
        
        ax_loss = axes[0]
        ax_acc = axes[1]
        
        data_loaded_for_dataset = False
        
        for model in models:
            # Construct the path to the specific history.csv
            csv_path = os.path.join(base_dir, dataset, model, "history.csv")
            
            if not os.path.exists(csv_path):
                print(f"[Warning] Missing data file: {csv_path}")
                continue
                
            data_loaded_for_dataset = True
            df = pd.read_csv(csv_path)
            
            epochs = df['epoch']
            
            # Plot Training and Validation Loss
            ax_loss.plot(epochs, df['train_loss'], label=f'{model} (Train)', 
                         color=color_map[model], linestyle='-', linewidth=2)
            ax_loss.plot(epochs, df['val_loss'], label=f'{model} (Val)', 
                         color=color_map[model], linestyle='--', linewidth=2)
            
            # Plot Training and Validation Accuracy
            ax_acc.plot(epochs, df['train_acc'], label=f'{model} (Train)', 
                        color=color_map[model], linestyle='-', linewidth=2)
            ax_acc.plot(epochs, df['val_acc'], label=f'{model} (Val)', 
                        color=color_map[model], linestyle='--', linewidth=2)
            
        if data_loaded_for_dataset:
            # Format Loss Subplot
            ax_loss.set_title("Cross-Entropy Loss", fontsize=14)
            ax_loss.set_xlabel("Epoch", fontsize=12)
            ax_loss.set_ylabel("Loss", fontsize=12)
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, linestyle=':', alpha=0.7)
            
            # Format Accuracy Subplot
            ax_acc.set_title("Accuracy", fontsize=14)
            ax_acc.set_xlabel("Epoch", fontsize=12)
            ax_acc.set_ylabel("Accuracy", fontsize=12)
            ax_acc.legend(loc='lower right')
            ax_acc.grid(True, linestyle=':', alpha=0.7)
            
            # Save the generated figure directly inside the extension_compare_triple folder
            save_path = os.path.join(base_dir, f"{dataset}_learning_curves.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Success] Saved plot for {dataset} -> {save_path}")
        else:
            print(f"[Skip] No valid CSV files found for {dataset}.")
            
        plt.close(fig)

if __name__ == "__main__":
    print("Starting plotting routine...")
    generate_learning_curves()