import sys
import os

# Add parent directory (repository root) to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import json
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import get_dataloaders


def compute_confusion_matrix(model, dataloader, device, num_classes=10):
    """
    Compute confusion matrix for the model (without sklearn).
    
    Args:
        model: ViT model
        dataloader: Test dataloader
        device: Device to run on
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (numpy array)
        All predictions (numpy array)
        All true labels (numpy array)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Computing predictions on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix manually
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(all_labels, all_preds):
        conf_matrix[true_label, pred_label] += 1
    
    return conf_matrix, all_preds, all_labels


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    Plot and save confusion matrix heatmap with zeroed diagonal to highlight errors.
    
    Args:
        conf_matrix: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 10))
    
    # Create a copy so we don't modify the original data used for statistics
    plot_data = conf_matrix.copy()
    
    # Set the diagonal to zero to highlight misclassifications
    np.fill_diagonal(plot_data, 0)
    
    # Create heatmap
    # Note: We use 'plot_data' for the colors/intensities, but keep 'conf_matrix' 
    # for the annotations if you still want to see the correct counts (which will be 0).
    sns.heatmap(plot_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Misclassification Count'})
    
    plt.title('Confusion Matrix - Misclassification Errors (Diagonal Zeroed)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error-focused confusion matrix to {save_path}")


def analyze_misclassification(conf_matrix, class_names, save_path):
    """
    Analyze and visualize misclassification statistics.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    num_classes = len(class_names)
    
    # Calculate misclassification rates for each class
    misclassification_rates = []
    total_misclassified = []
    
    for i in range(num_classes):
        total_samples = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        misclassified = total_samples - correct
        
        if total_samples > 0:
            rate = misclassified / total_samples * 100
        else:
            rate = 0
        
        misclassification_rates.append(rate)
        total_misclassified.append(misclassified)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Misclassification rate by class
    colors = ['red' if rate > 20 else 'orange' if rate > 10 else 'green' 
              for rate in misclassification_rates]
    axes[0].bar(range(num_classes), misclassification_rates, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Misclassification Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Misclassification Rate by Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (rate, cls_name) in enumerate(zip(misclassification_rates, class_names)):
        axes[0].text(i, rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')
    
    # Plot 2: Total number of misclassified samples
    axes[1].bar(range(num_classes), total_misclassified, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Misclassified Samples', fontsize=12, fontweight='bold')
    axes[1].set_title('Total Misclassified Samples by Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (count, cls_name) in enumerate(zip(total_misclassified, class_names)):
        axes[1].text(i, count + 0.5, str(int(count)), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved misclassification analysis to {save_path}")
    
    return misclassification_rates, total_misclassified


def calculate_metrics(conf_matrix):
    """
    Calculate precision, recall, and F1-score for each class (without sklearn).
    
    Args:
        conf_matrix: Confusion matrix
    
    Returns:
        Dictionary with precision, recall, f1 for each class
    """
    num_classes = conf_matrix.shape[0]
    metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    return metrics


def print_confusion_statistics(conf_matrix, class_names):
    """
    Print detailed confusion matrix statistics.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("CONFUSION MATRIX STATISTICS")
    print("="*80)
    
    # Overall accuracy
    accuracy = np.trace(conf_matrix) / conf_matrix.sum()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate metrics
    metrics = calculate_metrics(conf_matrix)
    
    # Per-class statistics
    print("\n" + "-"*80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['precision'][i]
        recall = metrics['recall'][i]
        f1 = metrics['f1'][i]
        support = int(conf_matrix[i].sum())
        
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    print("-"*80)
    
    # Highly misclassified classes
    print("\n" + "="*80)
    print("HIGHLY MISCLASSIFIED CLASSES")
    print("="*80)
    
    misclassification_rates = []
    for i in range(len(class_names)):
        total = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        if total > 0:
            rate = (total - correct) / total * 100
        else:
            rate = 0
        misclassification_rates.append((class_names[i], rate, total - correct, total))
    
    # Sort by misclassification rate
    misclassification_rates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Class':<15} {'Misclass. Rate':<20} {'Misclassified':<20} {'Total Samples':<15}")
    print("-"*80)
    
    for class_name, rate, misclass_count, total in misclassification_rates:
        print(f"{class_name:<15} {rate:<20.2f}% {int(misclass_count):<20} {int(total):<15}")
    
    # Most common confusions
    print("\n" + "="*80)
    print("MOST COMMON CONFUSIONS (Top 10)")
    print("="*80)
    
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i, j] > 0:
                confusions.append((class_names[i], class_names[j], conf_matrix[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'True Class':<15} {'Predicted As':<15} {'Count':<10} {'% of True Class':<15}")
    print("-"*80)
    
    for true_cls, pred_cls, count in confusions[:10]:
        true_idx = class_names.index(true_cls)
        percent = (count / conf_matrix[true_idx].sum()) * 100
        print(f"{true_cls:<15} {pred_cls:<15} {int(count):<10} {percent:<15.2f}%")
    
    print("\n" + "="*80)


def save_confusion_report(conf_matrix, class_names, save_dir):
    """
    Save confusion matrix statistics to text file (without sklearn).
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_dir: Directory to save report
    """
    report_path = os.path.join(save_dir, 'confusion_matrix_report.txt')
    
    # Calculate metrics
    metrics = calculate_metrics(conf_matrix)
    
    with open(report_path, 'w') as f:
        # Overall accuracy
        accuracy = np.trace(conf_matrix) / conf_matrix.sum()
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        # Per-class statistics
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            precision = metrics['precision'][i]
            recall = metrics['recall'][i]
            f1 = metrics['f1'][i]
            support = int(conf_matrix[i].sum())
            
            f.write(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}\n")
        
        # Misclassification rates
        f.write("\n" + "="*80 + "\n")
        f.write("HIGHLY MISCLASSIFIED CLASSES\n")
        f.write("="*80 + "\n\n")
        
        misclassification_rates = []
        for i in range(len(class_names)):
            total = conf_matrix[i].sum()
            correct = conf_matrix[i, i]
            if total > 0:
                rate = (total - correct) / total * 100
            else:
                rate = 0
            misclassification_rates.append((class_names[i], rate, total - correct, total))
        
        misclassification_rates.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"{'Class':<15} {'Misclass. Rate':<20} {'Misclassified':<20} {'Total Samples':<15}\n")
        f.write("-"*80 + "\n")
        
        for class_name, rate, misclass_count, total in misclassification_rates:
            f.write(f"{class_name:<15} {rate:<20.2f}% {int(misclass_count):<20} {int(total):<15}\n")
        
        # Confusion matrix as text
        f.write("\n" + "="*80 + "\n")
        f.write("CONFUSION MATRIX (rows=true, cols=predicted)\n")
        f.write("="*80 + "\n\n")
        
        f.write("     " + "".join([f"{name:<8}" for name in class_names]) + "\n")
        for i, true_class in enumerate(class_names):
            f.write(f"{true_class:<4} " + "".join([f"{int(conf_matrix[i, j]):<8}" for j in range(len(class_names))]) + "\n")
    
    print(f"✓ Saved confusion matrix report to {report_path}")


def load_and_analyze_confusion_matrix(checkpoint_path):
    """
    Load checkpoint and compute confusion matrix (without sklearn).
    
    Args:
        checkpoint_path: Path to checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    _, testloader, num_classes = get_dataloaders(
        dataset_name='cifar10',
        batch_size=32,
        img_size=224,
        num_workers=0
    )
    print(f"Loaded CIFAR-10 test set\n")
    
    # Create model
    print(f"Loading checkpoint from {checkpoint_path}...\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Checkpoint loaded successfully!")
    
    if 'best_acc' in checkpoint:
        print(f"✓ Checkpoint accuracy: {checkpoint['best_acc']:.4f}\n")
    
    # CIFAR-10 classes
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create output directory
    save_dir = "confusion_matrix_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    print("="*80)
    print("Computing Confusion Matrix")
    print("="*80 + "\n")
    
    conf_matrix, all_preds, all_labels = compute_confusion_matrix(model, testloader, device, num_classes)
    
    # Print statistics
    print_confusion_statistics(conf_matrix, cifar10_classes)
    
    # Save report
    save_confusion_report(conf_matrix, cifar10_classes, save_dir)
    
    # Plot confusion matrix
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80 + "\n")
    
    conf_matrix_path = os.path.join(save_dir, 'confusion_matrix_heatmap.png')
    plot_confusion_matrix(conf_matrix, cifar10_classes, conf_matrix_path)
    
    # Plot misclassification analysis
    misclass_path = os.path.join(save_dir, 'misclassification_analysis.png')
    analyze_misclassification(conf_matrix, cifar10_classes, misclass_path)
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to '{save_dir}' folder")
    print("="*80)


if __name__ == "__main__":
    # Path to checkpoint from train.py with vit_pretrained
    checkpoint_path = "../outputs/vit_pretrained/best_model.pth"
    
    load_and_analyze_confusion_matrix(checkpoint_path)
