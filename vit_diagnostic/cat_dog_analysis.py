import sys
import os

# Add parent directory (repository root) to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import timm
import numpy as np
from data.dataset import get_dataloaders
import torchvision.transforms as transforms


def get_logit_lens_predictions_timm(model, x):
    """
    Logit Lens for timm ViT models
    
    Args:
        model: timm ViT model
        x: Input images [B, 3, H, W]
    
    Returns:
        List of logits from each encoder block
    """
    model.eval()
    logits_per_layer = []
    
    with torch.no_grad():
        # Get patch embeddings
        x = model.patch_embed(x)  # [B, N, D]
        
        # Add CLS token manually
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D]
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        
        # Add positional embeddings
        x = x + model.pos_embed
        
        # Apply positional dropout if it exists
        x = model.pos_drop(x) if hasattr(model, 'pos_drop') else x
        
        # Pass through each block
        for i, block in enumerate(model.blocks):
            x = block(x)
            
            # Extract CLS token (first token)
            cls_token = x[:, 0]
            
            # Apply layer norm
            normalized_cls = model.norm(cls_token)
            
            # Get logits from head
            logits = model.head(normalized_cls)
            logits_per_layer.append(logits)
    
    return logits_per_layer


def tensor_to_image(tensor):
    """Convert tensor [1, 3, H, W] to PIL Image"""
    # Denormalize (reverse the normalization used in CIFAR-10)
    denormalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
            std=[1/0.2023, 1/0.1994, 1/0.2010]
        ),
        transforms.ToPILImage()
    ])
    
    img = tensor.squeeze(0).cpu()
    return denormalize(img)


def collect_cat_dog_samples(model, dataloader, device):
    """
    Collect cat and dog samples from the test set.
    
    Returns:
        - correct_cat: One correctly classified cat
        - correct_dog: One correctly classified dog
        - cat_dog_mismatches: List of cats predicted as dogs
        - dog_cat_mismatches: List of dogs predicted as cats
    """
    model.eval()
    
    correct_cat = None
    correct_dog = None
    cat_dog_mismatches = []  # Cat predicted as dog
    dog_cat_mismatches = []  # Dog predicted as cat
    
    CAT_CLASS = 3
    DOG_CLASS = 5
    
    print("Collecting cat and dog samples...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            for i in range(len(preds)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                img = images[i:i+1]
                
                # Correct cat classification
                if true_label == CAT_CLASS and pred_label == CAT_CLASS and correct_cat is None:
                    correct_cat = (img, true_label, pred_label)
                    print(f"✓ Found correct cat")
                
                # Correct dog classification
                elif true_label == DOG_CLASS and pred_label == DOG_CLASS and correct_dog is None:
                    correct_dog = (img, true_label, pred_label)
                    print(f"✓ Found correct dog")
                
                # Cat predicted as dog
                elif true_label == CAT_CLASS and pred_label == DOG_CLASS and len(cat_dog_mismatches) < 5:
                    cat_dog_mismatches.append((img, true_label, pred_label))
                    print(f"✓ Found cat→dog mismatch {len(cat_dog_mismatches)}/5")
                
                # Dog predicted as cat
                elif true_label == DOG_CLASS and pred_label == CAT_CLASS and len(dog_cat_mismatches) < 5:
                    dog_cat_mismatches.append((img, true_label, pred_label))
                    print(f"✓ Found dog→cat mismatch {len(dog_cat_mismatches)}/5")
                
                # Check if we have all samples
                if (correct_cat is not None and correct_dog is not None and 
                    len(cat_dog_mismatches) >= 5 and len(dog_cat_mismatches) >= 5):
                    break
            
            if (correct_cat is not None and correct_dog is not None and 
                len(cat_dog_mismatches) >= 5 and len(dog_cat_mismatches) >= 5):
                break
    
    return correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches


def plot_cat_dog_analysis(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches, 
                         model, device, save_dir="cat_dog_analysis_results"):
    """
    Create visualizations for cat-dog analysis with logit lens.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Organize samples
    samples = {}
    
    if correct_cat is not None:
        samples['Correct_Cat'] = correct_cat
    
    if correct_dog is not None:
        samples['Correct_Dog'] = correct_dog
    
    for i, sample in enumerate(cat_dog_mismatches, 1):
        samples[f'CatAsDog_{i}'] = sample
    
    for i, sample in enumerate(dog_cat_mismatches, 1):
        samples[f'DogAsCat_{i}'] = sample
    
    # Plot each sample
    for sample_name, (img, true_label, pred_label) in samples.items():
        print(f"\nAnalyzing {sample_name}...")
        
        # Get logit lens predictions
        logits_per_layer = get_logit_lens_predictions_timm(model, img)
        
        layer_indices = list(range(len(logits_per_layer)))
        confidences = []
        top1_preds = []
        
        for logits in logits_per_layer:
            probs = F.softmax(logits, dim=-1)
            conf, p_class = torch.max(probs, dim=-1)
            confidences.append(conf.item())
            top1_preds.append(cifar10_classes[p_class.item()])
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: Original image
        pil_img = tensor_to_image(img)
        axes[0].imshow(pil_img)
        
        true_class = cifar10_classes[true_label]
        pred_class = cifar10_classes[pred_label]
        
        # Determine color based on correctness
        if true_label == pred_label:
            color_box = 'green'
            title_text = f"✓ CORRECT\nTrue: {true_class} | Predicted: {pred_class}"
        else:
            color_box = 'red'
            title_text = f"✗ MISCLASSIFIED\nTrue: {true_class} | Predicted: {pred_class}"
        
        axes[0].set_title(title_text, fontsize=13, fontweight='bold', 
                         bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.7, edgecolor='black', linewidth=2),
                         color='white')
        axes[0].axis('off')
        
        # Right subplot: Logit lens plot
        color = 'green' if true_label == pred_label else 'red'
        axes[1].plot(layer_indices, confidences, marker='o', linestyle='-', 
                    color=color, linewidth=2.5, markersize=10, label='Confidence')
        
        # Annotate each layer with predicted class
        for i, txt in enumerate(top1_preds):
            axes[1].annotate(txt, (layer_indices[i], confidences[i]), 
                            textcoords="offset points", xytext=(0, 12), 
                            ha='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        axes[1].set_title(f"Logit Lens - Layer-by-Layer Predictions\n{sample_name}", 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Transformer Layer Index", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Confidence (Max Softmax Probability)", fontsize=11, fontweight='bold')
        axes[1].grid(True, ls='--', alpha=0.5)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xticks(layer_indices)
        axes[1].legend(fontsize=11, loc='lower right')
        
        # Save figure
        save_path = os.path.join(save_dir, f"{sample_name}_analysis.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {sample_name} to {save_path}")
        
        # Save original image separately
        img_save_path = os.path.join(save_dir, f"{sample_name}_original.png")
        pil_img.save(img_save_path)


def create_summary_table(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches, 
                        model, device, save_dir="cat_dog_analysis_results"):
    """
    Create a summary visualization showing all samples in a grid.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Organize all samples
    all_samples = []
    
    if correct_cat is not None:
        all_samples.append(('Correct Cat', correct_cat, 'green'))
    
    if correct_dog is not None:
        all_samples.append(('Correct Dog', correct_dog, 'green'))
    
    for i, sample in enumerate(cat_dog_mismatches, 1):
        all_samples.append((f'Cat→Dog {i}', sample, 'red'))
    
    for i, sample in enumerate(dog_cat_mismatches, 1):
        all_samples.append((f'Dog→Cat {i}', sample, 'red'))
    
    # Create grid of images
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (label, (img, true_label, pred_label), color) in enumerate(all_samples):
        ax = axes[idx]
        pil_img = tensor_to_image(img)
        ax.imshow(pil_img)
        
        true_class = cifar10_classes[true_label]
        pred_class = cifar10_classes[pred_label]
        
        # Title with status
        if true_label == pred_label:
            status = "✓"
            title = f"{status} {label}\n{true_class}"
        else:
            status = "✗"
            title = f"{status} {label}\n{true_class}→{pred_class}"
        
        ax.set_title(title, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='black', linewidth=2),
                    color='white')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(all_samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cat vs Dog Classification Analysis\nCorrect Examples and Misclassifications', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'cat_dog_summary_grid.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved summary grid to {summary_path}")


def print_cat_dog_statistics(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches):
    """Print statistics about cat-dog classification."""
    print("\n" + "="*80)
    print("CAT vs DOG CLASSIFICATION ANALYSIS")
    print("="*80)
    
    print(f"\nCorrect Classifications:")
    print(f"  ✓ Correct Cats: {1 if correct_cat is not None else 0}")
    print(f"  ✓ Correct Dogs: {1 if correct_dog is not None else 0}")
    
    print(f"\nMisclassifications (Cat-Dog Confusion):")
    print(f"  ✗ Cats predicted as Dogs: {len(cat_dog_mismatches)}")
    print(f"  ✗ Dogs predicted as Cats: {len(dog_cat_mismatches)}")
    
    total_cat_dog_confusion = len(cat_dog_mismatches) + len(dog_cat_mismatches)
    print(f"  Total Cat-Dog Confusions: {total_cat_dog_confusion}")
    
    print("\n" + "="*80)


def load_and_analyze_cat_dog(checkpoint_path):
    """
    Load checkpoint and perform cat-dog analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    _, testloader, num_classes = get_dataloaders(
        dataset_name='cifar10',
        batch_size=16,
        img_size=224,
        num_workers=0
    )
    print(f"Loaded CIFAR-10 test set\n")
    
    # Create and load model
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
    
    # Collect samples
    print("="*80)
    print("Collecting Cat and Dog Samples")
    print("="*80 + "\n")
    
    correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches = collect_cat_dog_samples(
        model, testloader, device
    )
    
    # Print statistics
    print_cat_dog_statistics(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80 + "\n")
    
    save_dir = "cat_dog_analysis_results"
    
    plot_cat_dog_analysis(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches,
                         model, device, save_dir)
    
    create_summary_table(correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches,
                        model, device, save_dir)
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to '{save_dir}' folder")
    print("="*80)


if __name__ == "__main__":
    # Path to checkpoint from train.py with vit_pretrained
    checkpoint_path = "../outputs/vit_pretrained/best_model.pth"
    
    load_and_analyze_cat_dog(checkpoint_path)
