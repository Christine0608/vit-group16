import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import timm
from data.dataset import get_dataloaders
from PIL import Image


def tensor_to_image(tensor):
    """Convert tensor [1, 3, H, W] to numpy array for visualization"""
    img = tensor.squeeze(0).cpu()
    
    # Denormalize (reverse CIFAR-10 normalization)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1)
    
    img_denorm = img * std + mean
    
    # Convert to numpy
    img_np = img_denorm.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    
    return img_np


def get_attention_weights_all_layers(model, img_tensor, device):
    """
    Extract attention weights from all layers.
    
    Args:
        model: Vision Transformer model
        img_tensor: Input image [1, 3, 224, 224]
        device: Device to run on
    
    Returns:
        Dict of attention weights for each layer
        Each element: [batch, num_heads, num_tokens, num_tokens]
    """
    model.eval()
    attention_weights = {}
    
    def qkv_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is [B, N, 3*C]
            B, N, C = output.shape
            num_heads = model.blocks[layer_idx].attn.num_heads
            head_dim = C // (3 * num_heads)
            
            # Reshape and split Q, K, V
            qkv = output.reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            
            attention_weights[layer_idx] = attn.detach()
        return hook_fn
    
    # Register hooks
    hooks = []
    for i in range(len(model.blocks)):
        hook = model.blocks[i].attn.qkv.register_forward_hook(qkv_hook(i))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights


def compute_rollout_attention(attentions, start_layer=0):
    """
    Compute attention rollout from start_layer to final layer.
    This shows how attention is propagated through layers.
    
    Args:
        attentions: Dict of attention matrices from all layers
        start_layer: Which layer to start rollout from
    
    Returns:
        Rollout attention matrix [num_tokens, num_tokens]
    """
    num_layers = len(attentions)
    
    # Start with attention from the first layer
    rollout = attentions[start_layer][0]  # [num_heads, num_tokens, num_tokens]
    
    # Average over heads
    rollout = rollout.mean(dim=0)  # [num_tokens, num_tokens]
    
    # Propagate through subsequent layers
    for layer_idx in range(start_layer + 1, num_layers):
        attn = attentions[layer_idx][0]  # [num_heads, num_tokens, num_tokens]
        attn = attn.mean(dim=0)  # [num_tokens, num_tokens]
        
        # Matrix multiplication to propagate attention
        rollout = torch.matmul(attn, rollout)
    
    return rollout


def visualize_attention_all_layers(model, img_tensor, true_label, pred_label, 
                                   cifar10_classes, save_dir, sample_name, device):
    """
    Visualize attention heatmaps for all layers.
    
    Args:
        model: Vision Transformer model
        img_tensor: Input image [1, 3, 224, 224]
        true_label: True class label
        pred_label: Predicted class label
        cifar10_classes: List of class names
        save_dir: Directory to save visualizations
        sample_name: Name of the sample
        device: Device to run on
    """
    print(f"\nGenerating attention visualizations for {sample_name}...")
    
    img_np = tensor_to_image(img_tensor)
    
    # Get attention weights from all layers
    attentions = get_attention_weights_all_layers(model, img_tensor, device)
    num_layers = len(attentions)
    
    # Calculate grid size dynamically based on number of layers
    # We need num_layers + 1 subplots (for original image + all layers)
    num_cols = 4
    num_rows = (num_layers + 1 + num_cols - 1) // num_cols  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))
    axes = axes.flatten()
    
    true_class = cifar10_classes[true_label]
    pred_class = cifar10_classes[pred_label]
    is_correct = true_label == pred_label
    status = "✓ CORRECT" if is_correct else "✗ MISCLASSIFIED"
    color_status = "green" if is_correct else "red"
    
    # Plot original image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\n{status}\nTrue: {true_class} | Pred: {pred_class}", 
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=color_status, alpha=0.7, edgecolor='black'),
                     color='white')
    axes[0].axis('off')
    
    # Plot attention for each layer
    for layer_idx in range(num_layers):
        ax = axes[layer_idx + 1]
        
        # Get attention from this layer
        attn_single = attentions[layer_idx][0].mean(dim=0)  # Average over heads
        
        # Get attention of CLS token to patches
        cls_attn = attn_single[0, 1:].cpu().numpy()
        
        # Reshape to 2D grid
        num_patches = int(np.sqrt(cls_attn.shape[0]))
        attn_2d = cls_attn.reshape(num_patches, num_patches)
        
        # Normalize
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        
        # Resize to image size
        h, w = img_np.shape[:2]
        attn_resized = cv2.resize(attn_2d, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend with image
        blended = 0.6 * img_np + 0.4 * heatmap
        
        # Plot
        ax.imshow(blended)
        ax.set_title(f"Layer {layer_idx}\nMin: {attn_2d.min():.3f}, Max: {attn_2d.max():.3f}", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(num_layers + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Attention Heatmap Evolution - {sample_name}\n{true_class} vs {pred_class}", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f"{sample_name}_attention_all_layers.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved attention heatmap to {save_path}")


def visualize_attention_rollout(model, img_tensor, true_label, pred_label,
                               cifar10_classes, save_dir, sample_name, device):
    """
    Visualize attention rollout (accumulated attention through all layers).
    
    Args:
        model: Vision Transformer model
        img_tensor: Input image [1, 3, 224, 224]
        true_label: True class label
        pred_label: Predicted class label
        cifar10_classes: List of class names
        save_dir: Directory to save visualizations
        sample_name: Name of the sample
        device: Device to run on
    """
    print(f"Generating attention rollout for {sample_name}...")
    
    img_np = tensor_to_image(img_tensor)
    
    # Get attention weights from all layers
    attentions = get_attention_weights_all_layers(model, img_tensor, device)
    num_layers = len(attentions)
    
    # Calculate grid size dynamically
    num_cols = 4
    num_rows = (num_layers + 1 + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5*num_rows))
    axes = axes.flatten()
    
    true_class = cifar10_classes[true_label]
    pred_class = cifar10_classes[pred_label]
    is_correct = true_label == pred_label
    status = "✓ CORRECT" if is_correct else "✗ MISCLASSIFIED"
    color_status = "green" if is_correct else "red"
    
    # Plot original image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\n{status}", fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=color_status, alpha=0.7, edgecolor='black'),
                     color='white')
    axes[0].axis('off')
    
    # Plot rollout from each layer
    for start_layer in range(num_layers):
        ax = axes[start_layer + 1]
        
        # Compute rollout from this layer onwards
        rollout = compute_rollout_attention(attentions, start_layer=start_layer)
        
        # Get CLS token attention
        cls_attn = rollout[0, 1:].cpu().numpy()
        
        # Reshape to 2D
        num_patches = int(np.sqrt(cls_attn.shape[0]))
        attn_2d = cls_attn.reshape(num_patches, num_patches)
        
        # Normalize
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        
        # Resize
        h, w = img_np.shape[:2]
        attn_resized = cv2.resize(attn_2d, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        blended = 0.6 * img_np + 0.4 * heatmap
        
        # Plot
        ax.imshow(blended)
        ax.set_title(f"Rollout from Layer {start_layer}\nMin: {attn_2d.min():.3f}, Max: {attn_2d.max():.3f}", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_layers + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Attention Rollout Analysis - {sample_name}\n{true_class} vs {pred_class}", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f"{sample_name}_attention_rollout.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved attention rollout to {save_path}")


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


def load_and_visualize_attention(checkpoint_path):
    """
    Load checkpoint and visualize attention for cat-dog samples.
    
    Args:
        checkpoint_path: Path to checkpoint
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
    
    # Collect samples
    print("="*80)
    print("Collecting Cat and Dog Samples")
    print("="*80 + "\n")
    
    correct_cat, correct_dog, cat_dog_mismatches, dog_cat_mismatches = collect_cat_dog_samples(
        model, testloader, device
    )
    
    # CIFAR-10 classes
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create output directory
    save_dir = "attention_rollout_results"
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Attention Visualizations")
    print("="*80 + "\n")
    
    for sample_name, (img, true_label, pred_label) in samples.items():
        # Visualize attention at each layer
        visualize_attention_all_layers(model, img, true_label, pred_label,
                                      cifar10_classes, save_dir, sample_name, device)
        
        # Visualize attention rollout
        visualize_attention_rollout(model, img, true_label, pred_label,
                                   cifar10_classes, save_dir, sample_name, device)
    
    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to '{save_dir}' folder")
    print("="*80)


if __name__ == "__main__":
    # Path to checkpoint from train.py with vit_pretrained
    checkpoint_path = "../outputs/vit_pretrained/best_model.pth"
    
    load_and_visualize_attention(checkpoint_path)
