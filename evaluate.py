#evaluate.py
import torch
import torch.nn as nn
from tqdm import tqdm
import timm
import argparse

from data.dataset import get_dataloaders
from utils.seed import set_seed

def accuracy(output, target, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Matches the metrics reported in the paper.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def evaluate_performance(model, loader, device):
    """
    Comprehensive evaluation loop.
    """
    model.eval()
    
    top1_accs = []
    top5_accs = []
    
    pbar = tqdm(loader, desc="Evaluating", leave=True)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        
        # Calculate Top-1 and Top-5 Accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        top1_accs.append(acc1.item())
        top5_accs.append(acc5.item())
        
        pbar.set_postfix({
            "Top-1": f"{sum(top1_accs)/len(top1_accs):.2f}%",
            "Top-5": f"{sum(top5_accs)/len(top5_accs):.2f}%"
        })

    avg_top1 = sum(top1_accs) / len(top1_accs)
    avg_top5 = sum(top5_accs) / len(top5_accs)
    
    return avg_top1, avg_top5

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data (Note: Paper often evaluates at higher resolution [cite: 77])
    _, test_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size # Higher res like 384 can be used here
    )

    # 2. Initialize Model
    # Using timm as seen in your train.py
    model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes)
    
    # 3. Load Checkpoint
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)

    # 4. Run Evaluation
    print(f"Starting evaluation on {args.dataset} at {args.img_size}x{args.img_size} resolution...")
    top1, top5 = evaluate_performance(model, test_loader, device)
    
    print("\n" + "="*30)
    print(f"Final Results for {args.model_name}:")
    print(f"Dataset: {args.dataset}")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Performance Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="Timm model name")
    parser.add_argument("--dataset", type=str, default="cifar100", help="cifar10 or cifar100")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224, help="Resolution (e.g., 224, 384)")
    
    args = parser.parse_args()
    main(args)