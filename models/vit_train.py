from data.dataset import get_dataloaders
from models.vit import ViT

import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    return avg_loss, acc


def train():
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Data
    trainloader, testloader, num_classes = get_dataloaders(
        dataset_name='cifar10',
        batch_size=16,
        img_size=224,
        num_workers=0
    )
    print("Data loaded. num_classes =", num_classes)

    # 3. Model
    model = ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        hidden_size=64,
        num_layers=2,
        mlp_dim=128,
        num_heads=4,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        representation_size=None,
        classifier="token",
    ).to(device)
    print("Model initialized.")

    # 4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 5. Training settings
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Optional progress print
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: processed {batch_idx+1} batches")

        train_loss = running_loss / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, testloader, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    train()
