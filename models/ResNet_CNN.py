"""
models/ResNet_CNN.py
Author: Yixiao Xu
Desc:   ResNet baseline for CIFAR-10 / CIFAR-100 classification.
        Designed to work with the existing dataset pipeline:
        input  -> (B, 3, 224, 224)
        output -> (B, num_classes)
"""

from typing import Type, List
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block used in ResNet-18 / ResNet-34.

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> + shortcut -> ReLU
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut branch
        # If shape changes (spatial size or channel size), use 1x1 conv projection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet backbone + classifier head.

    Args:
        block: block type (BasicBlock here)
        layers: number of blocks in each stage
                ResNet-18 -> [2, 2, 2, 2]
                ResNet-34 -> [3, 4, 6, 3]
        num_classes: number of output classes
        in_channels: input image channels, default = 3
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 10,
        in_channels: int = 3
    ):
        super().__init__()

        self.current_channels = 64

        # Stem
        # Since your dataset pipeline resizes CIFAR images to 224x224,
        # the standard ImageNet-style stem is reasonable here.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, out_channels=64,  blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, blocks=layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int
    ) -> nn.Sequential:
        """
        Build one residual stage.

        Args:
            out_channels: output channel size of this stage
            blocks: number of residual blocks in this stage
            stride: stride of the first block in this stage
        """
        layers = []

        # First block may downsample
        layers.append(block(self.current_channels, out_channels, stride))
        self.current_channels = out_channels * block.expansion

        # Remaining blocks keep shape unchanged
        for _ in range(1, blocks):
            layers.append(block(self.current_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Weight initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 3, 224, 224)

        Returns:
            logits: shape (B, num_classes)
        """
        # Stem
        x = self.conv1(x)      # (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B, 64, 56, 56)

        # Residual stages
        x = self.layer1(x)     # (B, 64, 56, 56)
        x = self.layer2(x)     # (B, 128, 28, 28)
        x = self.layer3(x)     # (B, 256, 14, 14)
        x = self.layer4(x)     # (B, 512, 7, 7)

        # Head
        x = self.avgpool(x)    # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        logits = self.fc(x)      # (B, num_classes)

        return logits


def resnet18(num_classes: int = 10) -> ResNet:
    """
    Construct ResNet-18.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 10) -> ResNet:
    """
    Construct ResNet-34.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":
    # Quick sanity check
    model = resnet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)

    print(model)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")   # expected: (4, 10)
