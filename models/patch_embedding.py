"""
models/patch_embedding.py
Author: Zhou Danding
Desc:   Patch Embedding + Class Token + Positional Encoding for ViT.
        Input : (B, 3, H, W)
        Output: (B, num_patches+1, embed_dim)   ← ready for Transformer encoder
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Split image into fixed-size patches, linearly project each patch,
    prepend a learnable [CLS] token, and add 1-D positional embeddings.

    Args:
        img_size    (int): input image size (assumes square). Default: 224
        patch_size  (int): size of each patch. Default: 16
        in_channels (int): number of input image channels. Default: 3
        embed_dim   (int): embedding dimension. Default: 768  (ViT-B)
        dropout     (float): dropout on embeddings. Default: 0.0
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2   # 196 for 224/16
        self.embed_dim  = embed_dim

        # Linear projection via Conv2d (equivalent to flattening + linear)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token (prepended to patch sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable 1-D positional embeddings (num_patches + 1 for CLS)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, num_patches+1, embed_dim)
        """
        B = x.shape[0]

        # (B, embed_dim, H/P, W/P)
        x = self.projection(x)

        # (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Expand CLS token to batch and prepend
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding

        return self.dropout(x)

    def __repr__(self):
        return (f"PatchEmbedding(img={self.img_size}, patch={self.patch_size}, "
                f"num_patches={self.num_patches}, embed_dim={self.embed_dim})")


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == '__main__':
    configs = [
        dict(img_size=224, patch_size=16, embed_dim=768),   # ViT-B/16
        dict(img_size=224, patch_size=32, embed_dim=768),   # ViT-B/32
    ]
    for cfg in configs:
        emb = PatchEmbedding(**cfg)
        x   = torch.randn(4, 3, cfg['img_size'], cfg['img_size'])
        out = emb(x)
        print(f"{emb}")
        print(f"  input : {tuple(x.shape)}")
        print(f"  output: {tuple(out.shape)}")   # expect (4, num_patches+1, 768)
        print()
