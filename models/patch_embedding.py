"""
models/patch_embedding.py
Author: Zhou Danding
Desc:   Patch Embedding + Class Token + Positional Encoding for ViT.
        Input : (B, 3, H, W)
        Output: (B, num_patches+1, embed_dim)   ← ready for Transformer encoder
        
        Supports two positional encoding types:
            'learnable'  - learnable 1-D pos embeddings (ViT paper default)
            'sinusoidal' - fixed sine/cosine embeddings (non-learnable)
"""

import math
import torch
import torch.nn as nn


def build_sinusoidal_embedding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """
    Fixed sinusoidal positional embedding (non-learnable).
    Returns shape: (1, num_positions, embed_dim)
    """
    pe = torch.zeros(num_positions, embed_dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, num_positions, embed_dim)


class PatchEmbedding(nn.Module):
    """
    Split image into fixed-size patches, linearly project each patch,
    prepend a learnable [CLS] token, and add 1-D positional embeddings.

    Args:
        img_size          (int): input image size (assumes square). Default: 224
        patch_size        (int): size of each patch. Default: 16
        in_channels       (int): number of input image channels. Default: 3
        embed_dim         (int): embedding dimension. Default: 768  (ViT-B)
        dropout           (float): dropout on embeddings. Default: 0.0
        pos_encoding_type (str): 'learnable' or 'sinusoidal'. Default: 'learnable'
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
        pos_encoding_type: str = 'learnable',
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        assert pos_encoding_type in ('learnable', 'sinusoidal'), \
            f"pos_encoding_type must be 'learnable' or 'sinusoidal', got '{pos_encoding_type}'"

        self.img_size         = img_size
        self.patch_size       = patch_size
        self.num_patches      = (img_size // patch_size) ** 2
        self.embed_dim        = embed_dim
        self.pos_encoding_type = pos_encoding_type

        # Linear projection via Conv2d
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding
        if pos_encoding_type == 'learnable':
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, embed_dim)
            )
        else:
            # Fixed sinusoidal — register as buffer (not a parameter)
            sinusoidal = build_sinusoidal_embedding(self.num_patches + 1, embed_dim)
            self.register_buffer('pos_embedding', sinusoidal)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        if self.pos_encoding_type == 'learnable':
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # sinusoidal is fixed, no init needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.projection(x)                            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)                 # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, num_patches+1, embed_dim)
        x = x + self.pos_embedding
        return self.dropout(x)

    def __repr__(self):
        return (f"PatchEmbedding(img={self.img_size}, patch={self.patch_size}, "
                f"num_patches={self.num_patches}, embed_dim={self.embed_dim}, "
                f"pos={self.pos_encoding_type})")


# ── Quick sanity check ────────────────────────────────────────────────────
if __name__ == '__main__':
    configs = [
        dict(img_size=224, patch_size=16, embed_dim=768, pos_encoding_type='learnable'),
        dict(img_size=224, patch_size=16, embed_dim=768, pos_encoding_type='sinusoidal'),
        dict(img_size=224, patch_size=32, embed_dim=768, pos_encoding_type='learnable'),
    ]
    for cfg in configs:
        emb = PatchEmbedding(**cfg)
        x   = torch.randn(4, 3, cfg['img_size'], cfg['img_size'])
        out = emb(x)
        print(f"{emb}")
        print(f"  input : {tuple(x.shape)}")
        print(f"  output: {tuple(out.shape)}")
        print()