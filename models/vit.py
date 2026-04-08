"""
models/vit.py
Author: Xu Yixiao
Desc:   MSA(multi-head self-attention) + MLP Block + Encoder(with LayerNorm) for ViT.
        Input : (B, num_patches+1, embed_dim) from PatchEmbedding
        Output: (B, num_classes) via final classifier head
"""
import torch
import torch.nn as nn

from patch_embedding import PatchEmbedding

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention for ViT.
    Input : (B, N, D)
    Output: (B, N, D)
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # (B, N, 3D)
        qkv = self.qkv(x)

        # (B, N, 3, heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # (3, B, heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # (B, heads, N, head_dim)
        out = attn @ v

        # (B, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out


class MLP(nn.Module):
    """
    Feed-forward network in Transformer block.
    Linear -> GELU -> Dropout -> Linear -> Dropout
    output: (B, N, D)
    """
    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        out_dim: int = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim

        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x

class Encoder1DBlock(nn.Module):
    """
    x = LN(x) -> MSA -> Dropout -> Residual
        1. x = LN(x)
        2. x = MSA(x)
        3. x = Dropout(x)
        4. x = x + residual
    y = LN(x) -> MLP -> Residual
        1. y = LN(x)
        2. y = MLP(y)
        3. out = x + y
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_dim=embed_dim,
            mlp_dim=mlp_dim,
            out_dim=embed_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Attention block
        x = self.norm1(inputs)
        x = self.attn(x)
        x = self.dropout1(x)
        x = x + inputs

        # MLP block
        y = self.norm2(x)
        y = self.mlp(y)

        return x + y

class Encoder(nn.Module):
    """
    Transformer encoder stack.

    Note:
    Since Danding's PatchEmbedding already adds:
        - CLS token
        - positional embedding
    this Encoder does NOT add positional embedding again.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Encoder1DBlock(
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
            )
            for _ in range(num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        x = self.encoder_norm(x)
        return x


class ViT(nn.Module):
    """
    ViT model compatible with PatchEmbedding and then composed of:
        - transformer encoder
        - token/gap pooling
        - optional pre_logits
        - classifier head
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            num_classes: int = 10,
            hidden_size: int = 768,
            num_layers: int = 12,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            representation_size: int = None,
            classifier: str = "token",   # "token" or "gap"
            head_bias_init: float = 0.0,
        ):
            super().__init__()

            assert classifier in ["token", "gap"], \
                f"Unsupported classifier={classifier}. Use 'token' or 'gap'."

            self.num_classes = num_classes
            self.hidden_size = hidden_size
            self.representation_size = representation_size
            self.classifier = classifier

            self.patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=hidden_size,
                dropout=dropout_rate,
            )

            self.encoder = Encoder(
                embed_dim=hidden_size,
                num_layers=num_layers,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
            )

            if representation_size is not None:
                self.pre_logits = nn.Linear(hidden_size, representation_size)
                self.pre_logits_act = nn.Tanh()
                self.head = nn.Linear(representation_size, num_classes)
            else:
                self.pre_logits = IdentityLayer()
                self.pre_logits_act = None
                self.head = nn.Linear(hidden_size, num_classes)

            self._init_head(head_bias_init)

    def _init_head(self, head_bias_init: float):
        if isinstance(self.pre_logits, nn.Linear):
            nn.init.xavier_uniform_(self.pre_logits.weight)
            nn.init.zeros_(self.pre_logits.bias)

        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, head_bias_init)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Already includes patch embedding + cls token + pos embedding
        x = self.patch_embed(x)    # (B, N+1, D)

        x = self.encoder(x)

        if self.classifier == "token":
            # CLS token is already the first token
            x = x[:, 0]
        elif self.classifier == "gap":
            # Mean over all tokens
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        if isinstance(self.pre_logits, nn.Linear):
            x = self.pre_logits(x)
            x = self.pre_logits_act(x)
        else:
            x = self.pre_logits(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Quick sanity check
if __name__ == "__main__":
    model = ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        hidden_size=192,
        num_layers=4,
        mlp_dim=768,
        num_heads=3,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        representation_size=None,
        classifier="token",
    )

    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    print(model.__class__.__name__)
    print("input :", x.shape)
    print("output:", out.shape)   # expect: (2, 10)
