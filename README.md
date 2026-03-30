# DSA5106 Group 16 — ViT Reproduction

Reproduction of **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** (Dosovitskiy et al., ICLR 2021).

## Project structure

```
vit-group16/
├── data/
│   ├── dataset.py          # Zhou Danding — CIFAR data pipeline
│   └── __init__.py
├── models/
│   ├── patch_embedding.py  # Zhou Danding — Patch embedding + CLS + PosEnc
│   ├── vit.py              # Xu Yixiao   — ViT encoder (MSA + MLP blocks)
│   ├── cnn.py              # Xu Yixiao   — ResNet-18 CNN baseline
│   └── __init__.py
├── configs/
│   ├── finetune_config.py      # Zhang Tianyue — training / model config
│   └── __init__.py
├── utils/
│   ├── seed.py                # Zhang Tianyue — random seed setup
│   ├── checkpoint.py          # Zhang Tianyue — save / load checkpoints
│   └── __init__.py
├── checkpoints/               # Saved model checkpoints and training history
│   ├── best_model.pth
│   ├── history.json
│   └── history.csv
├── train.py                # Zhang Tianyue — fine-tune pipeline
├── evaluate.py             # Duan Tianyu   — evaluation metrics
├── experiments/            # Lin / Wang    — extension experiment configs
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/Christine0608/vit-group16.git
cd vit-group16
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick test — data pipeline

```bash
python data/dataset.py
```

Expected output:
```
CIFAR10 loaded | train=50000 test=10000 | img_size=224 batch=64 classes=10
  batch shape : torch.Size([64, 3, 224, 224])
```

## Quick test — patch embedding

```bash
python models/patch_embedding.py
```

## Quick test - Fine tuning

Run fine-tuning with:

```bash
python train.py

Expected output:
```
PatchEmbedding(img=224, patch=16, num_patches=196, embed_dim=768)
  input : (4, 3, 224, 224)
  output: (4, 197, 768)
```

## Branch convention

| Branch | Owner | Task |
|---|---|---|
| `dd/data-pipeline` | Zhou Danding | data pipeline + patch embedding |
| `ty/train` | Zhang Tianyue | training script |
| `dt/eval` | Duan Tianyu | evaluation script |
| `yx/model` | Xu Yixiao | CNN baseline + ViT encoder |
| `ext/patch-size` | Lin Yaohaishan | patch size experiments |
| `ext/low-data` | Wang Ruiyi | low-data experiments |
| `ext/attention` | Yan An | attention visualization |

## Weekly sync

Every Saturday 10:30–11:00 on Teams.
