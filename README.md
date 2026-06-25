# Diff-Prv

PyTorch implementation for multimodal recommendation experiments.

## Run on Google Colab

1. Select `Runtime > Change runtime type > GPU`.
2. Clone and run a quick one-epoch test:

```bash
!git clone https://github.com/ZesyC/Diff-Prv.git
!cd Diff-Prv && python Main.py --data tiktok --epoch 1 --gpu 0
```

For full training:

```bash
!cd Diff-Prv && python Main.py --data tiktok --gpu 0
```

Supported datasets in this repo:

- `tiktok`
- `baby`

For `baby`, unzip image features first:

```bash
!cd Diff-Prv/Datasets/baby && unzip image_feat.npy.zip
!cd Diff-Prv && python Main.py --data baby --gpu 0
```
