# Diff-Prv

PyTorch implementation for multimodal recommendation experiments.

The maintained model path is **CFM**. It uses the optimized flow-matching implementation.

## Supported Datasets

- `baby`
- `tiktok`
- `sports`

For `baby`, the repo can read `Datasets/baby/image_feat.npy.zip` directly. You do not need to unzip it manually.
For `sports`, make sure `Datasets/sports/` contains `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, and `text_feat.npy`.

## Run

From the project root:

### Baby

```powershell
python .\Prv\Main.py `
  --data baby `
  --epoch 50 `
  --gpu 0 `
  --lr 1e-3 `
  --batch 1024 `
  --tstBat 256 `
  --reg 1e-5 `
  --latdim 64 `
  --gnn_layer 1 `
  --topk 20 `
  --model_type CFM `
  --ssl_reg 1e-1 `
  --temp 0.5 `
  --tstEpoch 1 `
  --seed 421 `
  --keepRate 1 `
  --dims "[1000]" `
  --d_emb_size 10 `
  --steps 5 `
  --noise_scale 0.1 `
  --noise_min 0.0001 `
  --noise_max 0.02 `
  --sampling_steps 0 `
  --rebuild_k 1 `
  --e_loss 0.1 `
  --ris_lambda 0.5 `
  --ris_adj_lambda 0.2 `
  --trans 0 `
  --cl_method 0 `
  --gate_dim 32 `
  --gate_reg 0 `
  --modal_cond 1 `
  --behavior_cond 1 `
  --cfm_lambda 0.1 `
  --cross_fm_weight 0.01
```

### TikTok

```powershell
python .\Prv\Main.py `
  --data tiktok `
  --epoch 50 `
  --gpu 0 `
  --lr 1e-3 `
  --batch 1024 `
  --tstBat 256 `
  --reg 1e-5 `
  --latdim 64 `
  --gnn_layer 1 `
  --topk 20 `
  --model_type CFM `
  --ssl_reg 1e-1 `
  --temp 0.5 `
  --tstEpoch 1 `
  --seed 421 `
  --keepRate 1 `
  --dims "[1000]" `
  --d_emb_size 10 `
  --steps 5 `
  --noise_scale 0.1 `
  --noise_min 0.0001 `
  --noise_max 0.02 `
  --sampling_steps 0 `
  --rebuild_k 1 `
  --e_loss 0.1 `
  --ris_lambda 0.5 `
  --ris_adj_lambda 0.2 `
  --trans 0 `
  --cl_method 0 `
  --gate_dim 32 `
  --gate_reg 0 `
  --modal_cond 1 `
  --behavior_cond 1 `
  --cfm_lambda 0.1 `
  --cross_fm_weight 0.01
```

### Sports

```powershell
python .\Prv\Main.py `
  --data sports `
  --epoch 50 `
  --gpu 0 `
  --lr 1e-3 `
  --batch 1024 `
  --tstBat 256 `
  --reg 1e-5 `
  --latdim 64 `
  --gnn_layer 1 `
  --topk 20 `
  --model_type CFM `
  --ssl_reg 1e-1 `
  --temp 0.5 `
  --tstEpoch 1 `
  --seed 421 `
  --keepRate 1 `
  --dims "[1000]" `
  --d_emb_size 10 `
  --steps 5 `
  --noise_scale 0.1 `
  --noise_min 0.0001 `
  --noise_max 0.02 `
  --sampling_steps 0 `
  --rebuild_k 1 `
  --e_loss 0.1 `
  --ris_lambda 0.5 `
  --ris_adj_lambda 0.2 `
  --trans 0 `
  --cl_method 0 `
  --gate_dim 32 `
  --gate_reg 0 `
  --modal_cond 1 `
  --behavior_cond 1 `
  --cfm_lambda 0.1 `
  --cross_fm_weight 0.01
```

`--model_type optimized` and `--model_type flowmatching_optimized` are accepted as backward-compatible aliases, but all of them are normalized to `CFM`.

`--data sport` is accepted as an alias for `--data sports`.
