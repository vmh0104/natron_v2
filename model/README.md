# Model Directory

This directory contains trained model checkpoints and artifacts.

## Files

### After Training:
- `natron_v2.pt` - Final supervised model (used for inference)
- `scaler.pkl` - Feature scaler for normalization
- `pretrain/` - Pretraining checkpoints
  - `best_pretrain.pt` - Best pretrained encoder
  - `pretrain_epoch_*.pt` - Periodic checkpoints
- `supervised/` - Supervised training checkpoints
  - `best_supervised.pt` - Best supervised model
  - `supervised_epoch_*.pt` - Periodic checkpoints

## Model Architecture

```
NatronTransformer
├── Input Projection (num_features → d_model=256)
├── Positional Encoding
├── Transformer Encoder (6 layers, 8 heads)
└── Task Heads:
    ├── Buy Head (→ 1)
    ├── Sell Head (→ 1)
    ├── Direction Head (→ 3)
    └── Regime Head (→ 6)
```

## Model Size

Typical model size: ~15-20 MB (compressed)

Parameters: ~5-10M depending on configuration
