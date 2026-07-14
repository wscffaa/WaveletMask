# WaveletMask

Self-contained Old Film Restoration paper repo generated from BasicOFR.

- Stage SSOT: `PAPER.yaml`
- Method code: `waveletmask/`
- Stages: `Idea/` · `Exp/` · `Paper/`

## Train / test

```bash
python train.py -opt Exp/options/train/<config>.yml
python test.py -opt Exp/options/test/<config>.yml
```

## Framework version

Copied from BasicOFR `1.0.0`. Upgrade with:

```bash
python ../BasicOFR/scripts/sync_framework.py --to .
```
