## Model Zoo

Pretrained weights will be released upon publication.

| Variant | Arch | Config | PSNR (SRWOV synth) |
|---------|------|--------|---------------------|
| RTN baseline | RTN | waveletmask_base.yml | 26.16 |
| **WaveletMask (proposed)** | WaveletMaskABNet | waveletmask_ab.yml | **26.60** |
| +LHFM only | WaveletMaskLocalNet | waveletmask_local.yml | 26.38 |
| +GFM only | WaveletMaskFlickerNet | waveletmask_flicker.yml | 26.10 |
