"""Sharded NR metrics (NIQE/BRISQUE/CLIPIQA+) for SRWOV real-world degraded inputs.

Same per-frame protocol as basicofr rtn_model.py validation:
cv2.imread (BGR) -> BGR2RGB -> calculate_niqe / calculate_brisque / calculate_clipiqa.
Each shard processes units[shard_idx::num_shards] and writes a partial JSON with
per-clip sums/counts so the merger can reproduce the exact global frame-level mean.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2

sys.path.insert(0, '/mnt/data/Antigravity_OFR/framework')

from basicofr.metrics.pyiqa_metrics import (  # noqa: E402
    calculate_brisque,
    calculate_clipiqa,
    calculate_niqe,
)

ROOT = '/mnt/data/datasets/OldFilmRestoration/val/RealFilm/MambaOFR_SRWOV'
OUT_DIR = '/mnt/data/OFR_Results/baselines/RealFilm_SRWOV/degraded_input'

METRICS = {
    'niqe': calculate_niqe,
    'brisque': calculate_brisque,
    'clipiqa': calculate_clipiqa,
}

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def list_units(root):
    units = []
    for video in sorted(os.listdir(root)):
        vdir = os.path.join(root, video)
        if not os.path.isdir(vdir):
            continue
        for unit in sorted(os.listdir(vdir)):
            fdir = os.path.join(vdir, unit, 'frames')
            if os.path.isdir(fdir):
                units.append((f'{video}/{unit}', fdir))
    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-idx', type=int, required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f'input_nr_partial_{args.shard_idx}.json')

    units = list_units(ROOT)
    my_units = units[args.shard_idx::args.num_shards]
    print(f'Shard {args.shard_idx}/{args.num_shards}: {len(my_units)} of {len(units)} units', flush=True)

    per_clip = {}
    t0 = time.time()
    for idx, (name, fdir) in enumerate(my_units, 1):
        frames = sorted(f for f in os.listdir(fdir) if os.path.splitext(f)[1].lower() in IMG_EXTS)
        sums = {m: 0.0 for m in METRICS}
        counts = {m: 0 for m in METRICS}
        for fn in frames:
            img = cv2.imread(os.path.join(fdir, fn))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for mname, fn_metric in METRICS.items():
                value = float(fn_metric(img))
                if not math.isnan(value):
                    sums[mname] += value
                    counts[mname] += 1
        means = {m: (sums[m] / counts[m] if counts[m] else float('nan')) for m in METRICS}
        per_clip[name] = {
            'num_frames': len(frames),
            'sums': sums,
            'counts': counts,
            'means': means,
        }
        elapsed = time.time() - t0
        print(f'[shard {args.shard_idx}] [{idx}/{len(my_units)}] {name}: {len(frames)} frames | '
              + ', '.join(f'{m}: {means[m]:.4f}' for m in METRICS)
              + f' | elapsed {elapsed/60:.1f} min', flush=True)

    with open(out_json, 'w') as f:
        json.dump({'shard_idx': args.shard_idx, 'num_shards': args.num_shards, 'per_clip': per_clip}, f, indent=2)
    print(f'Shard {args.shard_idx} DONE -> {out_json}', flush=True)


if __name__ == '__main__':
    main()
