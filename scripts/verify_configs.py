#!/usr/bin/env python3
"""Validate train configs reference paths inside this paper repo."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / 'Exp' / 'options' / 'train'
_WARMSTART_RE = re.compile(r'pretrain_network_g:\s*(\S+)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if not TRAIN_DIR.is_dir():
        print(f'ERROR: missing {TRAIN_DIR}', file=sys.stderr)
        return 1
    configs = sorted(TRAIN_DIR.rglob('*.yml'))
    problems = []
    for yml in configs:
        text = yml.read_text(encoding='utf-8')
        m = _WARMSTART_RE.search(text)
        if not m:
            continue
        value = m.group(1)
        if value in {'~', 'null', "''", '""'}:
            continue
        path = (REPO / value).resolve() if not Path(value).is_absolute() else Path(value)
        if not str(path).startswith(str(REPO.resolve())):
            problems.append(f'{yml.relative_to(REPO)}: warm-start escapes repo: {value}')
            continue
        if not path.exists() and not value.startswith('Exp/experiments'):
            problems.append(f'{yml.relative_to(REPO)}: missing warm-start {value}')
    if problems:
        print('CONFIG-CHECK-FAIL')
        print('\n'.join(problems))
        return 1
    print(f'CONFIG-CHECK-OK ({len(configs)} configs)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
