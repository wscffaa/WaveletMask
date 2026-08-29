#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--basicofr', type=Path, default=None, help='BasicOFR checkout for FRAMEWORK_REF verification')
    args = ap.parse_args()

    hydrate = ROOT / 'scripts' / 'hydrate_framework.py'
    if not hydrate.is_file():
        print('ERROR: missing scripts/hydrate_framework.py', file=sys.stderr)
        return 1
    if not (ROOT / 'FRAMEWORK_REF').is_file():
        print('ERROR: missing FRAMEWORK_REF', file=sys.stderr)
        return 1
    for rel in ('basicofr/archs', 'basicsr/archs'):
        if not (ROOT / rel).is_dir():
            print(f'ERROR: missing {rel}', file=sys.stderr)
            print('run: python scripts/hydrate_framework.py', file=sys.stderr)
            return 1
    cmd = [sys.executable, str(hydrate), '--check']
    if args.basicofr:
        cmd.extend(['--basicofr', str(args.basicofr)])
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout, end='')
        print(proc.stderr, end='', file=sys.stderr)
        return proc.returncode

    import basicofr  # noqa: F401
    import basicofr.archs  # noqa: F401
    import basicsr  # noqa: F401
    import basicsr.archs  # noqa: F401
    paper = ROOT / 'PAPER.yaml'
    if not paper.is_file():
        print('ERROR: missing PAPER.yaml', file=sys.stderr)
        return 1
    print('VERIFY-OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
