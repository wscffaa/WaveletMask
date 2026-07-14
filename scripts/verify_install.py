#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import basicofr  # noqa: F401
    import basicsr  # noqa: F401
    paper = ROOT / 'PAPER.yaml'
    if not paper.is_file():
        print('ERROR: missing PAPER.yaml', file=sys.stderr)
        return 1
    print('VERIFY-OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
