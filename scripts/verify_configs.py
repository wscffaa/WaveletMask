#!/usr/bin/env python3
"""Validate paper-repo configs and, optionally, local checkpoint artifacts."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPTIONS_DIR = REPO / 'Exp' / 'options'
TRAIN_DIR = OPTIONS_DIR / 'train'
TEST_DIR = OPTIONS_DIR / 'test'
_WARMSTART_RE = re.compile(r'pretrain_network_g:\s*(\S+)')
_MUST_PATCH = '__MUST_PATCH__'
_CHECKPOINT_KEYS = ('pretrain_network_g', 'pretrain_network_d', 'resume_state')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument(
        '--require-checkpoints', action='store_true',
        help='fail unless every checkpoint referenced by a test config exists locally')
    args = ap.parse_args()
    if not TRAIN_DIR.is_dir():
        print(f'ERROR: missing {TRAIN_DIR}', file=sys.stderr)
        return 1
    configs = sorted(TRAIN_DIR.rglob('*.yml'))
    test_configs = sorted(TEST_DIR.rglob('*.yml')) if TEST_DIR.is_dir() else []
    problems = []
    for yml in sorted(OPTIONS_DIR.rglob('*.yml')):
        if _MUST_PATCH in yml.read_text(encoding='utf-8'):
            problems.append(f'{yml.relative_to(REPO)}: unresolved {_MUST_PATCH} placeholder')
    for yml in configs:
        text = yml.read_text(encoding='utf-8')
        m = _WARMSTART_RE.search(text)
        if not m:
            continue
        value = m.group(1)
        if value in {'~', 'null', "''", '""'}:
            continue
        path = (REPO / value).resolve() if not Path(value).is_absolute() else Path(value)
        try:
            path.relative_to(REPO.resolve())
        except ValueError:
            problems.append(f'{yml.relative_to(REPO)}: warm-start escapes repo: {value}')
            continue
        # Exp/experiments 是历史 run 的产物、pretrained/ 的 canonical 权重随后供给,
        # 两者允许尚不存在;其余路径必须已存在。
        if not path.exists() and not value.startswith(('Exp/experiments', 'pretrained/')):
            problems.append(f'{yml.relative_to(REPO)}: missing warm-start {value}')
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is None:
        # registry/test 一致性门禁缺依赖时 fail-closed,不得跳过
        problems.append('配置一致性检查需要 pyyaml,当前环境不可用')
    else:
        # registry.yaml 与配置文件一致性:variant 引用的 config 必须存在
        registry = REPO / 'Exp' / 'registry.yaml'
        if registry.is_file():
            data = yaml.safe_load(registry.read_text(encoding='utf-8')) or {}
            for variant in (data.get('variants') or []):
                cfg = (variant or {}).get('config')
                if cfg and not (REPO / cfg).is_file():
                    vid = variant.get('id', '?')
                    problems.append(f'registry.yaml: variant {vid} 引用的配置不存在: {cfg}')
    checkpoint_refs = 0
    checkpoint_present = 0
    if yaml is not None:
        for yml in test_configs:
            try:
                data = yaml.safe_load(yml.read_text(encoding='utf-8')) or {}
            except Exception as exc:
                problems.append(f'{yml.relative_to(REPO)}: YAML 解析失败: {exc}')
                continue
            path_section = data.get('path') or {}
            if not isinstance(path_section, dict):
                problems.append(f'{yml.relative_to(REPO)}: path 必须是 mapping')
                continue
            for key in _CHECKPOINT_KEYS:
                value = path_section.get(key)
                if value in (None, ''):
                    continue
                checkpoint_refs += 1
                if not isinstance(value, str):
                    problems.append(
                        f'{yml.relative_to(REPO)}: {key} 必须是路径字符串: {value!r}')
                    continue
                path = Path(value)
                resolved = path if path.is_absolute() else REPO / path
                if resolved.is_file():
                    checkpoint_present += 1
                elif args.require_checkpoints:
                    problems.append(
                        f'{yml.relative_to(REPO)}: missing test checkpoint {key}: {value}')
    if problems:
        print('CONFIG-CHECK-FAIL')
        print('\n'.join(problems))
        return 1
    artifact_status = (
        f'{checkpoint_present}/{checkpoint_refs} test checkpoint refs present'
        if args.require_checkpoints else
        'test checkpoint artifacts not checked; use --require-checkpoints before test')
    print(
        f'CONFIG-CHECK-OK ({len(configs)} train configs; '
        f'{len(test_configs)} test configs; {artifact_status})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
