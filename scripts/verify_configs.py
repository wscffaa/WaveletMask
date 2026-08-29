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
_MUST_PATCH = '__MUST_PATCH__'
_CHECKPOINT_KEY_RE = re.compile(r'^(pretrain_network_[A-Za-z0-9_]+|resume_state)$')


def _checkpoint_refs(data: object) -> list[tuple[str, object]]:
    if not isinstance(data, dict):
        return []
    path_section = data.get('path') or {}
    if not isinstance(path_section, dict):
        return []
    refs = []
    for key, value in path_section.items():
        if not isinstance(key, str) or not _CHECKPOINT_KEY_RE.match(key):
            continue
        if value in (None, '', '~'):
            continue
        refs.append((key, value))
    return refs


def _is_legacy_config(yml: Path) -> bool:
    rel = yml.relative_to(OPTIONS_DIR)
    return any(part.startswith('legacy_') for part in rel.parts[:-1])


def _is_generated_config(yml: Path) -> bool:
    """Return whether a config is a generated/provenance-only derivative."""
    try:
        rel = yml.resolve().relative_to(OPTIONS_DIR.resolve())
    except ValueError:
        return False
    return '_resolved' in rel.parts[:-1]


def _check_checkpoint_path(
        yml: Path, key: str, value: object, *,
        require_artifact: bool, split: str, problems: list[str]) -> bool:
    if not isinstance(value, str):
        problems.append(f'{yml.relative_to(REPO)}: {key} 必须是路径字符串: {value!r}')
        return False
    path = Path(value)
    resolved = path if path.is_absolute() else REPO / path
    try:
        resolved.resolve().relative_to(REPO.resolve())
    except ValueError:
        problems.append(f'{yml.relative_to(REPO)}: checkpoint ref {key} escapes repo: {value}')
        return False
    if resolved.is_file():
        return True
    if require_artifact:
        problems.append(f'{yml.relative_to(REPO)}: missing {split} checkpoint {key}: {value}')
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument(
        '--require-checkpoints', action='store_true',
        help='fail unless every checkpoint referenced by active train/test configs exists locally; legacy_* and _resolved configs are provenance-only')
    args = ap.parse_args()
    if not TRAIN_DIR.is_dir():
        print(f'ERROR: missing {TRAIN_DIR}', file=sys.stderr)
        return 1
    option_configs = sorted(OPTIONS_DIR.rglob('*.yml'))
    generated_configs = [yml for yml in option_configs if _is_generated_config(yml)]
    configs = [
        yml for yml in sorted(TRAIN_DIR.rglob('*.yml'))
        if not _is_generated_config(yml)
    ]
    test_configs = [
        yml for yml in sorted(TEST_DIR.rglob('*.yml'))
        if not _is_generated_config(yml)
    ] if TEST_DIR.is_dir() else []
    problems = []
    for yml in option_configs:
        if _is_generated_config(yml):
            continue
        if _MUST_PATCH in yml.read_text(encoding='utf-8'):
            problems.append(f'{yml.relative_to(REPO)}: unresolved {_MUST_PATCH} placeholder')
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
            try:
                registry_data = yaml.safe_load(registry.read_text(encoding='utf-8')) or {}
            except Exception as exc:
                problems.append(f'Exp/registry.yaml: YAML 解析失败: {exc}')
                registry_data = {}
            if not isinstance(registry_data, dict):
                problems.append('Exp/registry.yaml: root 必须是 mapping')
                registry_data = {}
            variants = registry_data.get('variants') or []
            if not isinstance(variants, list):
                problems.append('Exp/registry.yaml: variants 必须是 list')
                variants = []
            for index, variant in enumerate(variants):
                if not isinstance(variant, dict):
                    problems.append(f'Exp/registry.yaml: variants[{index}] 必须是 mapping')
                    continue
                cfg = variant.get('config')
                if not cfg:
                    continue
                vid = variant.get('id', '?')
                if not isinstance(cfg, str):
                    problems.append(f'registry.yaml: variant {vid} 的 config 必须是路径字符串: {cfg!r}')
                    continue
                cfg_path = Path(cfg)
                resolved_cfg = cfg_path if cfg_path.is_absolute() else REPO / cfg_path
                try:
                    resolved_cfg.resolve().relative_to(REPO.resolve())
                except ValueError:
                    problems.append(f'registry.yaml: variant {vid} 的 config 越出仓: {cfg}')
                    continue
                if _is_generated_config(resolved_cfg):
                    problems.append(
                        f'registry.yaml: variant {vid} 不得引用 _resolved 生成配置: {cfg}')
                    continue
                if not resolved_cfg.is_file():
                    problems.append(f'registry.yaml: variant {vid} 引用的配置不存在: {cfg}')
    train_checkpoint_refs = 0
    train_checkpoint_present = 0
    test_checkpoint_refs = 0
    test_checkpoint_present = 0
    legacy_checkpoint_refs = 0
    if yaml is not None:
        for split, ymls in (('train', configs), ('test', test_configs)):
            for yml in ymls:
                try:
                    data = yaml.safe_load(yml.read_text(encoding='utf-8')) or {}
                except Exception as exc:
                    problems.append(f'{yml.relative_to(REPO)}: YAML 解析失败: {exc}')
                    continue
                if not isinstance(data, dict):
                    problems.append(f'{yml.relative_to(REPO)}: YAML root 必须是 mapping')
                    continue
                path_section = data.get('path') or {}
                if not isinstance(path_section, dict):
                    problems.append(f'{yml.relative_to(REPO)}: path 必须是 mapping')
                    continue
                refs = _checkpoint_refs(data)
                if _is_legacy_config(yml):
                    legacy_checkpoint_refs += len(refs)
                    continue
                if split == 'train':
                    train_checkpoint_refs += len(refs)
                else:
                    test_checkpoint_refs += len(refs)
                for key, value in refs:
                    present = _check_checkpoint_path(
                        yml, key, value,
                        require_artifact=args.require_checkpoints,
                        split=split,
                        problems=problems,
                    )
                    if present and split == 'train':
                        train_checkpoint_present += 1
                    elif present:
                        test_checkpoint_present += 1
    if problems:
        print('CONFIG-CHECK-FAIL')
        print('\n'.join(problems))
        return 1
    exclusion_status = (
        f'{legacy_checkpoint_refs} legacy checkpoint refs excluded; '
        f'{len(generated_configs)} _resolved generated configs excluded')
    artifact_status = (
        f'{train_checkpoint_present}/{train_checkpoint_refs} train checkpoint refs present; '
        f'{test_checkpoint_present}/{test_checkpoint_refs} test checkpoint refs present; '
        f'{exclusion_status}'
        if args.require_checkpoints else
        f'checkpoint artifacts not checked; {exclusion_status}; '
        'use --require-checkpoints before repository-wide test/reproduction')
    print(
        f'CONFIG-CHECK-OK ({len(configs)} train configs; '
        f'{len(test_configs)} test configs; {artifact_status})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
