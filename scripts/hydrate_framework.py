#!/usr/bin/env python3
"""Hydrate ignored BasicOFR framework arch trees in a paper repo."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCH_DIRS = ('basicofr/archs', 'basicsr/archs')
IGNORED_DIRS = {'__pycache__', '.DS_Store'}


def _run(
        cmd: list[str], *, cwd: Path,
        input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _text(cmd: list[str], *, cwd: Path) -> str:
    proc = _run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode('utf-8', 'replace').strip() or 'command failed')
    return proc.stdout.decode('utf-8').strip()


def _default_basicofr() -> Path:
    candidates = (REPO.parent.parent / 'BasicOFR', REPO.parent / 'BasicOFR')
    for candidate in candidates:
        if (candidate / '.git').exists() and (candidate / 'VERSION').is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _read_ref() -> str:
    ref_file = REPO / 'FRAMEWORK_REF'
    if not ref_file.is_file():
        raise RuntimeError('missing FRAMEWORK_REF')
    ref = ref_file.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', ref):
        raise RuntimeError(f'FRAMEWORK_REF must be a full commit SHA: {ref!r}')
    return ref


def _paper_version() -> str:
    paper = REPO / 'PAPER.yaml'
    if not paper.is_file():
        raise RuntimeError('missing PAPER.yaml')
    m = re.search(r'(?m)^basicofr_version:\s*[\'"]?([^\'"#\s]+)', paper.read_text(encoding='utf-8'))
    if not m:
        raise RuntimeError('PAPER.yaml missing basicofr_version')
    return m.group(1)


def _check_ref_and_version(basicofr: Path, ref: str) -> None:
    if not (basicofr / '.git').exists():
        raise RuntimeError(f'BasicOFR git repo not found: {basicofr}')
    if _text(['git', 'cat-file', '-t', ref], cwd=basicofr) != 'commit':
        raise RuntimeError(f'FRAMEWORK_REF is not a commit in {basicofr}: {ref}')
    ref_version = _text(['git', 'show', f'{ref}:VERSION'], cwd=basicofr)
    target_version = (REPO / 'VERSION').read_text(encoding='utf-8').strip()
    paper_version = _paper_version()
    if ref_version != target_version:
        raise RuntimeError(f'FRAMEWORK_REF VERSION {ref_version!r} != target VERSION {target_version!r}')
    if ref_version != paper_version:
        raise RuntimeError(f'FRAMEWORK_REF VERSION {ref_version!r} != PAPER basicofr_version {paper_version!r}')


def _ref_files(basicofr: Path, ref: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for arch_dir in ARCH_DIRS:
        if _run(['git', 'cat-file', '-e', f'{ref}:{arch_dir}'], cwd=basicofr).returncode != 0:
            raise RuntimeError(f'FRAMEWORK_REF missing {arch_dir}')
    proc = _run(['git', 'ls-tree', '-r', ref, '--', *ARCH_DIRS], cwd=basicofr)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode('utf-8', 'replace').strip())
    for line in proc.stdout.decode('utf-8').splitlines():
        meta, path = line.split('\t', 1)
        mode, kind, blob = meta.split()
        if mode == '120000':
            raise RuntimeError(f'FRAMEWORK_REF contains a symlink: {path}')
        if kind == 'blob':
            entries[path] = blob
    if not entries:
        raise RuntimeError('FRAMEWORK_REF has no framework arch files')
    return entries


def _local_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for arch_dir in ARCH_DIRS:
        root = REPO / arch_dir
        if root.is_symlink():
            raise RuntimeError(f'framework arch dir is a symlink: {arch_dir}')
        for path in root.rglob('*'):
            rel = path.relative_to(REPO)
            if any(part in IGNORED_DIRS for part in rel.parts) or path.name.endswith('.pyc'):
                continue
            if path.is_symlink():
                raise RuntimeError(f'framework arch path is a symlink: {rel}')
            if path.is_file():
                files[str(rel)] = path
    return files


def _check_local_matches(basicofr: Path, ref: str) -> None:
    expected = _ref_files(basicofr, ref)
    actual = _local_files()
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        detail = []
        if missing:
            detail.append('missing files: ' + ', '.join(missing[:8]))
        if extra:
            detail.append('extra files: ' + ', '.join(extra[:8]))
        raise RuntimeError('; '.join(detail))
    for rel, path in actual.items():
        blob = _text(['git', 'hash-object', str(path)], cwd=REPO)
        if blob != expected[rel]:
            raise RuntimeError(f'framework arch file differs from FRAMEWORK_REF: {rel}')


def _arch_state() -> tuple[list[str], list[str]]:
    present = []
    missing = []
    for arch_dir in ARCH_DIRS:
        path = REPO / arch_dir
        parent = path.parent
        if parent.is_symlink():
            raise RuntimeError(f'framework parent dir is a symlink: {parent.relative_to(REPO)}')
        if not parent.is_dir():
            raise RuntimeError(f'missing framework parent dir: {parent.relative_to(REPO)}')
        if path.is_symlink():
            raise RuntimeError(f'framework arch dir is a symlink: {arch_dir}')
        if path.exists():
            present.append(arch_dir)
        else:
            missing.append(arch_dir)
    return present, missing


def _hint(basicofr: Path) -> str:
    return f'python scripts/hydrate_framework.py --basicofr {basicofr}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basicofr', type=Path, default=None, help='BasicOFR checkout; defaults to ../../BasicOFR')
    ap.add_argument('--check', action='store_true', help='verify only, do not hydrate missing arch trees')
    args = ap.parse_args()

    basicofr = args.basicofr.expanduser().resolve() if args.basicofr else _default_basicofr()
    try:
        ref = _read_ref()
        _check_ref_and_version(basicofr, ref)
        present, missing = _arch_state()
        if missing and present:
            raise RuntimeError(f'partial framework arch state: present={present}, missing={missing}')
        if missing:
            if args.check:
                raise RuntimeError(f'missing framework archs: {", ".join(missing)}; run: {_hint(basicofr)}')
            archive = _run(['git', 'archive', ref, *ARCH_DIRS], cwd=basicofr)
            if archive.returncode != 0:
                raise RuntimeError(archive.stderr.decode('utf-8', 'replace').strip())
            extract = _run(['tar', '-x', '-C', str(REPO)], cwd=REPO, input_bytes=archive.stdout)
            if extract.returncode != 0:
                raise RuntimeError(extract.stderr.decode('utf-8', 'replace').strip())
            _check_local_matches(basicofr, ref)
            print(f'HYDRATE-OK hydrated and verified framework archs from {ref[:12]}')
            return 0
        _check_local_matches(basicofr, ref)
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print('HYDRATE-CHECK-FAIL' if args.check else 'HYDRATE-FAIL')
        print(f'ERROR: {exc}', file=sys.stderr)
        if 'missing framework archs' not in str(exc):
            print(f'run: {_hint(basicofr)}')
        return 1

    print(f'HYDRATE-OK {ref[:12]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
