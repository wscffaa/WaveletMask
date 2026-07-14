"""Checkpoint fingerprinting for test-resume safety.

Fixes the cross-checkpoint incident's secondary failure mode: the test resume
logic skipped a clip whenever ``output_dir`` already held enough PNGs, without
checking WHICH checkpoint produced them. Swapping the checkpoint then silently
reused stale images. We stamp a fingerprint of the producing checkpoint next to
its outputs and re-infer whenever it cannot be confirmed to match the current
checkpoint (fail-closed).

Pure stdlib (no torch) so it is unit-testable without a GPU/model instance.
"""
from __future__ import annotations

import json
import os
from os import path as osp
from typing import Optional

FINGERPRINT_FILENAME = ".ckpt_fingerprint"


def checkpoint_fingerprint(checkpoint_path: Optional[str]) -> dict:
    """Fingerprint a checkpoint by its path + byte size.

    Path distinguishes different variants (the incident's core failure); size
    distinguishes a re-trained checkpoint reusing the same filename. Size is
    ``None`` when the file is absent (e.g. remote-only), in which case matching
    falls back to path equality alone.
    """
    if not checkpoint_path:
        return {"path": None, "size": None}
    try:
        size = os.stat(checkpoint_path).st_size
    except OSError:
        size = None
    return {"path": str(checkpoint_path), "size": size}


def read_fingerprint(output_dir: str) -> Optional[dict]:
    fp_file = osp.join(output_dir, FINGERPRINT_FILENAME)
    if not osp.isfile(fp_file):
        return None
    try:
        with open(fp_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_fingerprint(output_dir: str, fingerprint: dict) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(osp.join(output_dir, FINGERPRINT_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(fingerprint, fh)
    except OSError:
        pass


def fingerprint_matches(output_dir: str, current: dict) -> bool:
    """True only when the recorded fingerprint provably matches *current*.

    Fail-closed: a missing or unreadable fingerprint returns ``False`` so legacy
    or unverifiable outputs are re-inferred rather than trusted. When the stored
    size is unknown (``None``), fall back to path equality so remote-only
    checkpoints still resume correctly.
    """
    stored = read_fingerprint(output_dir)
    if stored is None:
        return False
    if stored.get("path") != current.get("path"):
        return False
    if stored.get("size") is None or current.get("size") is None:
        return True
    return stored.get("size") == current.get("size")
