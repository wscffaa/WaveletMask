"""Regression test for ERR: NR-test crash when val.metrics is null.

Root cause: RTNGANModel._compute_dataset_level_metrics used
`val_opt.get('metrics', {})`. When a TestNR config sets `metrics: null`,
the key exists with value None, so .get returns None (not {}), and the
subsequent `'fid' not in None` raised TypeError. This blocked every
RealMovie TestNR run (RRTN/MGFR/MambaOFR/WaveMamba).
"""
from __future__ import annotations

import pytest

from basicofr.models.rtn_model import RTNGANModel


def _call(opt_val):
    """Invoke the guard path without constructing a full model."""
    model = RTNGANModel.__new__(RTNGANModel)
    model.opt = {"val": opt_val}
    return RTNGANModel._compute_dataset_level_metrics(
        model, dataloader=None, save_img=True
    )


@pytest.mark.unit
def test_metrics_null_does_not_crash():
    # NR config: `metrics: null` -> must early-return, not raise TypeError
    assert _call({"metrics": None}) is None


@pytest.mark.unit
def test_metrics_key_absent_does_not_crash():
    assert _call({}) is None


@pytest.mark.unit
def test_metrics_without_fid_early_returns():
    assert _call({"metrics": {"psnr": {"type": "calculate_psnr"}}}) is None
