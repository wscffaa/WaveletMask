"""Regression tests for RTNGANModel best-checkpoint saving policy."""
from __future__ import annotations

import torch
import pytest

from basicofr.models.rtn_model import RTNGANModel, _resolve_save_best_settings


@pytest.mark.unit
def test_save_best_defaults_enabled_when_validation_metrics_exist():
    enabled, metric = _resolve_save_best_settings(
        train_opt={},
        val_opt={"metrics": {"psnr": {"type": "calculate_psnr"}, "ssim": {"type": "calculate_ssim"}}},
    )

    assert enabled is True
    assert metric == "psnr"


@pytest.mark.unit
def test_save_best_explicit_false_keeps_legacy_opt_out():
    enabled, metric = _resolve_save_best_settings(
        train_opt={"save_best": False},
        val_opt={"metrics": {"psnr": {"type": "calculate_psnr"}}},
    )

    assert enabled is False
    assert metric == "psnr"


@pytest.mark.unit
def test_save_best_rejects_metric_not_present_in_validation_metrics():
    with pytest.raises(ValueError, match="not in val.metrics"):
        _resolve_save_best_settings(
            train_opt={"save_best_metric": "lpips"},
            val_opt={"metrics": {"psnr": {"type": "calculate_psnr"}}},
        )


@pytest.mark.unit
def test_save_best_true_requires_validation_metrics():
    with pytest.raises(ValueError, match="requires val.metrics"):
        _resolve_save_best_settings(train_opt={"save_best": True}, val_opt={"metrics": None})


@pytest.mark.unit
def test_save_best_model_writes_net_g_best_with_metadata(tmp_path):
    model = RTNGANModel.__new__(RTNGANModel)
    model.opt = {"path": {"models": str(tmp_path)}}
    model.ema_decay = 0
    model.net_g = torch.nn.Linear(1, 1)

    RTNGANModel._save_best_model(model, current_iter=5000, metric_name="psnr", metric_value=20.1234)

    checkpoint = torch.load(tmp_path / "net_g_best.pth", map_location="cpu")
    assert "params" in checkpoint
    assert checkpoint["best_iter"] == 5000
    assert checkpoint["best_metric"] == "psnr"
    assert checkpoint["best_value"] == pytest.approx(20.1234)
