"""Unit tests for peak-memory + FPS extensions to the complexity analyzer.

Context: Symmetry round-1 revision R1-4 / R2-5 ask for inference peak GPU
memory and FPS alongside Params/FLOPs/Runtime. These must be added to the
shared tool without breaking its existing output contract (consumed by
``batch_complexity.py`` and paper-data generation).

CPU-only assertions run on Mac (tool-availability check); the CUDA peak-memory
path is exercised on a GPU server.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from basicofr.utils.complexity import analyze_model, measure_peak_memory


class _PerFrameConv(nn.Module):
    """Minimal (B,T,C,H,W) model so analyze_model's 5D path works on CPU."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        y = self.conv(x.reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w)


@pytest.mark.unit
def test_analyze_model_exposes_fps_and_peak_mem_keys():
    r = analyze_model(_PerFrameConv(), "tiny", (1, 7, 3, 128, 128),
                      device="cpu", measure_speed=True)
    assert "fps" in r
    assert "peak_mem_gb" in r


@pytest.mark.unit
def test_existing_contract_keys_preserved():
    # Downstream readers (batch_complexity, paper_data) rely on these keys.
    r = analyze_model(_PerFrameConv(), "tiny", (1, 7, 3, 128, 128),
                      device="cpu", measure_speed=True)
    for key in ("name", "params_m", "params_all_m", "macs_g", "flops_g",
                "flops_paper_g", "per_frame_flops_paper_g", "flops_backend",
                "runtime_ms", "ms_per_frame", "per_frame_runtime_ms",
                "input_size"):
        assert key in r, f"missing contract key: {key}"


@pytest.mark.unit
def test_fps_matches_inverse_of_per_frame_runtime():
    r = analyze_model(_PerFrameConv(), "tiny", (1, 7, 3, 128, 128),
                      device="cpu", measure_speed=True)
    assert r["per_frame_runtime_ms"] > 0
    expected = 1000.0 / r["per_frame_runtime_ms"]
    assert r["fps"] == pytest.approx(expected, rel=0.05)


@pytest.mark.unit
def test_no_speed_disables_fps_and_peak_mem():
    r = analyze_model(_PerFrameConv(), "tiny", (1, 7, 3, 128, 128),
                      device="cpu", measure_speed=False)
    assert r["fps"] == -1
    assert r["peak_mem_gb"] == -1.0


@pytest.mark.unit
def test_peak_memory_cpu_returns_sentinel():
    # CPU cannot measure GPU peak memory -> explicit sentinel, never fake 0.
    assert measure_peak_memory(_PerFrameConv(), (1, 7, 3, 128, 128),
                               device="cpu") == -1.0


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peak_memory_cuda_is_positive():
    mem = measure_peak_memory(_PerFrameConv(), (1, 7, 3, 64, 64), device="cuda")
    assert mem > 0.0
