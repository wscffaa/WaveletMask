#!/usr/bin/env python3
"""Smoke test for WaveletMask imports, registry entries, and option files."""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import basicofr.archs  # noqa: F401,E402
import basicofr.losses  # noqa: F401,E402
import basicofr.models  # noqa: F401,E402
import waveletmask.archs  # noqa: F401,E402
import waveletmask.models  # noqa: F401,E402
from basicsr.utils.registry import ARCH_REGISTRY, MODEL_REGISTRY  # noqa: E402


def main() -> None:
    # Check archs are registered
    arch_names = [
        "WaveletMaskABNet",
        "WaveletMaskLocalNet",
        "WaveletMaskFlickerNet",
        "WaveletMaskFullNet",
        "WaveletMaskSimpleNet",
        "WaveletMaskLocalFusionNet",
        "WaveletMaskFlickerFusionNet",
        "WaveletMaskGatedFlickerNet",
    ]
    for name in arch_names:
        assert ARCH_REGISTRY.get(name) is not None, f"ARCH_REGISTRY missing: {name}"

    # Check model is registered
    assert MODEL_REGISTRY.get("WaveletMaskModel") is not None

    # Check option files
    option_files = sorted((REPO_ROOT / "options").rglob("*.yml"))
    assert option_files, "no option files found"
    for option_file in option_files:
        data = yaml.safe_load(option_file.read_text())
        assert data.get("name"), option_file
        assert data.get("model_type"), option_file
        assert data.get("network_g", {}).get("type"), option_file

    print("WaveletMask import, registry, and config smoke test passed")


if __name__ == "__main__":
    main()
