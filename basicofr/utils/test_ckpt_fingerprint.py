"""Unit tests for ckpt_fingerprint — test-resume must not reuse stale outputs
produced by a different checkpoint (the cross-checkpoint incident)."""

import importlib.util
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parent / "ckpt_fingerprint.py"
_spec = importlib.util.spec_from_file_location("ckpt_fingerprint_under_test", _MOD_PATH)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


@pytest.mark.unit
class TestCheckpointFingerprint:
    def test_fingerprint_records_path_and_size(self, tmp_path):
        ckpt = tmp_path / "net_g_65000.pth"
        ckpt.write_bytes(b"\x00" * 123)
        fp = cf.checkpoint_fingerprint(str(ckpt))
        assert fp["path"] == str(ckpt)
        assert fp["size"] == 123

    def test_fingerprint_missing_file_size_none(self, tmp_path):
        fp = cf.checkpoint_fingerprint(str(tmp_path / "absent.pth"))
        assert fp["path"].endswith("absent.pth")
        assert fp["size"] is None

    def test_none_checkpoint(self):
        assert cf.checkpoint_fingerprint(None) == {"path": None, "size": None}


@pytest.mark.unit
class TestFingerprintMatches:
    def test_missing_fingerprint_is_failclosed(self, tmp_path):
        # legacy outputs with no fingerprint -> must re-infer (False)
        out = tmp_path / "outputs"
        out.mkdir()
        assert cf.fingerprint_matches(str(out), {"path": "a/b.pth", "size": 10}) is False

    def test_same_checkpoint_matches(self, tmp_path):
        out = tmp_path / "outputs"
        out.mkdir()
        fp = {"path": "exp/a1/models/net_g.pth", "size": 999}
        cf.write_fingerprint(str(out), fp)
        assert cf.fingerprint_matches(str(out), fp) is True

    def test_different_variant_path_does_not_match(self, tmp_path):
        out = tmp_path / "outputs"
        out.mkdir()
        cf.write_fingerprint(str(out), {"path": "exp/a1/models/net_g.pth", "size": 999})
        # same size but a DIFFERENT variant path -> must re-infer
        assert (
            cf.fingerprint_matches(str(out), {"path": "exp/a3/models/net_g.pth", "size": 999})
            is False
        )

    def test_same_path_different_size_does_not_match(self, tmp_path):
        out = tmp_path / "outputs"
        out.mkdir()
        cf.write_fingerprint(str(out), {"path": "exp/a1/models/net_g.pth", "size": 100})
        assert (
            cf.fingerprint_matches(str(out), {"path": "exp/a1/models/net_g.pth", "size": 200})
            is False
        )

    def test_unknown_size_falls_back_to_path_equality(self, tmp_path):
        out = tmp_path / "outputs"
        out.mkdir()
        cf.write_fingerprint(str(out), {"path": "exp/a1/models/net_g.pth", "size": None})
        assert (
            cf.fingerprint_matches(str(out), {"path": "exp/a1/models/net_g.pth", "size": None})
            is True
        )
