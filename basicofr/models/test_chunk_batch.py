"""L1 unit tests for chunk batching plan (test-time batch inference speedup).

`plan_chunk_batches` groups same-size chunks into batches of at most
`test_batch_size`, so RTNGANModel.nondist_validation can run multiple equal-size
chunks through the network in one batched forward.

Hard invariant: `test_batch_size <= 1` must reproduce the original
one-chunk-at-a-time sequence (protects the shared training-validation path).

Pure-python module → runnable on CPU without torch/mamba_ssm.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from chunk_batch import plan_chunk_batches  # noqa: E402


def _flatten(groups):
    return [c for g in groups for c in g]


@pytest.mark.unit
def test_default_batch_size_one_is_per_chunk():
    # 5 chunks, chunk_size 30, eff 125 (tail chunk size 5)
    groups = plan_chunk_batches(num_chunks=5, chunk_size=30, effective_frames=125, test_batch_size=1)
    assert [len(g) for g in groups] == [1, 1, 1, 1, 1]
    assert _flatten(groups) == [(0, 0, 30), (1, 30, 60), (2, 60, 90), (3, 90, 120), (4, 120, 125)]


@pytest.mark.unit
def test_batch_groups_same_size_chunks():
    groups = plan_chunk_batches(num_chunks=5, chunk_size=30, effective_frames=125, test_batch_size=8)
    # first 4 size-30 chunks batched together, tail size-5 chunk alone
    assert [len(g) for g in groups] == [4, 1]
    assert groups[0] == [(0, 0, 30), (1, 30, 60), (2, 60, 90), (3, 90, 120)]
    assert groups[1] == [(4, 120, 125)]


@pytest.mark.unit
def test_batch_size_caps_group_length():
    groups = plan_chunk_batches(num_chunks=5, chunk_size=30, effective_frames=125, test_batch_size=2)
    assert groups == [
        [(0, 0, 30), (1, 30, 60)],
        [(2, 60, 90), (3, 90, 120)],
        [(4, 120, 125)],
    ]


@pytest.mark.unit
def test_equal_size_all_chunks_single_batch():
    # eff exactly divisible: 4 chunks size 30 = 120, all same size
    groups = plan_chunk_batches(num_chunks=4, chunk_size=30, effective_frames=120, test_batch_size=8)
    assert [len(g) for g in groups] == [4]


@pytest.mark.unit
def test_full_coverage_no_overlap():
    groups = plan_chunk_batches(num_chunks=4, chunk_size=30, effective_frames=120, test_batch_size=8)
    chunks = _flatten(groups)
    assert chunks[0][1] == 0
    assert chunks[-1][2] == 120
    for a, b in zip(chunks, chunks[1:]):
        assert a[2] == b[1]  # contiguous: end == next start


@pytest.mark.unit
def test_no_chunking_single_chunk():
    groups = plan_chunk_batches(num_chunks=1, chunk_size=125, effective_frames=125, test_batch_size=8)
    assert groups == [[(0, 0, 125)]]
