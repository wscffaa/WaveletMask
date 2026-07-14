"""Chunk batching plan for test-time inference speedup.

Pure-python helper (no torch) so it stays unit-testable on CPU without
mamba_ssm. Groups same-size chunks into batches of at most ``test_batch_size``
so ``RTNGANModel.nondist_validation`` can run multiple equal-size chunks through
the network in a single batched forward.

Hard invariant: ``test_batch_size <= 1`` reproduces the original
one-chunk-at-a-time sequence, keeping the shared training-validation path
numerically and behaviorally unchanged.

Tail chunks whose length differs from ``chunk_size`` naturally fall into their
own batch (size mismatch breaks the group), so no padding is introduced and
inference results stay numerically equivalent to per-chunk processing.
"""
from __future__ import annotations

from typing import List, Tuple

ChunkSpan = Tuple[int, int, int]  # (chunk_idx, start, end)


def plan_chunk_batches(
    num_chunks: int,
    chunk_size: int,
    effective_frames: int,
    test_batch_size: int,
) -> List[List[ChunkSpan]]:
    """Return chunk batches as a list of groups; each group is a list of
    ``(chunk_idx, start, end)`` spans sharing the same length.

    The (start, end) computation mirrors ``nondist_validation`` so the batched
    path covers exactly the same frames in the same order.
    """
    chunks: List[ChunkSpan] = []
    for ci in range(num_chunks):
        start = ci * chunk_size
        if ci == num_chunks - 1:
            end = effective_frames
        else:
            end = min((ci + 1) * chunk_size, effective_frames)
        chunks.append((ci, start, end))

    bs = max(int(test_batch_size), 1)
    if bs == 1:
        # One chunk per group == original per-chunk loop (training-val safe).
        return [[c] for c in chunks]

    groups: List[List[ChunkSpan]] = []
    cur: List[ChunkSpan] = []
    cur_size = None
    for c in chunks:
        size = c[2] - c[1]
        if cur and (size != cur_size or len(cur) >= bs):
            groups.append(cur)
            cur = []
        cur.append(c)
        cur_size = size
    if cur:
        groups.append(cur)
    return groups
