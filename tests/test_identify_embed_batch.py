"""Regression test for identify._embed_batch — the batched ECAPA window embedder.

Uses a fake model (no GPU / no real ECAPA) to verify the batching logic: order is
preserved and each clip's embedding equals what the model would return for that clip
alone, across MIXED clip lengths. This guards the grouping-by-length design that makes
batching bit-exact (zero-padding + wav_lens was proven NOT exact — see the docstring).
"""
import numpy as np
import torch

from pipeline.identify.runner import _embed_batch


class FakeModel:
    """encode_batch(wavs) -> (B, 1, 3) deterministic per-row 'embedding' [sum, mean, len].

    Each row depends ONLY on its own input row, so a correct _embed_batch must return,
    for every clip, exactly what encode_batch returns for that clip embedded alone.
    """

    def encode_batch(self, batch):  # batch: (B, T) float tensor, all rows same length
        s = batch.sum(dim=1)
        m = batch.mean(dim=1)
        ln = torch.full((batch.shape[0],), float(batch.shape[1]))
        return torch.stack([s, m, ln], dim=1).unsqueeze(1)  # (B, 1, 3)


def _ref(model, clip):
    t = torch.from_numpy(clip).unsqueeze(0)
    return model.encode_batch(t).squeeze(1)[0]


def test_embed_batch_matches_per_clip_and_preserves_order():
    rng = np.random.default_rng(0)
    # Mixed lengths: mostly 24000 (full 1.5s window) with every 5th half-length, like a
    # real sliding-window grid whose last window is clamped short.
    clips = [
        rng.standard_normal(24000 if i % 5 else 12000).astype("float32")
        for i in range(53)
    ]
    model = FakeModel()
    out = _embed_batch(model, clips, device="cpu", max_bs=16)

    assert len(out) == len(clips)
    for i, c in enumerate(clips):
        assert torch.allclose(out[i], _ref(model, c), atol=1e-5), f"mismatch at index {i}"


def test_embed_batch_all_same_length():
    rng = np.random.default_rng(1)
    clips = [rng.standard_normal(24000).astype("float32") for _ in range(40)]
    model = FakeModel()
    out = _embed_batch(model, clips, device="cpu", max_bs=128)
    assert len(out) == 40
    for i, c in enumerate(clips):
        assert torch.allclose(out[i], _ref(model, c), atol=1e-5)


def test_embed_batch_empty():
    assert _embed_batch(FakeModel(), [], device="cpu") == []
