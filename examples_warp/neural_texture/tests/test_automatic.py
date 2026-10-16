# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for bounded automatic size selection."""

import numpy as np
import pytest

pytest.importorskip("warp_nn")
pytestmark = pytest.mark.training

import warp as wp
from neural_texture import (
    compress_texture,
    compress_texture_set,
    decode_image,
    evaluation,
    training,
)
from neural_texture.reference import _decode_pixels


@pytest.mark.parametrize("sacrifice_scalar", [False, True])
def test_automatic_choice_protects_each_texture_and_uses_common_samples(monkeypatch, sacrifice_scalar):
    seen = []

    class Scorer:
        count = 65536

        def __init__(self, image, *, sampled, device):
            assert sampled
            seen.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def mse(self, asset):
            prediction = np.array([0.1, 0.1, 0.1, 0.01], dtype=np.float64)
            if asset.metadata["latent_scale"] == 5:
                if sacrifice_scalar:
                    # Aggregate error improves, but the scalar map gets worse.
                    prediction[:3] = 0.08
                    prediction[3] = 0.02
                else:
                    prediction *= 1.03
            return prediction**2

    monkeypatch.setattr(evaluation, "_QualityEvaluator", Scorer)
    result = compress_texture_set(
        {
            "color": np.zeros((257, 257, 3), dtype=np.float32),
            "scalar": np.zeros((257, 257), dtype=np.float32),
        },
        steps=0,
        refinement_steps=0,
    )
    assert result.asset.metadata["latent_scale"] == (4 if sacrifice_scalar else 5)
    assert result.evaluation_pixels == 65536
    assert result.selection_reason
    assert len(seen) == 1  # Both candidates share the same source/sample evaluator.
    assert seen[0].closed


def test_selected_pixel_decode_matches_full_decode():
    result = compress_texture(
        np.zeros((7, 11, 3), dtype=np.float32),
        latent_scale=4,
        steps=0,
        refinement_steps=0,
    )
    indices = np.array([76, 0, 17, 17, 4, 33], dtype=np.int64)
    expected = decode_image(result.asset).reshape(-1, 3)[indices]
    actual = _decode_pixels(result.asset, pixel_indices=indices, batch_size=2)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert result.evaluation_pixels == 77
    assert not result.selection_reason


def test_gpu_sampling_and_block_reduction():
    indices = wp.empty(257, dtype=wp.int32, device="cuda:0")
    wp.launch(evaluation._sample_indices, dim=257, inputs=[indices, 1001])
    chosen = indices.numpy()
    assert np.all(chosen >= 0) and np.all(chosen < 1001)
    assert len(np.unique(chosen)) == 257
    # Includes a partial last block and multiple channels with different errors.
    errors = np.zeros((3, 512), dtype=np.float32)
    errors[:, :257] = np.array([1.0, 2.0, 3.0])[:, None]
    gpu_errors = wp.array(errors.reshape(-1), device="cuda:0")
    sums = wp.zeros(3, dtype=wp.float32, device="cuda:0")
    wp.launch_tiled(
        evaluation._sum_errors,
        dim=(2, 3),
        inputs=[gpu_errors, 512],
        outputs=[sums],
        block_dim=128,
    )
    np.testing.assert_array_equal(sums.numpy(), errors.sum(axis=1))


def test_tiled_training_loss_gradient_including_partial_block():
    source = wp.array(np.arange(259, dtype=np.float32), device="cuda:0", requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch_tiled(
            training._sum_training_loss,
            dim=2,
            inputs=[source],
            outputs=[loss],
            block_dim=128,
        )
    np.testing.assert_allclose(loss.numpy(), np.arange(259).sum())
    tape.backward(loss)
    np.testing.assert_array_equal(source.grad.numpy(), np.ones(259))


def test_weighted_training_loss_gradient():
    values = np.arange(6, dtype=np.float32).reshape(3, 2) / 5
    targets = np.array([[0.5, 0.25], [0.0, 1.0]], dtype=np.float16)
    indices = np.array([1, 0, 1], dtype=np.int32)
    weights = np.array([1.0, 2.0], dtype=np.float32)
    prediction = wp.array(values, device="cuda:0", requires_grad=True)
    target = wp.array(targets, device="cuda:0")
    pixels = wp.array(indices, device="cuda:0")
    channel_weights = wp.array(weights, device="cuda:0")
    errors = wp.zeros((3, 2), device="cuda:0", requires_grad=True)
    flat_errors = errors.flatten()
    loss = wp.zeros(1, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(
            training._mse,
            dim=(3, 2),
            inputs=[prediction, target, pixels, channel_weights, 1.0 / 9.0],
            outputs=[errors],
        )
        wp.launch_tiled(
            training._sum_training_loss,
            dim=1,
            inputs=[flat_errors],
            outputs=[loss],
            block_dim=128,
        )
    difference = values - targets[indices].astype(np.float32)
    np.testing.assert_allclose(loss.numpy(), [(difference**2 * weights).sum() / 9], rtol=1e-6)
    tape.backward(loss)
    np.testing.assert_allclose(prediction.grad.numpy(), 2 * difference * weights / 9, atol=1e-7)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluator_destroys_context_even_on_initialization_failure(monkeypatch, fail):
    destroyed = []

    class Context:
        def destroy(self):
            destroyed.append(True)

    def initialize(self, image, pixels):
        self.context = Context()
        if fail:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(evaluation._QualityEvaluator, "_initialize", initialize)
    monkeypatch.setattr(wp, "synchronize_device", lambda device: None)
    if fail:
        with pytest.raises(RuntimeError, match="initialization failed"):
            evaluation._QualityEvaluator(np.zeros((2, 2, 3)), sampled=True, device="cuda:0")
    else:
        with evaluation._QualityEvaluator(np.zeros((2, 2, 3)), sampled=True, device="cuda:0") as scorer:
            pass
        scorer.close()  # Idempotent.
    assert destroyed == [True]
