# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that training quantization matches storage and propagates gradients."""

import numpy as np
import pytest
import warp as wp

pytest.importorskip("warp_nn")
pytestmark = pytest.mark.training
from neural_texture import pack_latents, unpack_latents
from neural_texture.training import _quantize_latent


@wp.kernel
def quantize(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = _quantize_latent(values[i])


def test_latent_quantization_matches_export_and_has_straight_through_gradient():
    # Include representable half-bin boundaries, saturation, and ordinary values.
    values = np.concatenate(
        (
            np.linspace(-2.0, 2.0, 257, dtype=np.float32),
            np.array([-2.0 / 3.0, 0.0, 2.0 / 3.0], dtype=np.float32),
        )
    )
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    source = wp.array(values, device=device, requires_grad=True)
    result = wp.empty_like(source, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(quantize, dim=values.size, inputs=[source], outputs=[result], device=device)
    expected = unpack_latents(pack_latents(values.reshape(1, -1, 1)), 1).reshape(-1)
    np.testing.assert_allclose(result.numpy(), expected, atol=2e-7, rtol=0.0)
    tape.backward(grads={result: wp.ones_like(result)})
    np.testing.assert_array_equal(source.grad.numpy(), np.ones_like(values))
