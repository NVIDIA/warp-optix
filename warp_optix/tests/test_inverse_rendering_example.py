# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples_warp"))

import example_warp_optix_inverse_rendering as example  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]


def test_light_is_recovered_through_detached_visibility():
    with wp.ScopedDevice("cuda:0"), wo.create_context() as ctx:
        renderer = example.InverseRenderer(ctx, 96, 72)
        grey = wp.full(renderer.vertices.shape[0], wp.vec3(0.5, 0.5, 0.5), dtype=wp.vec3)
        true_light = np.array([[2.0, 3.5, 1.5]], dtype=np.float32)
        photo = wp.empty(renderer.shape, dtype=wp.vec3)
        renderer.render(grey, wp.array(true_light, dtype=wp.vec3), photo)

        light = wp.array([[-1.5, 2.5, 2.5]], dtype=wp.vec3, requires_grad=True)
        example.fit(renderer, photo, grey, light, [light], 0.1, 300, "light")
        np.testing.assert_allclose(light.numpy(), true_light, atol=1e-3)
