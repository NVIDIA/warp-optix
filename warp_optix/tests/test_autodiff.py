# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX launches on a ``wp.Tape``: detached by default, or with a supplied adjoint."""

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]

# A tilted quad, so hit distances differ per ray and the adjoint must re-trace them.
QUAD_VERTICES = np.array([[-1, -1, 0], [1, -1, 1], [1, 1, 1], [-1, 1, 0]], dtype=np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
SIDE = 8


@wp.func
def _ray_origin(i: int) -> wp.vec3:
    x = (float(i % SIDE) + 0.5) / float(SIDE) * 1.6 - 0.8
    y = (float(i // SIDE) + 0.5) / float(SIDE) * 1.6 - 0.8
    return wp.vec3(x, y, 5.0)


@wp.func
def _hit_distance(scene: wp.uint64, i: int) -> float:
    hit = wo.closest_hit(scene, _ray_origin(i), wp.vec3(0.0, 0.0, -1.0), 0.0, 100.0, wp.uint32(0xFF), wp.uint32(0))
    return wp.max(hit.t, 0.0)


@wp.kernel(module="unique")
def scaled_depth(scene: wp.uint64, scale: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = scale[0] * _hit_distance(scene, i)


@wp.kernel(module="unique")
def scaled_depth_adjoint(scene: wp.uint64, adj_out: wp.array[float], adj_scale: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(adj_scale, 0, adj_out[i] * _hit_distance(scene, i))


@wp.kernel
def sum_squares(values: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, values[i] * values[i])


@pytest.fixture
def scene():
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    vertices = wp.array(QUAD_VERTICES, dtype=wp.vec3, device=device)
    indices = wp.array(QUAD_INDICES, dtype=wp.int32, device=device)
    gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], device=device)
    forward = wo.create_pipeline(ctx, scaled_depth)
    adjoint = wo.create_pipeline(ctx, scaled_depth_adjoint)
    yield device, gas, forward, adjoint
    wo.destroy_context(ctx)


def _arrays(device, scale_value=1.5):
    scale = wp.array([scale_value], dtype=float, device=device, requires_grad=True)
    out = wp.zeros(SIDE * SIDE, dtype=float, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    return scale, out, loss


def test_supplied_adjoint_matches_analytic_gradient(scene):
    device, gas, forward, adjoint = scene
    scale, out, loss = _arrays(device)

    def backward():
        wo.launch(adjoint, dim=out.shape, inputs=[gas.handle, out.grad, scale.grad])

    with wp.Tape() as tape:
        wo.launch(forward, dim=out.shape, inputs=[gas.handle, scale, out], backward=backward)
        wp.launch(sum_squares, dim=out.shape, inputs=[out, loss], device=device)
    tape.backward(loss)

    depth = out.numpy() / 1.5
    assert np.all(depth > 0.0)
    # d/ds sum((s * t_i)^2) = 2 s sum(t_i^2)
    np.testing.assert_allclose(scale.grad.numpy()[0], 2.0 * 1.5 * np.sum(depth**2), rtol=1e-5)

    tape.zero()
    assert scale.grad.numpy()[0] == 0.0


def test_supplied_adjoint_matches_finite_differences(scene):
    device, gas, forward, adjoint = scene

    def evaluate(s):
        scale, out, loss = _arrays(device, s)
        wo.launch(forward, dim=out.shape, inputs=[gas.handle, scale, out])
        wp.launch(sum_squares, dim=out.shape, inputs=[out, loss], device=device)
        return float(loss.numpy()[0])

    scale, out, loss = _arrays(device)
    with wp.Tape() as tape:
        wo.launch(
            forward,
            dim=out.shape,
            inputs=[gas.handle, scale, out],
            backward=lambda: wo.launch(adjoint, dim=out.shape, inputs=[gas.handle, out.grad, scale.grad]),
        )
        wp.launch(sum_squares, dim=out.shape, inputs=[out, loss], device=device)
    tape.backward(loss)

    eps = 1e-2
    fd = (evaluate(1.5 + eps) - evaluate(1.5 - eps)) / (2.0 * eps)
    np.testing.assert_allclose(scale.grad.numpy()[0], fd, rtol=1e-3)


def test_launch_without_adjoint_is_detached(scene):
    device, gas, forward, _ = scene
    scale, out, loss = _arrays(device)
    with wp.Tape() as tape:
        wo.launch(forward, dim=out.shape, inputs=[gas.handle, scale, out])
        wp.launch(sum_squares, dim=out.shape, inputs=[out, loss], device=device)
    tape.backward(loss)
    # Gradients reach the launch outputs and stop there.
    assert np.any(out.grad.numpy() != 0.0)
    assert scale.grad.numpy()[0] == 0.0


def test_backward_is_ignored_without_tape(scene):
    device, gas, forward, _ = scene
    scale, out, _ = _arrays(device)
    calls = []
    wo.launch(forward, dim=out.shape, inputs=[gas.handle, scale, out], backward=lambda: calls.append(1))
    wp.synchronize_device(device)
    assert not calls


def test_backward_must_be_callable(scene):
    device, gas, forward, _ = scene
    scale, out, _ = _arrays(device)
    with pytest.raises(TypeError, match="backward"):
        wo.launch(forward, dim=out.shape, inputs=[gas.handle, scale, out], backward=1)
