# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

import warp_optix as woptix
from warp_optix._runtime import runtime as _runtime
from warp_optix._runtime.hit_kernels import HitKernel as _NativeHitKernel
from warp_optix.pathtracing import PathTracerAPI


def _new_gpu_api(width=64, height=48):
    try:
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA device unavailable")
        _runtime.require_optix()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")

    api = PathTracerAPI(
        width=width,
        height=height,
        denoiser="none",
        enable_set=False,
        enable_cuda_graphs=True,
        backface_culling=False,
    )
    assert api.initialize()
    return api


def _render_glass_sheet(reverse_winding):
    api = _new_gpu_api()
    try:
        glass = api.scene.materials.add_glass(
            ior=1.45,
            tint=(0.58, 0.74, 1.0),
            transmission=0.97,
        )
        backdrop = api.create_emissive_material((0.8, 0.3, 0.05), intensity=3.0)
        quad = np.asarray(
            ((-1.4, -1.0, 0.0), (1.4, -1.0, 0.0), (1.4, 1.0, 0.0), (-1.4, 1.0, 0.0)),
            dtype=np.float32,
        )
        glass_indices = np.asarray(
            ((0, 2, 1), (0, 3, 2)) if reverse_winding else ((0, 1, 2), (0, 2, 3)),
            dtype=np.uint32,
        )
        backdrop_indices = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
        glass_geometry = api.create_mesh(
            quad + np.asarray((0.0, 0.0, 0.5), dtype=np.float32),
            glass_indices,
            material_id=glass,
        )
        backdrop_geometry = api.create_mesh(
            quad,
            backdrop_indices,
            material_id=backdrop,
        )
        api.create_instance(glass_geometry)
        api.create_instance(backdrop_geometry)
        api.set_camera_look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        api.build_scene()
        api.set_environment_color((0.02, 0.04, 0.08))

        frames = []
        for _ in range(4):
            api.render_frame()
            frames.append(api.get_frame()[..., :3].copy())
        return float(np.mean(frames))
    finally:
        api.close()


def test_two_sided_physical_glass_transmits_for_both_windings():
    """Catch double face orientation in the real thin-glass render path."""
    front_luminance = _render_glass_sheet(reverse_winding=False)
    back_luminance = _render_glass_sheet(reverse_winding=True)

    assert front_luminance > 0.2
    assert back_luminance > 0.2
    assert back_luminance / front_luminance == pytest.approx(1.0, rel=0.15)


def test_host_instance_transform_update_produces_motion_vectors():
    """Keep the motion transform buffer synchronized with host TLAS updates."""
    api = _new_gpu_api(width=96, height=64)
    try:
        material = api.create_emissive_material((1.0, 0.3, 0.05), intensity=2.0)
        quad = np.asarray(
            ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)),
            dtype=np.float32,
        )
        indices = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
        geometry = api.create_mesh(quad, indices, material_id=material)
        instance = api.create_instance_with_transform(
            geometry,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        api.set_camera_look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        api.build_scene()

        api.render_frame()
        api.render_frame()
        static_motion = api._viewer._motion_buffer.numpy().copy()
        assert float(np.max(np.abs(static_motion))) < 5.0e-5

        api.set_instance_transform(
            instance,
            (0.25, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        api.rebuild_tlas()
        api.render_frame()
        moving_motion = api._viewer._motion_buffer.numpy()
        magnitude = np.linalg.norm(moving_motion, axis=-1)

        assert float(np.max(magnitude)) > 1.0
        assert int(np.count_nonzero(magnitude > 1.0e-5)) > 100
    finally:
        api.close()


def test_scene_rebuild_after_adding_instances_resizes_motion_buffers():
    """Runtime instance additions must not upload into the old motion buffer."""
    api = _new_gpu_api()
    try:
        material = api.create_emissive_material((1.0, 0.3, 0.05), intensity=2.0)
        vertices = np.asarray(
            ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
            dtype=np.float32,
        )
        geometry = api.create_mesh(
            vertices, np.asarray(((0, 1, 2),), dtype=np.uint32), material_id=material
        )
        api.set_camera_look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        for count in (1, 2, 3):
            api.create_instance(geometry)
            api.build_scene()
            api.render_frame()
            assert api.scene._device_instance_transforms.shape == (count, 12)
            assert api._viewer._prev_instance_transforms_buffer.shape == (count, 12)
            assert np.isfinite(api.get_frame()).all()
    finally:
        api.close()


def test_tlas_updates_periodically_rebuild_for_traversal_quality():
    """Repeated refits must eventually rebuild the TLAS."""
    api = _new_gpu_api()
    try:
        material = api.create_emissive_material((1.0, 1.0, 1.0), intensity=1.0)
        vertices = np.asarray(
            ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
            dtype=np.float32,
        )
        geometry = api.create_mesh(
            vertices, np.asarray(((0, 1, 2),), dtype=np.uint32), material_id=material
        )
        api.create_instance(geometry)
        api.build_scene()
        scene = api.scene
        for _ in range(scene._max_tlas_updates):
            api.rebuild_tlas()
        assert scene._tlas_updates_since_build == scene._max_tlas_updates
        api.rebuild_tlas()
        assert scene._tlas_updates_since_build == 0
    finally:
        api.close()
