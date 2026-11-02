# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples_warp"))

from example_warp_optix_usd_pathtracing import (
    _authored_camera,
    _camera_for_scene,
    _default_camera_speed,
    _world_bounds,
)
from warp_optix.pathtracing.scene import Scene


@pytest.mark.usd
def test_usd_bound_camera_is_preferred_and_bounds_are_world_space(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    path = tmp_path / "camera.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    camera = UsdGeom.Camera.Define(stage, "/World/Overview")
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 5.0))
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(-1.0, -1.0, 0.0), Gf.Vec3f(1.0, -1.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().customLayerData = {"cameraSettings": {"boundCamera": "/World/Overview"}}
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False)
    result = _camera_for_scene(SimpleNamespace(scene=scene, usd_scene=scene.usd_scene))
    position, _yaw, _pitch, _fov, radius, camera_path = result
    np.testing.assert_allclose(position, (0.0, 2.0, 5.0))
    assert radius == pytest.approx(np.sqrt(2.0))
    assert camera_path == "/World/Overview"


def test_fallback_camera_bounds_ignore_extreme_flat_ground():
    subject = SimpleNamespace(
        mesh_index=0,
    )
    ground = SimpleNamespace(
        mesh_index=1,
    )
    scene = SimpleNamespace(
        _instances=[subject, ground],
        _meshes=[
            SimpleNamespace(vertices=np.asarray(((-0.1, 0.0, -0.05), (0.1, 0.08, 0.05)))),
            SimpleNamespace(vertices=np.asarray(((-25.0, 0.0, -25.0), (25.0, 0.0, 25.0)))),
        ],
        _instance_transform_cache=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
    )

    minimum, maximum = _world_bounds(SimpleNamespace(scene=scene))

    np.testing.assert_allclose(minimum, (-0.1, 0.0, -0.05))
    np.testing.assert_allclose(maximum, (0.1, 0.08, 0.05))


def test_camera_speed_has_practical_floor_for_centimeter_assets():
    assert _default_camera_speed(0.1) == pytest.approx(0.25)
    assert _default_camera_speed(10.0) == pytest.approx(7.5)


@pytest.mark.usd
def test_unbound_follow_camera_is_not_selected_automatically(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "follow_camera.usda"))
    camera = UsdGeom.Camera.Define(stage, "/World/VelocityFollowCamera")
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 10.0, 0.0))
    stage.GetRootLayer().Save()
    usd_scene = SimpleNamespace(stage=stage)

    assert _authored_camera(SimpleNamespace(usd_scene=usd_scene)) is None
