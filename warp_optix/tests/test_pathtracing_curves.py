# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
from warp_optix.pathtracing import Curve, Mesh, PathTracerAPI, Scene


def _points():
    return np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    ("points", "basis", "indices", "primitive_type", "control_points", "sbt_offset", "geometry_type"),
    [
        (_points(), "linear", (0, 1), "round_linear", 2, 2, 1),
        (np.arange(21, dtype=np.float32).reshape(7, 3), "cubic_bezier", (0, 3), "round_cubic_bezier", 4, 4, 2),
    ],
)
def test_curve_default_topology_and_sbt_slot(
    points, basis, indices, primitive_type, control_points, sbt_offset, geometry_type
):
    curve = Curve(points, radii=np.linspace(0.1, 0.04, len(points)), basis=basis)

    np.testing.assert_array_equal(curve.indices, indices)
    assert curve.basis == basis
    assert curve.primitive_type == primitive_type
    assert curve.control_point_count == control_points
    assert curve.sbt_offset == sbt_offset
    assert curve.geometry_type == geometry_type


def test_curve_accepts_scalar_radius_disjoint_segments_and_per_primitive_materials():
    points = np.vstack((_points(), _points() + (4.0, 0.0, 0.0)))
    curve = Curve(points, radii=0.05, segment_indices=np.array((0, 1, 3, 4)), material_id=7, material_ids=(2, 4, 1, 0))
    mesh = Mesh(
        vertices=np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)), dtype=np.float32),
        indices=np.array(((0, 1, 2), (1, 3, 2)), dtype=np.uint32),
        material_id=7,
        material_ids=(3, 5),
    )

    np.testing.assert_array_equal(curve.indices, (0, 1, 3, 4))
    np.testing.assert_allclose(curve.radii, 0.05)
    np.testing.assert_array_equal(curve.material_ids, (2, 4, 1, 0))
    np.testing.assert_array_equal(mesh.material_ids, (3, 5))
    # Dynamic curves may collapse segments to zero radius; static curves may not.
    np.testing.assert_allclose(Curve(_points(), radii=(0.1, 0.0, 0.1), dynamic=True).radii, (0.1, 0.0, 0.1))


@pytest.mark.parametrize(
    ("points", "kwargs", "message"),
    [
        (np.zeros((6, 3), dtype=np.float32), {"basis": "cubic_bezier"}, "3\\*N \\+ 1"),
        (np.zeros((7, 3), dtype=np.float32), {"basis": "cubic_bezier", "segment_indices": (4,)}, "four consecutive"),
        (_points(), {"material_ids": (1,)}, "material_ids"),
        (_points(), {"material_ids": (0.0, 1.0)}, "material_ids"),
        (_points(), {"material_ids": (-1, 0)}, "material_ids"),
        (_points(), {"radii": (0.1, 0.2)}, "shape"),
        (_points(), {"radii": (0.1, 0.0, 0.1)}, "positive"),
        (_points(), {"segment_indices": (2,)}, "two consecutive"),
        (_points(), {"segment_indices": ()}, "non-empty"),
    ],
)
def test_curve_rejects_invalid_inputs(points, kwargs, message):
    with pytest.raises(ValueError, match=message):
        Curve(points, **{"radii": 0.1} | kwargs)


@pytest.mark.parametrize("basis", ["linear", "cubic_bezier"])
def test_pathtracer_api_creates_reusable_curve_geometry(basis):
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))
    points = _points() if basis == "linear" else np.zeros((4, 3), dtype=np.float32)
    curve_id = api.create_curve(points, radii=0.05, material_id=1, basis=basis, dynamic=True)
    instance_id = api.create_instance(curve_id)
    curve = scene._meshes[curve_id]

    assert (curve_id, instance_id) == (0, 0)
    assert (scene.curve_count, scene.mesh_count, scene.geometry_count) == (1, 0, 1)
    assert scene._instances[instance_id].mesh_index == curve_id
    assert curve.basis == basis
    assert curve.dynamic
    assert curve.material_id == 1
    assert (curve.material_ids == 1).all()
    assert scene.materials.count == 2


def test_pathtracer_api_creates_one_dynamic_curve_instance_for_arrow_batch():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))

    batch = api.create_arrow_batch(
        capacity=4,
        small_radius=0.02,
        large_radius=0.06,
        tip_length_ratio=0.25,
        material_ids=(0, 1, 0, 1),
    )
    curve = scene._meshes[batch.geometry_id]

    assert batch.capacity == 4
    assert scene.geometry_count == 1
    assert scene.instance_count == 1
    assert scene._instances[batch.instance_id].mesh_index == batch.geometry_id
    assert curve.dynamic
    np.testing.assert_array_equal(curve.indices, (0, 2, 4, 6, 8, 10, 12, 14))
    np.testing.assert_array_equal(curve.material_ids, (0, 0, 1, 1, 0, 0, 1, 1))
    np.testing.assert_allclose(curve.radii, 0.0)


def test_host_arrow_batch_update_builds_shaft_tip_and_clears_inactive_slots():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))
    batch = api.create_arrow_batch(3, 0.02, 0.06, 0.25)

    api.update_arrow_batch(
        batch,
        starts=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ends=((0.0, 0.0, 2.0), (2.0, 0.0, 0.0)),
        material_ids=(1, 0),
    )
    curve = scene._meshes[batch.geometry_id]
    points = curve.vertices.reshape(3, 4, 3)
    widths = curve.radii.reshape(3, 4)

    np.testing.assert_allclose(
        points[0],
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.5), (0.0, 0.0, 1.5), (0.0, 0.0, 2.0)),
    )
    np.testing.assert_allclose(
        points[1],
        ((1.0, 0.0, 0.0), (1.75, 0.0, 0.0), (1.75, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(widths[:2], ((0.02, 0.02, 0.06, 0.0),) * 2)
    np.testing.assert_allclose(points[2], 0.0)
    np.testing.assert_allclose(widths[2], 0.0)
    np.testing.assert_array_equal(curve.material_ids[:4], (1, 1, 0, 0))
    assert batch.active_count == 2

    api.update_arrow_batch(
        batch,
        starts=((3.0, 0.0, 0.0),),
        ends=((3.0, 1.0, 0.0),),
    )
    np.testing.assert_allclose(curve.vertices[4:], 0.0)
    np.testing.assert_allclose(curve.radii[4:], 0.0)
    assert batch.active_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0, "small_radius": 0.1, "large_radius": 0.2}, "capacity"),
        ({"capacity": 1, "small_radius": 0.0, "large_radius": 0.2}, "positive"),
        ({"capacity": 1, "small_radius": 0.2, "large_radius": 0.1}, "at least"),
        (
            {
                "capacity": 1,
                "small_radius": 0.1,
                "large_radius": 0.2,
                "tip_length_ratio": 1.0,
            },
            "between",
        ),
    ],
)
def test_arrow_batch_rejects_invalid_shape_parameters(kwargs, message):
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    with pytest.raises(ValueError, match=message):
        api.create_arrow_batch(**kwargs)
