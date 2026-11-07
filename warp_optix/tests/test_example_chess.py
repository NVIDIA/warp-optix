# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the example's generated solids independently of GPU rendering."""

import importlib.util
from collections import Counter
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_helper():
    path = Path(__file__).resolve().parents[2] / "examples" / "scenes" / "chess.py"
    spec = importlib.util.spec_from_file_location("procedural_chess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_chess_components_are_closed_and_outward_facing():
    meshes = []
    instances = []
    textures = []
    materials = count()
    material_info = {}
    mesh_materials = []
    overrides = {}

    def add_material(**kwargs):
        mid = next(materials)
        material_info[mid] = kwargs
        return mid

    def create_instance(*args, **kwargs):
        instances.append((args, kwargs))
        return len(instances) - 1

    def create_mesh(p, t, normals, uvs, material_id):
        meshes.append((p, t, normals, uvs))
        mesh_materials.append(material_id)
        return len(meshes) - 1

    api = SimpleNamespace(
        scene=SimpleNamespace(
            set_gltf_textures=lambda images, **kwargs: textures.extend(images),
            materials=SimpleNamespace(add_gltf_material=add_material),
            add_light_sphere=lambda **kwargs: None,
        ),
        create_pbr_material=lambda **kwargs: next(materials),
        create_mesh=create_mesh,
        create_instance_with_transform=create_instance,
        set_instance_material=lambda instance, material: overrides.__setitem__(instance, material),
        set_use_procedural_sky=lambda *args: None,
        set_sky_parameters=lambda *args, **kwargs: None,
    )
    _load_helper().build_chess_scene(api)
    assert len(textures) == 4
    assert len(instances) > 100
    assert len(set(overrides.values())) == 32
    offsets = set()
    for material in set(overrides.values()):
        texture = material_info[material]["base_color_texture"]
        assert texture["index"] in (2, 3)
        assert set(texture["transform"]) == {"offset"}  # no grain rescaling
        offsets.add(tuple(texture["transform"]["offset"]))
    assert len(offsets) == 32
    piece_materials = {}
    for instance, (args, _) in enumerate(instances):
        mid, position = args[:2]
        if mesh_materials[mid] in (2, 3):
            assert instance in overrides
            # All wooden components within a board square use the same cut.
            square = tuple(np.round(np.asarray(position)[[0, 2]] + 3.5).astype(int))
            piece_materials.setdefault(square, set()).add(overrides[instance])
        else:
            assert instance not in overrides
    assert len(piece_materials) == 32
    assert all(len(values) == 1 for values in piece_materials.values())
    for p, t, n, uv in meshes:
        assert np.isfinite(p).all() and np.isfinite(n).all() and np.isfinite(uv).all()
        assert uv.shape == (len(p), 2)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        area = np.cross(b - a, c - a)
        assert (np.linalg.norm(area, axis=1) > 1e-9).all()
        assert (np.sum(area * n[t].mean(axis=1), axis=1) > 0).all()
        assert np.sum(a * np.cross(b, c)) / 6 > 0  # positive enclosed volume
        # Weld only for topology checks: UV seams and sharp edges need duplicates.
        _, welded = np.unique(np.round(p, 6), axis=0, return_inverse=True)
        tri = welded[t]
        directed = Counter((int(a), int(b)) for face in tri for a, b in zip(face, np.roll(face, -1)))
        assert all(count == 1 and directed[(b, a)] == 1 for (a, b), count in directed.items())


def test_procedural_textures_are_repeatable_rgba():
    helper = _load_helper()
    first = helper.wood_texture((0.8, 0.6, 0.4), (0.2, 0.1, 0.05), 64)
    second = helper.wood_texture((0.8, 0.6, 0.4), (0.2, 0.1, 0.05), 64)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8 and first.shape == (64, 64, 4)
    assert (first[:, :, 3] == 255).all()
    assert np.ptp(first[:, :, 0]) > 30


def test_beveled_extrusion_caps_have_planar_normals():
    # A concave outline makes ear-clipped triangles very different in area.
    outline = [(-1, -1), (1, -1), (1, 0), (0, 0), (0, 1), (-1, 1)]
    p, t, n, _ = _load_helper().beveled_extrude(outline, 0.3)
    cap_triangles = t[: 2 * (len(outline) - 2)]
    for triangle in cap_triangles:
        expected = np.array([0, 0, np.sign(p[triangle[0], 2])])
        np.testing.assert_allclose(n[triangle], np.tile(expected, (3, 1)), atol=1e-7)


def test_lathe_normals_are_continuous_across_seams_and_poles():
    p, _, n, _ = _load_helper().lathe([(0, 0), (0.3, 0.2), (0.3, 0.5), (0, 0.7)])
    poles = np.linalg.norm(p[:, [0, 2]], axis=1) < 1e-8
    np.testing.assert_allclose(n[poles][:, [0, 2]], 0, atol=1e-7)
    np.testing.assert_allclose(np.abs(n[poles, 1]), 1, atol=1e-7)
    groups = {}
    for position, normal in zip(p, n):
        key = tuple(np.round(position, 6))
        if key in groups:
            np.testing.assert_allclose(normal, groups[key], atol=1e-6)
        else:
            groups[key] = normal


def test_wood_mapping_uses_scene_units_for_board_and_pieces():
    helper = _load_helper()
    points = np.array([[-0.3, 0, -0.2], [0.3, 0.5, 0.2], [0.1, 1.4, 0.05]], dtype=np.float32)
    piece_uv = helper.wood_uvs(points, grain_axis=1)
    # A board blank with fibers along Z has exactly the same grain scale
    # as an upright piece cut from the corresponding timber coordinates.
    board_uv = helper.wood_uvs(points[:, [0, 2, 1]], grain_axis=2)
    np.testing.assert_allclose(piece_uv, board_uv)
    # Moving one scene unit along the fibers always moves half a texture,
    # independently of mesh dimensions or its tessellation.
    shifted = helper.wood_uvs(points + [0, 1, 0], grain_axis=1)
    np.testing.assert_allclose(shifted - piece_uv, np.tile([0, 0.5], (3, 1)), atol=1e-7)
    np.testing.assert_allclose(
        helper.wood_uvs(points, origin=(0.5, 0, 0.2)),
        helper.wood_uvs(points + [0.5, 0, 0.2]),
    )


def test_wood_growth_rings_follow_round_trunk_cross_sections():
    helper = _load_helper()
    angles = np.linspace(0, 2 * np.pi, 25)
    points = np.column_stack((0.4 * np.cos(angles) - 1.7, np.zeros_like(angles), 0.4 * np.sin(angles) - 0.8))
    uv = helper.wood_uvs(points)
    np.testing.assert_allclose(uv[:, 0], 0.2, atol=1e-7)
    np.testing.assert_allclose(uv[:, 1], 0, atol=1e-7)
