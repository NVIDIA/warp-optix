# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from warp_optix.pathtracing.scene import Scene
from warp_optix.pathtracing.usd_loader import _bound_material, _mesh_arrays

pytestmark = pytest.mark.usd
pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


def _write_transform_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Xform.Define(stage, "/body")
    body.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, "/body/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()


def test_usd_scene_retains_paths_and_local_geometry(tmp_path):
    path = tmp_path / "transform.usda"
    _write_transform_stage(path)
    scene = Scene(None)

    assert scene.load_from_usd(path, apply_stage_units=False)
    usd_scene = scene.usd_scene
    body = usd_scene.require_transform("/body")
    mesh = usd_scene.require_transform("/body/mesh")

    assert usd_scene.get_transform("/missing") is None
    assert usd_scene.transform_count == 2
    assert usd_scene.source_path == str(path.resolve())
    assert usd_scene.get_prim("/body").IsValid()
    np.testing.assert_allclose(scene._meshes[0].vertices[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(usd_scene.get_world_transform(mesh)[:3, 3], (2.0, 0.0, 0.0))
    assert usd_scene.instance_ids(body) == ()
    assert usd_scene.instance_ids(mesh) == (0,)

    usd_scene.set_visible(False)
    assert not scene._instance_visibility_cache[0]
    usd_scene.set_visible(True)
    assert scene._instance_visibility_cache[0]

    updated = np.eye(4, dtype=np.float32)
    updated[:3, 3] = (4.0, 5.0, 6.0)
    usd_scene.update_local_transform(body, updated)
    np.testing.assert_allclose(usd_scene.get_world_transform(mesh)[:3, 3], (4.0, 5.0, 6.0))
    np.testing.assert_allclose(scene._instance_transform_cache[0, :3, 3], (4.0, 5.0, 6.0))

    scene.clear()
    with pytest.raises(RuntimeError, match="invalidated"):
        usd_scene.get_world_transform(mesh)


def test_direct_material_relationship_without_applied_api_resolves(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Material")
    subset = UsdGeom.Subset.Define(stage, "/Subset")
    subset.GetPrim().CreateRelationship("material:binding").SetTargets([material.GetPath()])

    assert not subset.GetPrim().HasAPI(UsdShade.MaterialBindingAPI)
    assert _bound_material(subset.GetPrim(), UsdShade).GetPath() == material.GetPath()


def test_stronger_ancestor_overrides_unapplied_referenced_subset_binding(tmp_path):
    asset_path = tmp_path / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset_root.GetPrim())
    UsdGeom.Mesh.Define(asset_stage, "/Asset/Mesh")
    subset = UsdGeom.Subset.Define(asset_stage, "/Asset/Mesh/subset")
    original = UsdShade.Material.Define(asset_stage, "/Asset/Looks/Original")
    subset.GetPrim().CreateRelationship("material:binding").SetTargets([original.GetPath()])
    asset_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    referenced_root = UsdGeom.Xform.Define(stage, "/ReferencedAsset")
    referenced_root.GetPrim().GetReferences().AddReference(str(asset_path))
    override = UsdShade.Material.Define(stage, "/Looks/Override")
    UsdShade.MaterialBindingAPI.Apply(referenced_root.GetPrim()).Bind(
        override, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    referenced_subset = stage.GetPrimAtPath("/ReferencedAsset/Mesh/subset")

    assert not referenced_subset.HasAPI(UsdShade.MaterialBindingAPI)
    assert _bound_material(referenced_subset, UsdShade).GetPath() == override.GetPath()


def test_mesh_arrays_selects_unreal_st_zero_uv_set():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    expected = [Gf.Vec2f(0.1, 0.2), Gf.Vec2f(0.3, 0.4), Gf.Vec2f(0.5, 0.6)]
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st_0", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        expected
    )

    arrays = _mesh_arrays(mesh, UsdGeom)

    np.testing.assert_allclose(arrays[2], expected)


@pytest.mark.parametrize(
    ("name", "type_name", "values"),
    [
        (
            "uv0",
            Sdf.ValueTypeNames.Float2Array,
            [Gf.Vec2f(0.1, 0.2), Gf.Vec2f(0.3, 0.4), Gf.Vec2f(0.5, 0.6)],
        ),
        (
            "map1",
            Sdf.ValueTypeNames.TexCoord2dArray,
            [Gf.Vec2d(0.1, 0.2), Gf.Vec2d(0.3, 0.4), Gf.Vec2d(0.5, 0.6)],
        ),
    ],
)
def test_mesh_arrays_accepts_common_uv_names_and_types(name, type_name, values):
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(name, type_name, UsdGeom.Tokens.faceVarying).Set(values)

    np.testing.assert_allclose(_mesh_arrays(mesh, UsdGeom)[2], values)


def test_sphere_light_imports_as_five_centimeter_analytic_light(tmp_path):
    path = tmp_path / "sphere_light.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    light = UsdLux.SphereLight.Define(stage, "/Light")
    light.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 300.0))
    light.CreateColorAttr(Gf.Vec3f(0.5, 0.25, 1.0))
    light.CreateIntensityAttr(10.0)
    light.CreateExposureAttr(1.0)
    light.CreateRadiusAttr(5.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, convert_up_axis=False, load_usd_lights=True)
    assert scene.mesh_count == 1
    assert len(scene._instances) == 1
    assert scene.light_count == 1
    position, radius, color, intensity = scene._sphere_lights[0]
    np.testing.assert_allclose(position, (1.0, 2.0, 3.0))
    assert radius == pytest.approx(0.05)
    np.testing.assert_allclose(color, (0.5, 0.25, 1.0))
    assert intensity == pytest.approx(20.0 / 80000.0)
    assert scene.usd_scene.instance_ids(scene.usd_scene.require_transform("/Light")) == ()

    disabled_scene = Scene(None)
    assert disabled_scene.load_from_usd(path, load_usd_lights=False)
    assert disabled_scene.mesh_count == 1
    assert disabled_scene.light_count == 0


def test_sphere_light_proxy_preserves_authored_power(tmp_path):
    """Use authored radius by default and preserve power for explicit proxies."""
    path = tmp_path / "sphere_light_power.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    light = UsdLux.SphereLight.Define(stage, "/Light")
    light.CreateIntensityAttr(100.0)
    light.CreateRadiusAttr(4.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    authored_scene = Scene(None)
    assert authored_scene.load_from_usd(path, convert_up_axis=False, load_usd_lights=True)
    assert authored_scene._sphere_lights[0][1] == pytest.approx(0.04)
    assert authored_scene._sphere_lights[0][3] == pytest.approx(100.0 / 80000.0)

    scene = Scene(None)
    assert scene.load_from_usd(
        path,
        convert_up_axis=False,
        load_usd_lights=True,
        usd_light_radius=0.05,
    )
    assert scene._sphere_lights[0][3] == pytest.approx(64.0 / 80000.0)

    light.CreateNormalizeAttr(True)
    stage.GetRootLayer().Save()
    normalized_scene = Scene(None)
    assert normalized_scene.load_from_usd(path, convert_up_axis=False, load_usd_lights=True)
    assert normalized_scene._sphere_lights[0][1] == pytest.approx(0.04)
    assert normalized_scene._sphere_lights[0][3] == pytest.approx(100.0 / (80000.0 * 4.0 * np.pi * 0.04**2))


def test_reflected_face_varying_mesh_keeps_world_shading_frame_aligned(tmp_path):
    path = tmp_path / "reflected.usda"
    stage = Usd.Stage.CreateNew(str(path))
    reflected = UsdGeom.Xform.Define(stage, "/reflected")
    reflected.AddScaleOp().Set(Gf.Vec3f(-1.0, 1.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/reflected/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)]
    )
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    loaded = scene._meshes[0]
    world = scene._instance_transform_cache[0]
    world_positions = (
        np.concatenate(
            (loaded.vertices, np.ones((len(loaded.vertices), 1), dtype=np.float32)),
            axis=1,
        )
        @ world.T
    )[:, :3]
    tri = loaded.indices[0]
    geometric = np.cross(
        world_positions[tri[1]] - world_positions[tri[0]],
        world_positions[tri[2]] - world_positions[tri[0]],
    )
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    shading = loaded.normals[tri[0]] @ normal_matrix.T

    assert np.dot(geometric, shading) > 0.0
    np.testing.assert_array_equal(tri, (0, 1, 2))


def test_authored_normals_are_preserved_in_robust_two_sided_mode(tmp_path):
    path = tmp_path / "authored_normals.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, -1.0)] * 3)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)]
    )
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    np.testing.assert_allclose(scene._meshes[0].normals, ((0.0, 0.0, -1.0),) * 3)
    np.testing.assert_array_equal(scene._meshes[0].indices[0], (0, 1, 2))
    assert scene._instances[0].double_sided

    strict_scene = Scene(None)
    assert strict_scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False, strict_sidedness=True)
    assert not strict_scene._instances[0].double_sided


def test_left_handed_winding_and_double_sided_are_converted_for_optix(tmp_path):
    path = tmp_path / "left_handed.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mesh.CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    np.testing.assert_array_equal(scene._meshes[0].indices[0], (0, 2, 1))
    np.testing.assert_allclose(scene._meshes[0].normals, ((0.0, 0.0, -1.0),) * 3)
    assert scene._instances[0].double_sided
