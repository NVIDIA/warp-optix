# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from PIL import Image
from warp_optix.pathtracing import usd_loader
from warp_optix.pathtracing.scene import Scene
from warp_optix.pathtracing.usd_loader import (
    _MIP_CHAIN_MEMORY_FACTOR,
    _ambient_light_from_custom_layer_data,
    _compact_corners,
    _decode_packed_base_opacity,
    _decode_packed_orm,
    _decode_udim,
    _fit_textures_to_budget,
    _mdl_texture_inputs,
    _normalize_mesh_uvs_for_udim_atlas,
    _rebase_missing_texture,
    _triangles_by_material,
)


def test_corner_compaction_preserves_exact_attribute_seams():
    vertices = np.zeros((5, 3), dtype=np.float32)
    normals = np.asarray(
        ((0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 0, 1)),
        dtype=np.float32,
    )
    texcoords = np.asarray(((0, 0), (0, 0), (0, 0), (1, 0), (0, 0)), dtype=np.float32)
    point_indices = np.asarray((7, 7, 7, 7, 9), dtype=np.int64)
    used = np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
    corner_remap = np.asarray((0, 1, 2, 1, 3, 4), dtype=np.int64)

    selected, triangles = _compact_corners(vertices, normals, texcoords, point_indices, used, corner_remap)

    reconstructed = np.column_stack(
        (
            point_indices[selected][triangles.reshape(-1)],
            normals[selected][triangles.reshape(-1)],
            texcoords[selected][triangles.reshape(-1)],
        )
    )
    source = np.column_stack(
        (
            point_indices[used][corner_remap],
            normals[used][corner_remap],
            texcoords[used][corner_remap],
        )
    )
    np.testing.assert_array_equal(reconstructed, source)
    assert len(selected) == 4


def test_triangle_material_groups_preserve_authored_order():
    triangles = np.asarray(
        ((0, 1, 2, 0), (3, 4, 5, 1), (6, 7, 8, 2), (9, 10, 11, 3)),
        dtype=np.int64,
    )
    face_materials = np.asarray((4, 2, 4, 2), dtype=np.int64)

    groups = _triangles_by_material(triangles, face_materials)

    assert [material_id for material_id, _ in groups] == [4, 2]
    np.testing.assert_array_equal(groups[0][1], ((0, 1, 2), (6, 7, 8)))
    np.testing.assert_array_equal(groups[1][1], ((3, 4, 5), (9, 10, 11)))


def test_texture_budget_proportionally_downscales_large_atlas():
    textures = [
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((16, 16, 4), dtype=np.uint8),
    ]

    resized = _fit_textures_to_budget(textures, max_bytes=512)

    assert sum(texture.nbytes for texture in resized) * _MIP_CHAIN_MEMORY_FACTOR <= 512
    assert resized[0].shape == resized[1].shape
    assert resized[0].shape[0] < textures[0].shape[0]
    assert all(texture.flags.c_contiguous for texture in resized)
    assert _fit_textures_to_budget(textures, max_bytes=4096) is textures


@pytest.mark.parametrize("free_memory", [8192, 1024])
def test_texture_budget_keeps_gpu_headroom(monkeypatch, free_memory):
    """Keep source textures when free VRAM leaves headroom; otherwise downscale."""
    textures = [np.zeros((16, 16, 4), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(wp, "get_device", lambda _alias: SimpleNamespace(free_memory=free_memory))
    monkeypatch.setattr(usd_loader, "_MIN_TEXTURE_MEMORY_BUDGET_BYTES", 256)
    monkeypatch.setattr(usd_loader, "_TEXTURE_VRAM_RESERVE_BYTES", 128)

    resized = _fit_textures_to_budget(textures)

    if free_memory == 8192:
        assert resized is textures
    else:
        assert sum(texture.nbytes for texture in resized) <= 512


def test_missing_texture_rebases_to_stage_local_texture_tree(tmp_path):
    stage_path = tmp_path / "variant" / "scene.usd"
    stage_path.parent.mkdir()
    stage_path.touch()
    texture = stage_path.parent / "textures" / "kit" / "albedo.1001.png"
    texture.parent.mkdir(parents=True)
    texture.touch()

    broken = tmp_path / "textures" / "kit" / "albedo.<UDIM>.png"

    assert _rebase_missing_texture(stage_path, broken) == texture.with_name("albedo.<UDIM>.png")


@pytest.mark.usd
def test_referenced_mdl_udim_resolves_from_authoring_layer(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    tile_path = texture_dir / "albedo.1001.png"
    Image.new("RGBA", (2, 2), (160, 120, 80, 255)).save(tile_path)

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    asset_path = asset_dir / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset_root.GetPrim())
    mesh = UsdGeom.Mesh.Define(asset_stage, "/Asset/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(asset_stage, "/Asset/Looks/Material")
    shader = UsdShade.Shader.Define(asset_stage, "/Asset/Looks/Material/Shader")
    shader.CreateIdAttr("CustomMdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("../textures/albedo.<UDIM>.png"))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    asset_stage.GetRootLayer().Save()

    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    referenced_root = UsdGeom.Xform.Define(stage, "/ReferencedAsset")
    referenced_root.GetPrim().GetReferences().AddReference(str(asset_path))
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(scene_path, apply_stage_units=False, max_texture_memory_bytes=1 << 20)
    assert scene.texture_count == 1
    np.testing.assert_array_equal(scene._gltf_textures[0][0, 0], (90, 48, 20, 255))


def test_omniverse_ambient_light_metadata_is_preserved():
    custom_data = {
        "renderSettings": {
            "rtx:sceneDb:ambientLightColor": (0.68, 0.68, 0.95),
            "rtx:sceneDb:ambientLightIntensity": 1.5,
        }
    }

    np.testing.assert_allclose(
        _ambient_light_from_custom_layer_data(custom_data),
        (1.02, 1.02, 1.425),
    )


class _Input:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def GetBaseName(self):
        return self._name

    def Get(self):
        return self._value

    def GetAttr(self):
        return None


class _Shader:
    def __init__(self, inputs, shader_id="", source_asset="", sub_identifier=""):
        self._inputs = [_Input(name, value) for name, value in inputs.items()]
        self._shader_id = shader_id
        self._source_asset = source_asset
        self._sub_identifier = sub_identifier

    def GetInputs(self):
        return self._inputs

    def GetInput(self, _name):
        return None

    def GetIdAttr(self):
        return _Input("id", self._shader_id)

    def GetSourceAsset(self, context):
        if context != "mdl" or not self._source_asset:
            return None
        return type("Asset", (), {"resolvedPath": "", "path": self._source_asset})()

    def GetSourceAssetSubIdentifier(self, context):
        return self._sub_identifier if context == "mdl" else ""


def test_collected_mdl_source_exposes_pbr_textures(monkeypatch, tmp_path):
    texture_paths = {}
    for name in ("base.png", "normal.png", "mask.png"):
        texture_path = tmp_path / name
        texture_path.touch()
        texture_paths[name] = texture_path
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        export material Example(
            uniform texture_2d Num1_BaseColor = texture_2d("base.png"),
            uniform texture_2d Num2_Normal = texture_2d("normal.png"),
            uniform texture_2d Num3_Mask = texture_2d("mask.png"))
        = material();
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path))
    inputs = _mdl_texture_inputs(shader)

    assert inputs == {
        "diffuse_texture": texture_paths["base.png"],
        "normalmap_texture": texture_paths["normal.png"],
        "ORM_texture": texture_paths["mask.png"],
    }

    requests = []
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    material = type("Material", (), {"GetInputs": lambda self: []})()
    result = usd_loader._material_to_pbr(
        material,
        object(),
        lambda path, srgb: requests.append((path, srgb)) or len(requests) - 1,
    )

    assert result["base_color_texture"]["index"] == 0
    assert result["normal_texture"]["index"] == 2
    assert result["metallic_roughness_texture"]["index"] == 1
    assert requests == [
        (texture_paths["base.png"], True),
        (texture_paths["mask.png"], False),
        (texture_paths["normal.png"], False),
    ]


def test_collected_mdl_source_infers_common_texture_parameter_names(tmp_path):
    texture_names = (
        "albedo.png",
        "normal.png",
        "roughness.png",
        "metalness.png",
        "emission.png",
        "opacity.png",
    )
    for name in texture_names:
        (tmp_path / name).touch()
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        export material Example(
            uniform texture_2d AlbedoMap = texture_2d("albedo.png"),



            uniform texture_2d SurfaceNormalTexture = texture_2d("normal.png"),
            uniform texture_2d SurfaceRoughnessMap = texture_2d("roughness.png"),
            uniform texture_2d MetalnessTexture = texture_2d("metalness.png"),
            uniform texture_2d EmissionMap = texture_2d("emission.png"),
            uniform texture_2d OpacityMap = texture_2d("opacity.png"))
        = material();
        """
    )

    assert _mdl_texture_inputs(_Shader({}, source_asset=str(mdl_path))) == {
        "diffuse_texture": tmp_path / "albedo.png",
        "normalmap_texture": tmp_path / "normal.png",
        "reflectionroughness_texture": tmp_path / "roughness.png",
        "metallic_texture": tmp_path / "metalness.png",
        "emissive_texture": tmp_path / "emission.png",
    }


def test_collected_mdl_source_preserves_literal_omnipbr_defaults(monkeypatch, tmp_path):
    mdl_path = tmp_path / "handles.mdl"
    mdl_path.write_text(
        """
        export material handles(*) = ::OmniPBR::OmniPBR(
            diffuse_color_constant: color(0.8f, 0.7f, 0.6f),
            reflection_roughness_constant: 0.05f,
            metallic_constant: 1.f,
            specular_level: 0.5f,
            enable_ORM_texture: false,
            texture_translate: float2(0.1f, 0.2f),
            texture_scale: float2(2.f));
        """
    )
    shader = _Shader(
        {"metallic_constant": 0.75},
        source_asset=str(mdl_path),
        sub_identifier="handles",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: -1)

    assert result["base_color"] == (0.8, 0.7, 0.6, 1.0)
    assert result["roughness"] == 0.05
    assert result["metallic"] == 0.75
    assert result["specular"] == 0.5
    assert result["base_color_texture"]["transform"] == {
        "scale": (2.0, 2.0),
        "offset": (0.1, 0.2),
        "rotation": 0.0,
    }


def test_omnisurface_lite_input_aliases_are_preserved(monkeypatch):
    shader = _Shader(
        {
            "diffuse_reflection_color": (0.48, 0.09, 0.09),
            "specular_reflection_roughness": 0.65,
            "specular_reflection_weight": 0.2,
        },
        source_asset="OmniSurfaceLite.mdl",
        sub_identifier="OmniSurfaceLite",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: -1)

    assert result["base_color"] == (0.48, 0.09, 0.09, 1.0)
    assert result["roughness"] == 0.65
    assert result["specular"] == 0.2


def test_authored_mdl_inputs_infer_common_texture_parameter_names(monkeypatch, tmp_path):
    albedo = tmp_path / "albedo.png"
    albedo.touch()
    shader = _Shader({"Base_Color_Map": str(albedo)}, shader_id="CustomMdl")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    requests = []

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: requests.append((path, srgb)) or 0)

    assert result["base_color_texture"]["index"] == 0
    assert requests == [(albedo, True)]


def test_separate_roughness_map_is_packed_with_metallic_constant(tmp_path):
    roughness_path = tmp_path / "roughness.png"
    Image.fromarray(np.asarray([[[64, 64, 64, 255]]], dtype=np.uint8), mode="RGBA").save(roughness_path)
    spec = (
        "packed_orm",
        Path(roughness_path),
        None,
        0.1,
        1.0,
        0.8,
        1.0,
    )

    packed = _decode_packed_orm(spec)

    assert packed.shape == (1, 1, 4)
    assert packed.dtype == np.uint8
    np.testing.assert_allclose(packed[0, 0, 0], 255)
    np.testing.assert_allclose(packed[0, 0, 1], round((0.1 * 0.2 + (64.0 / 255.0) * 0.8) * 255.0))
    np.testing.assert_allclose(packed[0, 0, 2], 255)


def test_omnipbr_separate_maps_select_packed_gltf_workflow(monkeypatch, tmp_path):
    roughness_path = tmp_path / "roughness.dds"
    roughness_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(roughness_path), "path": ""})()
    shader = _Shader(
        {
            "reflectionroughness_texture": asset,
            "reflection_roughness_constant": 0.084,
            "reflection_roughness_texture_influence": 0.806,
            "metallic_constant": 1.0,
        }
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    requests = []

    material = type("Material", (), {"GetInputs": lambda self: []})()
    result = usd_loader._material_to_pbr(
        material,
        object(),
        lambda path, srgb: -1,
        lambda *args: requests.append(args) or 7,
    )

    assert result["roughness"] == 1.0
    assert result["metallic"] == 1.0
    assert result["metallic_roughness_texture"]["index"] == 7
    assert requests == [(roughness_path, None, 0.084, 1.0, 0.806, 1.0)]


def test_fresnel_emissive_mdl_inputs_are_preserved(monkeypatch):
    shader = _Shader(
        {
            "albedo_color": (0.001, 0.012, 0.0),
            "emissive_color_normal": (0.302, 1.0, 0.0),
            "emissive_intensity": 7.0,
            "reflection_roughness": 0.05,
        }
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: -1)

    np.testing.assert_allclose(result["base_color"][:3], (0.001, 0.012, 0.0))
    np.testing.assert_allclose(result["emissive_factor"], (2.114, 7.0, 0.0))
    assert result["roughness"] == 0.05


def test_omniglass_source_asset_selects_transmission_without_authored_inputs(
    monkeypatch,
):
    shader = _Shader(
        {},
        shader_id="mdlMaterial",
        source_asset="OmniGlass.mdl",
        sub_identifier="OmniGlass",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: -1)

    assert result["transmission"] == 1.0
    assert result["roughness"] == 0.0
    assert result["thickness"] == 1.0


def test_collected_omni_ue4_translucent_wrapper_preserves_glass(monkeypatch, tmp_path):
    """Convert UE translucent glass to a stable dielectric interface."""
    for name in ("color.png", "normal.png", "mask.png"):
        (tmp_path / name).touch()
    mdl_path = tmp_path / "custom_material.mdl"
    mdl_path.write_text(
        """
        export material custom_material(
            uniform texture_2d Num1_BaseColor = texture_2d("color.png"),
            uniform texture_2d Num2_Normal = texture_2d("normal.png"),
            uniform texture_2d Num3_Mask = texture_2d("mask.png"),
            float Opacity_low = 0.2,
            float Opacity_hi = 0.35,
            float Opacity_Fallof = 2.0,
            float Opacity_multiplayer = 1.0,
            uniform float Refraction_hi = 1.1)
        = ::OmniUe4Translucent(
            base_color: color(1.0),
            opacity: Opacity_low * Opacity_multiplayer,
            refraction: Refraction_hi);
        """
    )
    shader = _Shader(
        {},
        source_asset=str(mdl_path),
        sub_identifier="custom_material",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    requests = []

    result = usd_loader._material_to_pbr(
        None,
        object(),
        lambda path, srgb: requests.append((path, srgb)) or len(requests) - 1,
    )

    assert result["base_color"][3] == pytest.approx(1.0)
    assert result["opacity_fresnel"] is None
    assert result["transmission"] == 1.0
    assert result["transmission_color"] == pytest.approx((1.0, 1.0, 1.0))
    assert result["metallic"] == 0.0
    assert result["ior"] == pytest.approx(1.1)
    assert result["thickness"] == 0.0
    assert result["alpha_mode"] == "OPAQUE"
    assert result["base_color_texture"]["index"] == 0
    assert result["metallic_roughness_texture"]["index"] == 1
    assert result["normal_texture"]["index"] == 2
    assert requests == [
        (tmp_path / "color.png", True),
        (tmp_path / "mask.png", False),
        (tmp_path / "normal.png", False),
    ]


def test_collected_omni_ue4_translucent_wool_uses_surface_coverage(monkeypatch, tmp_path):
    """Interpret the authored wool shell as coverage instead of glass."""
    for name in ("color.png", "normal.png", "mask.png"):
        (tmp_path / name).touch()
    mdl_path = tmp_path / "wool_glass.mdl"
    mdl_path.write_text(
        """
        export material wool_glass(
            uniform texture_2d Num1_BaseColor = texture_2d("color.png"),
            uniform texture_2d Num2_Normal = texture_2d("normal.png"),
            uniform texture_2d Num3_Mask = texture_2d("mask.png", ::tex::gamma_linear),
            float Opacity_low = 0.0,
            float Opacity_hi = 1.0,
            float Opacity_Fallof = 2.0,
            float Opacity_multiplayer = 1.0,
            uniform float Refraction_hi = 1.1)
        = ::OmniUe4Translucent(
            base_color: color(1.0),
            opacity: Opacity_low * Opacity_multiplayer,
            refraction: Refraction_hi);
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path), sub_identifier="wool_glass")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: 1)

    assert result["opacity_fresnel"] == pytest.approx((0.0, 1.0, 2.0))
    assert result["transmission"] == 0.0
    assert result["transmission_color"] is None
    assert result["ior"] == pytest.approx(1.1)
    assert result["alpha_mode"] == "BLEND"


def test_collected_translucent_wrapper_packs_authored_opacity(monkeypatch, tmp_path):
    """Pack an authored MDL opacity graph into base-color alpha."""
    for name in ("color.png", "normal.png", "mask.png", "opacity.png"):
        (tmp_path / name).touch()
    mdl_path = tmp_path / "custom_material.mdl"
    mdl_path.write_text(
        """
        export material custom_material(
            uniform texture_2d Num1_BaseColor = texture_2d("color.png", ::tex::gamma_srgb),
            uniform texture_2d Num2_Normal = texture_2d("normal.png", ::tex::gamma_linear),
            uniform texture_2d Num3_Mask = texture_2d("mask.png", ::tex::gamma_linear),
            uniform texture_2d Num4_Opacity = texture_2d("opacity.png", ::tex::gamma_srgb),
            float Opacity_Tex_rougness_Contrast = 0.25,
            float Opacity_Tex_roughness_Amount = 0.75,
            float Opacity_Tex_roughness_multi = 0.5,
            float Opacity_multiplayer = 0.8)
        = ::OmniUe4Translucent(opacity: 1.0);
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path), sub_identifier="custom_material")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    packed_requests = []

    result = usd_loader._material_to_pbr(
        None,
        object(),
        lambda path, srgb: 1,
        packed_opacity_index=lambda *args: packed_requests.append(args) or 9,
    )

    assert result["base_color_texture"]["index"] == 9
    assert packed_requests == [
        (
            tmp_path / "color.png",
            tmp_path / "opacity.png",
            tmp_path / "mask.png",
            True,
            False,
            0.25,
            0.75,
            0.5,
            0.8,
        )
    ]


def test_packed_base_opacity_evaluates_mdl_graph(tmp_path):
    """Evaluate the collected OmniUe4 opacity graph into base alpha."""
    Image.new("RGBA", (1, 1), (128, 64, 32, 255)).save(tmp_path / "base.png")
    Image.new("RGBA", (1, 1), (128, 0, 0, 255)).save(tmp_path / "opacity.png")
    Image.new("RGBA", (1, 1), (0, 128, 0, 255)).save(tmp_path / "orm.png")
    spec = (
        "packed_base_opacity",
        tmp_path / "base.png",
        tmp_path / "opacity.png",
        tmp_path / "orm.png",
        False,
        False,
        0.0,
        1.0,
        0.5,
        1.0,
    )

    packed = _decode_packed_base_opacity(spec)

    np.testing.assert_array_equal(packed[0, 0, :3], (128, 64, 32))
    assert packed[0, 0, 3] == pytest.approx(32, abs=1)


def test_collected_omni_ue4_base_preserves_surface_controls(monkeypatch, tmp_path):
    """Preserve scalar lobes and AO strength from collected OmniUe4Base MDL."""
    for name in ("color.png", "normal.png", "mask.png"):
        (tmp_path / name).touch()
    mdl_path = tmp_path / "custom_material.mdl"
    mdl_path.write_text(
        """
        export material custom_material(
            uniform texture_2d Num1_BaseColor = texture_2d("color.png"),
            uniform texture_2d Num2_Normal = texture_2d("normal.png"),
            uniform texture_2d Num3_Mask = texture_2d("mask.png"))
        = let {
            float AOamount = 0.65;
            float Specular_mdl = 0.5;
            float ClearCoat_mdl = 1.0;
            float ClearCoatRoughness_mdl = 0.1;
        } in ::OmniUe4Base(
            specular: Specular_mdl,
            clear_coat: ClearCoat_mdl,
            clear_coat_roughness: ClearCoatRoughness_mdl);
        """
    )
    shader = _Shader(
        {},
        source_asset=str(mdl_path),
        sub_identifier="custom_material",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(
        None,
        object(),
        lambda path, srgb: 0,
    )

    assert result["specular"] == pytest.approx(0.5)
    assert result["clearcoat"] == pytest.approx(1.0)
    assert result["clearcoat_roughness"] == pytest.approx(0.1)
    assert result["occlusion_strength"] == pytest.approx(0.65)
    assert result["normal_texture"]["scale"] == pytest.approx((1.0, -1.0))


def test_collected_mdl_honors_authored_srgb_orm(monkeypatch, tmp_path):
    """Decode an ORM texture as sRGB when its MDL declaration requires it."""
    for name in ("color.png", "normal.png", "mask.png"):
        (tmp_path / name).touch()
    mdl_path = tmp_path / "custom_material.mdl"
    mdl_path.write_text(
        """
        export material custom_material(
            uniform texture_2d Num1_BaseColor = texture_2d("color.png", ::tex::gamma_srgb),
            uniform texture_2d Num2_Normal = texture_2d("normal.png", ::tex::gamma_linear),
            uniform texture_2d Num3_Mask = texture_2d("mask.png", ::tex::gamma_srgb))
        = ::OmniUe4Base();
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path), sub_identifier="custom_material")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    requests = []

    usd_loader._material_to_pbr(
        None,
        object(),
        lambda path, srgb: requests.append((path, srgb)) or len(requests) - 1,
    )

    assert (tmp_path / "mask.png", True) in requests
    assert (tmp_path / "normal.png", False) in requests


def test_collected_mdl_ignores_unconnected_clearcoat_locals(monkeypatch, tmp_path):
    """Ignore generated clearcoat locals that do not feed the surface."""
    mdl_path = tmp_path / "wood.mdl"
    mdl_path.write_text(
        """
        export material wood() = let {
            float ClearCoat_mdl = 1.0;
            float ClearCoatRoughness_mdl = 0.1;
        } in ::OmniUe4Base(base_color: color(0.4), roughness: 0.7);
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path), sub_identifier="wood")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: -1)

    assert result["clearcoat"] == 0.0
    assert result["clearcoat_roughness"] == pytest.approx(0.1)


def test_collected_ue4_emissive_texture_uses_isaac_radiance_scale(monkeypatch, tmp_path):
    """Match Isaac Sim's OmniUe4Base emission multiplier in renderer units."""
    (tmp_path / "emissive.png").touch()
    mdl_path = tmp_path / "emissive.mdl"
    mdl_path.write_text(
        """
        export material emissive(
            uniform texture_2d Num4_Emissive = texture_2d(
                "emissive.png", ::tex::gamma_srgb),
            float EmissiveMultiplayer = 1.5)
        = ::OmniUe4Base(emissive_color: EmissiveColor_mdl);
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path), sub_identifier="emissive")
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: 3)

    assert result["emissive_texture"]["index"] == 3
    assert result["emissive_factor"] == pytest.approx((0.048, 0.048, 0.048))


def test_omniglass_uses_authored_color_texture(monkeypatch, tmp_path):
    """Use OmniGlass color textures as sRGB base color inputs."""
    color_path = tmp_path / "glass_color.png"
    color_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(color_path), "path": ""})()
    shader = _Shader(
        {"glass_color": (0.5, 0.75, 1.0), "glass_color_texture": asset},
        source_asset="OmniGlass.mdl",
        sub_identifier="OmniGlass",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    requests = []

    result = usd_loader._material_to_pbr(
        None,
        object(),
        lambda path, srgb: requests.append((path, srgb)) or 4,
    )

    assert result["base_color"] == (0.5, 0.75, 1.0, 1.0)
    assert result["base_color_texture"]["index"] == 4
    assert requests == [(color_path, True)]


def test_omnipbr_preserves_normal_axis_flips(monkeypatch, tmp_path):
    """Apply authored OmniPBR tangent-axis flips to normal-map channels."""
    normal_path = tmp_path / "normal.png"
    normal_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(normal_path), "path": ""})()
    shader = _Shader(
        {
            "normalmap_texture": asset,
            "bump_factor": 0.75,
            "flip_tangent_u": False,
            "flip_tangent_v": True,
        }
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: 2)

    assert result["normal_texture"]["index"] == 2
    assert result["normal_texture"]["scale"] == (0.75, -0.75)


def test_omnipbr_uses_its_authored_normal_green_channel_default(monkeypatch, tmp_path):
    """Use OmniPBR's default flipped V tangent when no override is authored."""
    normal_path = tmp_path / "normal.png"
    normal_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(normal_path), "path": ""})()
    shader = _Shader(
        {"normalmap_texture": asset},
        source_asset="OmniPBR.mdl",
        sub_identifier="OmniPBR",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)

    result = usd_loader._material_to_pbr(None, object(), lambda path, srgb: 2)

    assert result["normal_texture"]["scale"] == (1.0, -1.0)


def test_omnipbr_honors_disabled_orm_and_surface_controls(monkeypatch, tmp_path):
    orm_path = tmp_path / "orm.png"
    orm_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(orm_path), "path": ""})()
    shader = _Shader(
        {
            "ORM_texture": asset,
            "enable_ORM_texture": False,
            "specular_level": 0.4,
            "albedo_add": 0.2,
            "albedo_desaturation": 0.3,
            "texture_scale": (2.0, 3.0),
            "texture_translate": (0.1, 0.25),
            "texture_rotate": 90.0,
        },
        source_asset="OmniPBR.mdl",
        sub_identifier="OmniPBR",
    )
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: 3)

    assert result["metallic_roughness_texture"]["index"] == -1
    assert result["occlusion_texture"]["index"] == -1
    assert result["specular"] == 0.4
    assert result["base_color_add"] == 0.2
    assert result["base_color_desaturation"] == 0.3
    assert result["base_color_texture"]["transform"]["scale"] == (2.0, 3.0)
    assert result["base_color_texture"]["transform"]["offset"] == (0.1, 0.25)
    assert result["base_color_texture"]["transform"]["rotation"] == pytest.approx(np.pi / 2.0)


def test_udim_tiles_decode_to_horizontal_atlas_with_per_tile_size_limit(tmp_path):
    paths = []
    for tile, color in ((1001, (255, 0, 0, 255)), (1002, (0, 255, 0, 255)), (1003, (0, 0, 255, 255))):
        tile_path = tmp_path / f"color.{tile}.png"
        Image.fromarray(np.full((8, 8, 4), color, dtype=np.uint8), mode="RGBA").save(tile_path)
        paths.append((tile, tile_path))

    atlas = _decode_udim(tuple(paths))
    assert atlas.shape == (8, 24, 4)
    assert atlas.dtype == np.uint8
    np.testing.assert_array_equal(atlas[0, (0, 8, 16), :3], np.eye(3, dtype=np.uint8) * 255)
    assert _decode_udim(tuple(paths), max_size=4).shape == (4, 12, 4)


@pytest.mark.parametrize(
    "texcoords",
    [
        ((0.0, 0.25), (0.5, 0.75), (1.0, 1.0)),  # local UVs map to the first atlas cell
        ((0.0, 0.0), (1.5, 0.5), (2.9, 1.0)),  # multi-tile UVs use the material atlas dimensions
    ],
)
def test_udim_mesh_uvs_scale_to_atlas(texcoords):
    texcoords = np.asarray(texcoords, dtype=np.float32)
    np.testing.assert_allclose(_normalize_mesh_uvs_for_udim_atlas(texcoords, 4, 1), texcoords / (4.0, 1.0))
