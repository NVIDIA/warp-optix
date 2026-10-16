# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Modified Python/NumPy adaptation of glTF material and texture structures:
# https://github.com/nvpro-samples/nvpro_core/blob/bc19d6ac3ef62938d0ea0e099735878457ce1b6e/nvvkhl/shaders/dh_scn_desc.h
# Original notices: licenses/vk_denoise_dlssrr-NOTICE.txt and
# licenses/nvpro_core-NOTICE.txt.

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

"""
Material management for path tracing viewer.
Provides PBR material creation.
"""

from __future__ import annotations

import numpy as np

# Alpha modes
ALPHA_OPAQUE = 0
ALPHA_MASK = 1
ALPHA_BLEND = 2


def _create_texture_info_dtype():
    """Create numpy dtype for GpuTextureInfo (32 bytes)."""
    return np.dtype(
        [
            ("uvTransform00", np.float32),
            ("uvTransform01", np.float32),
            ("uvTransform02", np.float32),
            ("uvTransform10", np.float32),
            ("uvTransform11", np.float32),
            ("uvTransform12", np.float32),
            ("index", np.int32),
            ("texCoord", np.int32),
        ]
    )


def _create_material_dtype():
    """Create numpy dtype for GpuMaterial structure."""
    tex_info_dt = _create_texture_info_dtype()

    return np.dtype(
        [
            # Base PBR properties
            ("pbrBaseColorFactor", np.float32, (4,)),
            ("emissiveFactor", np.float32, (3,)),
            ("normalTextureScale", np.float32, (2,)),
            ("pbrRoughnessFactor", np.float32),
            ("pbrMetallicFactor", np.float32),
            ("uSubdiv", np.float32),
            ("vSubdiv", np.float32),
            ("baseColorScale", np.float32),
            ("alphaMode", np.int32),
            ("alphaCutoff", np.float32),
            ("opacityFresnelLow", np.float32),
            ("opacityFresnelHigh", np.float32),
            ("opacityFresnelFalloff", np.float32),
            # Transmission/Volume
            ("transmissionFactor", np.float32),
            ("transmissionColorFactor", np.float32, (3,)),
            ("ior", np.float32),
            ("attenuationColor", np.float32, (3,)),
            ("thicknessFactor", np.float32),
            ("attenuationDistance", np.float32),
            # Clearcoat
            ("clearcoatFactor", np.float32),
            ("clearcoatRoughness", np.float32),
            # Specular
            ("specularColorFactor", np.float32, (3,)),
            ("specularFactor", np.float32),
            # Misc
            ("unlit", np.int32),
            # Iridescence
            ("iridescenceFactor", np.float32),
            ("iridescenceThicknessMaximum", np.float32),
            ("iridescenceThicknessMinimum", np.float32),
            ("iridescenceIor", np.float32),
            # Anisotropy
            ("anisotropyStrength", np.float32),
            ("anisotropyRotation", np.float32, (2,)),
            # Sheen
            ("sheenRoughnessFactor", np.float32),
            ("sheenColorFactor", np.float32, (3,)),
            # Occlusion & Dispersion
            ("occlusionStrength", np.float32),
            ("dispersion", np.float32),
            # Specular-Glossiness workflow
            ("pbrDiffuseFactor", np.float32, (4,)),
            ("pbrSpecularFactor", np.float32, (3,)),
            ("usePbrSpecularGlossiness", np.int32),
            ("pbrGlossinessFactor", np.float32),
            # Diffuse transmission
            ("diffuseTransmissionColor", np.float32, (3,)),
            ("diffuseTransmissionFactor", np.float32),
            # Padding
            ("pad", np.int32),
            # Texture infos (21 textures)
            ("pbrBaseColorTexture", tex_info_dt),
            ("normalTexture", tex_info_dt),
            ("pbrMetallicRoughnessTexture", tex_info_dt),
            ("emissiveTexture", tex_info_dt),
            ("transmissionTexture", tex_info_dt),
            ("thicknessTexture", tex_info_dt),
            ("clearcoatTexture", tex_info_dt),
            ("clearcoatRoughnessTexture", tex_info_dt),
            ("clearcoatNormalTexture", tex_info_dt),
            ("specularTexture", tex_info_dt),
            ("specularColorTexture", tex_info_dt),
            ("iridescenceTexture", tex_info_dt),
            ("iridescenceThicknessTexture", tex_info_dt),
            ("anisotropyTexture", tex_info_dt),
            ("sheenColorTexture", tex_info_dt),
            ("sheenRoughnessTexture", tex_info_dt),
            ("occlusionTexture", tex_info_dt),
            ("pbrDiffuseTexture", tex_info_dt),
            ("pbrSpecularGlossinessTexture", tex_info_dt),
            ("diffuseTransmissionTexture", tex_info_dt),
            ("diffuseTransmissionColorTexture", tex_info_dt),
        ]
    )


def _default_texture_info():
    """Create a default (no texture) texture info."""
    tex_dt = _create_texture_info_dtype()
    info = np.zeros(1, dtype=tex_dt)[0]
    info["uvTransform00"] = 1.0
    info["uvTransform11"] = 1.0
    info["index"] = -1
    info["texCoord"] = 0
    return info


def _texture_info_from_index(index: int, uv_set: int = 0):
    """Create a texture info from a texture index."""
    tex_dt = _create_texture_info_dtype()
    info = np.zeros(1, dtype=tex_dt)[0]
    info["uvTransform00"] = 1.0
    info["uvTransform11"] = 1.0
    info["index"] = index
    info["texCoord"] = uv_set
    return info


def _texture_info_from_gltf(info_dict: dict | None):
    """Create texture info including KHR_texture_transform mapping."""
    if not info_dict:
        return _default_texture_info()
    tex_dt = _create_texture_info_dtype()
    info = np.zeros(1, dtype=tex_dt)[0]
    info["uvTransform00"] = 1.0
    info["uvTransform11"] = 1.0
    info["index"] = int(info_dict.get("index", -1))
    info["texCoord"] = int(info_dict.get("texCoord", 0))
    xform = info_dict.get("transform")
    if isinstance(xform, dict):
        scale = xform.get("scale", [1.0, 1.0])
        offset = xform.get("offset", [0.0, 0.0])
        rotation = float(xform.get("rotation", 0.0))
        c = float(np.cos(rotation))
        s = float(np.sin(rotation))
        sx = float(scale[0]) if len(scale) > 0 else 1.0
        sy = float(scale[1]) if len(scale) > 1 else 1.0
        ox = float(offset[0]) if len(offset) > 0 else 0.0
        oy = float(offset[1]) if len(offset) > 1 else 0.0
        info["uvTransform00"] = c * sx
        info["uvTransform01"] = -s * sy
        info["uvTransform02"] = ox
        info["uvTransform10"] = s * sx
        info["uvTransform11"] = c * sy
        info["uvTransform12"] = oy
    return info


class MaterialManager:
    """Manages the host-side PBR material table."""

    def __init__(self):
        self._materials = []
        self._material_dtype = _create_material_dtype()

    @property
    def count(self) -> int:
        """Get the number of materials."""
        return len(self._materials)

    def get_material_entries(self):
        """Return the underlying material entry list for packing helpers."""
        return self._materials

    def add_default(self) -> int:
        """Add a default PBR material and return its index."""
        mat = self._create_default_material()
        # Pure-white diffuse reflectance is unrealistic and needlessly noisy.
        # Match Newton's neutral 0.7 display-sRGB ground color in linear space.
        mat["pbrBaseColorFactor"] = (0.4479884, 0.4479884, 0.4479884, 1.0)
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_diffuse(self, color: tuple, roughness: float = 0.9) -> int:
        """Add a simple diffuse material."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (*color, 1.0)
        mat["pbrMetallicFactor"] = 0.0
        mat["pbrRoughnessFactor"] = roughness
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_metal(self, color: tuple, roughness: float = 0.1) -> int:
        """Add a metallic material."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (*color, 1.0)
        mat["pbrMetallicFactor"] = 1.0
        mat["pbrRoughnessFactor"] = roughness
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_emissive(self, color: tuple, intensity: float = 10.0) -> int:
        """Add an emissive light material."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (1.0, 1.0, 1.0, 1.0)
        mat["emissiveFactor"] = tuple(c * intensity for c in color)
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_glass(
        self,
        ior: float = 1.5,
        tint: tuple = (1.0, 1.0, 1.0),
        transmission: float = 1.0,
    ) -> int:
        """Add a glass/transmissive material."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (*tint, 1.0)
        mat["pbrRoughnessFactor"] = 0.0
        mat["pbrMetallicFactor"] = 0.0
        mat["transmissionFactor"] = transmission
        mat["ior"] = ior
        mat["thicknessFactor"] = 0.0  # thin-walled
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_pbr(
        self,
        base_color: tuple = (1.0, 1.0, 1.0),
        metallic: float = 0.0,
        roughness: float = 0.5,
        ior: float = 1.5,
        specular: float = 1.0,
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.1,
        u_subdiv: float = 0.0,
        v_subdiv: float = 0.0,
        base_color_scale: float = 0.75,
        emissive: tuple = (0.0, 0.0, 0.0),
    ) -> int:
        """Add a PBR material with custom properties."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (*base_color, 1.0)
        mat["emissiveFactor"] = tuple(float(v) for v in emissive)
        mat["pbrMetallicFactor"] = metallic
        mat["pbrRoughnessFactor"] = roughness
        mat["ior"] = ior
        mat["specularFactor"] = specular
        mat["clearcoatFactor"] = clearcoat
        mat["clearcoatRoughness"] = clearcoat_roughness
        self._set_checker_overlay(mat, u_subdiv, v_subdiv, base_color_scale)
        self._materials.append(mat)
        return len(self._materials) - 1

    def add_gltf_material(
        self,
        *,
        base_color: tuple = (1.0, 1.0, 1.0, 1.0),
        emissive_factor: tuple = (0.0, 0.0, 0.0),
        metallic: float = 1.0,
        roughness: float = 1.0,
        alpha_mode: str = "OPAQUE",
        alpha_cutoff: float = 0.5,
        double_sided: bool = False,
        opacity_fresnel: tuple[float, float, float] | None = None,
        base_color_texture: dict | None = None,
        normal_texture: dict | None = None,
        metallic_roughness_texture: dict | None = None,
        emissive_texture: dict | None = None,
        occlusion_texture: dict | None = None,
        occlusion_strength: float = 1.0,
        transmission: float = 0.0,
        ior: float = 1.5,
        specular: float = 1.0,
        specular_color: tuple = (1.0, 1.0, 1.0),
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.1,
        thickness: float = 0.0,
        attenuation_color: tuple = (1.0, 1.0, 1.0),
        attenuation_distance: float = 1.0e10,
        transmission_color: tuple | None = None,
        base_color_add: float = 0.0,
        base_color_desaturation: float = 0.0,
        u_subdiv: float = 0.0,
        v_subdiv: float = 0.0,
        base_color_scale: float = 0.75,
    ) -> int:
        """Add a glTF PBR material using reference loader-compatible fields."""
        mat = self._create_default_material()
        mat["pbrBaseColorFactor"] = (
            float(base_color[0]),
            float(base_color[1]),
            float(base_color[2]),
            float(base_color[3]),
        )
        mat["emissiveFactor"] = (
            float(emissive_factor[0]),
            float(emissive_factor[1]),
            float(emissive_factor[2]),
        )
        mat["pbrMetallicFactor"] = float(np.clip(metallic, 0.0, 1.0))
        mat["pbrRoughnessFactor"] = float(np.clip(roughness, 0.0, 1.0))
        mat["alphaCutoff"] = float(alpha_cutoff)
        mat["alphaMode"] = {
            "OPAQUE": ALPHA_OPAQUE,
            "MASK": ALPHA_MASK,
            "BLEND": ALPHA_BLEND,
        }.get(str(alpha_mode).upper(), ALPHA_OPAQUE)
        if opacity_fresnel is not None:
            mat["opacityFresnelLow"] = float(np.clip(opacity_fresnel[0], 0.0, 1.0))
            mat["opacityFresnelHigh"] = float(np.clip(opacity_fresnel[1], 0.0, 1.0))
            mat["opacityFresnelFalloff"] = float(max(opacity_fresnel[2], 0.0))
        mat["transmissionFactor"] = float(np.clip(transmission, 0.0, 1.0))
        if transmission_color is not None:
            mat["transmissionColorFactor"] = tuple(float(channel) for channel in transmission_color)
        mat["ior"] = float(max(ior, 1.0))
        mat["specularFactor"] = float(max(specular, 0.0))
        mat["specularColorFactor"] = tuple(float(channel) for channel in specular_color)
        mat["clearcoatFactor"] = float(np.clip(clearcoat, 0.0, 1.0))
        mat["clearcoatRoughness"] = float(np.clip(clearcoat_roughness, 0.0, 1.0))
        mat["occlusionStrength"] = float(np.clip(occlusion_strength, 0.0, 1.0))
        mat["thicknessFactor"] = float(max(thickness, 0.0))
        mat["attenuationColor"] = (
            float(attenuation_color[0]),
            float(attenuation_color[1]),
            float(attenuation_color[2]),
        )
        mat["attenuationDistance"] = float(max(attenuation_distance, 1.0e-6))
        mat["pbrDiffuseFactor"][0] = float(base_color_add)
        mat["pbrDiffuseFactor"][1] = float(np.clip(base_color_desaturation, 0.0, 1.0))
        self._set_checker_overlay(mat, u_subdiv, v_subdiv, base_color_scale)
        # Keep this information available although current pipeline does not use culling flags.
        mat["unlit"] = 0

        if normal_texture:
            scale = normal_texture.get("scale", 1.0)
            if np.isscalar(scale):
                scale = (scale, scale)
            mat["normalTextureScale"] = (float(scale[0]), float(scale[1]))

        def _set_tex(field: str, info: dict | None):
            if not info:
                return
            mat[field] = _texture_info_from_gltf(info)

        _set_tex("pbrBaseColorTexture", base_color_texture)
        _set_tex("normalTexture", normal_texture)
        _set_tex("pbrMetallicRoughnessTexture", metallic_roughness_texture)
        _set_tex("emissiveTexture", emissive_texture)
        _set_tex("occlusionTexture", occlusion_texture)

        self._materials.append(mat)
        return len(self._materials) - 1

    @staticmethod
    def _set_checker_overlay(mat, u_subdiv: float, v_subdiv: float, base_color_scale: float) -> None:
        """Configure the procedural UV checker overlay."""
        u_subdiv = float(u_subdiv)
        v_subdiv = float(v_subdiv)
        base_color_scale = float(base_color_scale)
        if u_subdiv < 0.0 or v_subdiv < 0.0:
            raise ValueError("u_subdiv and v_subdiv must be nonnegative")
        if not 0.0 <= base_color_scale <= 1.0:
            raise ValueError("base_color_scale must be in the range [0, 1]")
        mat["uSubdiv"] = u_subdiv
        mat["vSubdiv"] = v_subdiv
        mat["baseColorScale"] = base_color_scale

    def clear(self):
        """Remove all materials."""
        self._materials.clear()

    def _create_default_material(self):
        """Create a default PBR material."""
        mat = np.zeros(1, dtype=self._material_dtype)[0]

        # Base PBR
        mat["pbrBaseColorFactor"] = (1.0, 1.0, 1.0, 1.0)
        mat["emissiveFactor"] = (0.0, 0.0, 0.0)
        mat["normalTextureScale"] = (1.0, 1.0)
        mat["pbrRoughnessFactor"] = 0.5
        mat["pbrMetallicFactor"] = 0.0
        mat["uSubdiv"] = 0.0
        mat["vSubdiv"] = 0.0
        mat["baseColorScale"] = 0.75
        mat["alphaMode"] = ALPHA_OPAQUE
        mat["alphaCutoff"] = 0.5
        mat["opacityFresnelLow"] = -1.0
        mat["opacityFresnelHigh"] = 1.0
        mat["opacityFresnelFalloff"] = 1.0

        # Transmission
        mat["transmissionFactor"] = 0.0
        # Negative means transmission follows the evaluated, textured base color.
        mat["transmissionColorFactor"] = (-1.0, -1.0, -1.0)

        mat["ior"] = 1.5
        mat["attenuationColor"] = (1.0, 1.0, 1.0)
        mat["thicknessFactor"] = 0.0
        mat["attenuationDistance"] = 1e10

        # Clearcoat
        mat["clearcoatFactor"] = 0.0
        mat["clearcoatRoughness"] = 0.0

        # Specular
        mat["specularColorFactor"] = (1.0, 1.0, 1.0)
        mat["specularFactor"] = 1.0

        # Misc
        mat["unlit"] = 0

        # Iridescence
        mat["iridescenceFactor"] = 0.0
        mat["iridescenceThicknessMaximum"] = 400.0
        mat["iridescenceThicknessMinimum"] = 100.0
        mat["iridescenceIor"] = 1.3

        # Anisotropy
        mat["anisotropyStrength"] = 0.0
        mat["anisotropyRotation"] = (0.0, 1.0)

        # Sheen
        mat["sheenRoughnessFactor"] = 0.0
        mat["sheenColorFactor"] = (0.0, 0.0, 0.0)

        # Occlusion & Dispersion
        mat["occlusionStrength"] = 1.0
        mat["dispersion"] = 0.0

        # The compact material path stores base-color add and desaturation in
        # the first two legacy specular-glossiness channels.
        mat["pbrDiffuseFactor"] = (0.0, 0.0, 1.0, 1.0)
        mat["pbrSpecularFactor"] = (1.0, 1.0, 1.0)
        mat["usePbrSpecularGlossiness"] = 0
        mat["pbrGlossinessFactor"] = 1.0

        # Diffuse transmission
        mat["diffuseTransmissionColor"] = (1.0, 1.0, 1.0)
        mat["diffuseTransmissionFactor"] = 0.0

        # All textures default to no texture
        default_tex = _default_texture_info()
        for tex_name in [
            "pbrBaseColorTexture",
            "normalTexture",
            "pbrMetallicRoughnessTexture",
            "emissiveTexture",
            "transmissionTexture",
            "thicknessTexture",
            "clearcoatTexture",
            "clearcoatRoughnessTexture",
            "clearcoatNormalTexture",
            "specularTexture",
            "specularColorTexture",
            "iridescenceTexture",
            "iridescenceThicknessTexture",
            "anisotropyTexture",
            "sheenColorTexture",
            "sheenRoughnessTexture",
            "occlusionTexture",
            "pbrDiffuseTexture",
            "pbrSpecularGlossinessTexture",
            "diffuseTransmissionTexture",
            "diffuseTransmissionColorTexture",
        ]:
            mat[tex_name] = default_tex

        return mat
