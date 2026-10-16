# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Modified Python/Warp adaptation of vk_denoise_dlssrr ray-generation, hit,
# miss, and DLSS guide-buffer shaders, through an intermediate CUDA/OptiX port:
# https://github.com/nvpro-samples/vk_denoise_dlssrr/tree/f8113a84fa952278bb1286e7899f81b7afe8f41e/dlss_rr/shaders
# Sky, BSDF, material evaluation, and sampling helpers derive from nvpro_core:
# https://github.com/nvpro-samples/nvpro_core/blob/bc19d6ac3ef62938d0ea0e099735878457ce1b6e/nvvkhl/shaders
# See the source-to-file mapping in THIRD_PARTY_NOTICES.txt.
# Original notices: licenses/vk_denoise_dlssrr-NOTICE.txt and
# licenses/nvpro_core-NOTICE.txt.


from __future__ import annotations

import warp as wp

from warp_optix.constants import (
    OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
)

from .func_common import Mat16f, Vec6f


@wp.struct
class PrimaryPayload:
    """Matches C++ RayPayload layout using 19 32-bit payload words.

    Register map (same as primary_rchit.h):
      0: hitT            (float)  -- DLSS_INF_DISTANCE for miss
      1-3: normal/envRad  (float3) -- shading normal for hits, env radiance for miss
      4-6: tangent        (float3)
      7-8: uv             (float2)
      9: materialId       (uint32)
     10: bitangentSign    (float)
     11: instanceId       (int as uint32)
     12: frontFace        (uint32)
     13: primitiveId      (uint32)
     14-16: lod/barycentrics (float3: texture LOD, barycentric x/y)
     17-18: uv1            (float2)
    """

    hit_t: wp.float32
    normal: wp.vec3
    tangent: wp.vec3
    uv: wp.vec2
    material_id: wp.uint32
    bitangent_sign: wp.float32
    instance_id: wp.int32
    front_face: wp.uint32
    primitive_id: wp.uint32
    barycentrics: wp.vec3
    uv1: wp.vec2


@wp.struct
class PrimaryMissPayload:
    """Compact miss payload layout (registers 0..3 only)."""

    hit_t: wp.float32
    env_radiance: wp.vec3


@wp.struct
class ShadowPayload:
    visible: wp.uint32
    seed: wp.uint32


@wp.struct
class VertexBuffers:
    position_offset: wp.uint32
    normal_offset: wp.uint32
    radius_offset: wp.uint32
    tangent_offset: wp.uint32
    texcoord0_offset: wp.uint32
    texcoord1_offset: wp.uint32
    prev_position_offset: wp.uint32
    has_texcoord1: wp.uint32
    has_prev_position: wp.uint32


@wp.struct
class RenderPrimitive:
    index_offset: wp.uint32
    material_id_offset: wp.uint32
    vertex_buffer: VertexBuffers
    num_indices: wp.uint32
    num_vertices: wp.uint32


@wp.struct
class CompactMaterial:
    base_color: wp.vec3
    emissive: wp.vec3
    roughness: wp.float32
    metallic: wp.float32
    u_subdiv: wp.float32
    v_subdiv: wp.float32
    base_color_scale: wp.float32
    base_color_add: wp.float32
    base_color_desaturation: wp.float32
    alpha_mode: wp.int32
    alpha_cutoff: wp.float32
    transmission: wp.float32
    transmission_color: wp.vec3
    texture_size: wp.float32
    ior: wp.float32
    specular_color: wp.vec3
    specular: wp.float32
    clearcoat: wp.float32
    clearcoat_roughness: wp.float32
    sheen_roughness: wp.float32
    occlusion: wp.float32
    occlusion_tex_index: wp.int32
    occlusion_tex_coord: wp.int32
    sheen_color: wp.vec3
    diffuse_transmission_factor: wp.float32
    diffuse_transmission_color: wp.vec3
    is_thin_walled: wp.int32
    clearcoat_normal_tex_index: wp.int32
    clearcoat_normal_tex_coord: wp.int32
    opacity: wp.float32
    opacity_fresnel_low: wp.float32
    opacity_fresnel_high: wp.float32
    opacity_fresnel_falloff: wp.float32
    base_color_tex_index: wp.int32
    base_color_tex_coord: wp.int32
    metallic_roughness_tex_index: wp.int32
    metallic_roughness_tex_coord: wp.int32
    normal_tex_index: wp.int32
    normal_tex_coord: wp.int32
    emissive_tex_index: wp.int32
    emissive_tex_coord: wp.int32
    normal_scale: wp.vec2
    base_color_uv_transform: Vec6f
    metallic_roughness_uv_transform: Vec6f
    normal_uv_transform: Vec6f
    emissive_uv_transform: Vec6f
    occlusion_uv_transform: Vec6f
    clearcoat_normal_uv_transform: Vec6f


@wp.struct
class EnvAccel:
    alias: wp.uint32
    q: wp.float32
    pdf: wp.float32
    pad: wp.float32


@wp.struct
class SphereLight:
    position_radius: wp.vec4
    radiance: wp.vec3
    pad: wp.float32


@wp.struct
class TransformMatrix3x4:
    row0: wp.vec4
    row1: wp.vec4
    row2: wp.vec4


@wp.struct
class PhysicalSkyParams:
    """1:1 match with C++ PhysicalSkyParameters (sky_common.h line 32)."""

    rgb_unit_conversion: wp.vec3
    multiplier: wp.float32
    haze: wp.float32
    redblueshift: wp.float32
    saturation: wp.float32
    horizon_height: wp.float32
    ground_color: wp.vec3
    horizon_blur: wp.float32
    night_color: wp.vec3
    sun_disk_intensity: wp.float32
    sun_direction: wp.vec3
    sun_disk_scale: wp.float32
    sun_glow_intensity: wp.float32
    y_is_up: wp.int32
    grayscale: wp.float32


@wp.struct
class DeviceCameraState:
    position: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_half_fov: wp.float32
    aspect: wp.float32
    previous_position: wp.vec3
    previous_forward: wp.vec3
    previous_right: wp.vec3
    previous_up: wp.vec3
    previous_tan_half_fov: wp.float32
    previous_aspect: wp.float32


@wp.struct
class PathtraceLaunchParams:
    tlas: wp.uint64
    width: wp.uint32
    height: wp.uint32
    frame_index: wp.uint32
    max_bounces: wp.uint32
    direct_light_samples: wp.uint32
    russian_roulette_start_bounce: wp.uint32
    output_mode: wp.int32

    device_camera: wp.array(dtype=DeviceCameraState)
    cam_pos: wp.vec3
    cam_forward: wp.vec3
    cam_right: wp.vec3
    cam_up: wp.vec3
    cam_tan_half_fov: wp.float32
    cam_aspect: wp.float32
    jitter: wp.vec2
    view: Mat16f
    proj: Mat16f
    view_inv: Mat16f
    proj_inv: Mat16f
    prev_mvp: Mat16f
    env_intensity: wp.vec3
    ambient_light: wp.vec3
    env_rotation: wp.float32
    flags: wp.uint32
    override_roughness: wp.float32
    override_metallic: wp.float32
    bitangent_flip: wp.float32
    use_procedural_sky: wp.uint32
    sky_lut: wp.array2d(dtype=wp.vec4)
    sky_lut_width: wp.uint32
    sky_lut_height: wp.uint32
    env_map: wp.array(dtype=wp.float32)
    env_map_length: wp.uint32
    env_map_width: wp.uint32
    env_map_height: wp.uint32
    env_accel: wp.array(dtype=EnvAccel)
    env_accel_count: wp.uint32
    sky: PhysicalSkyParams
    sphere_lights: wp.array(dtype=SphereLight)
    sphere_light_count: wp.uint32
    analytic_light_intensity: wp.float32
    emissive_material_intensity: wp.float32

    render_primitives: wp.array(dtype=RenderPrimitive)
    render_prim_count: wp.uint32
    instance_render_prim_ids: wp.array(dtype=wp.uint32)
    instance_geometry_types: wp.array(dtype=wp.uint32)
    instance_material_ids: wp.array(dtype=wp.uint32)
    instance_count: wp.uint32
    instance_transforms: wp.array(dtype=TransformMatrix3x4)
    prev_instance_transforms: wp.array(dtype=TransformMatrix3x4)
    compact_materials: wp.array(dtype=CompactMaterial)
    packed_indices: wp.array(dtype=wp.uint32)
    packed_positions: wp.array(dtype=wp.float32)
    packed_radii: wp.array(dtype=wp.float32)
    packed_normals: wp.array(dtype=wp.float32)
    packed_tangents: wp.array(dtype=wp.float32)
    packed_texcoords0: wp.array(dtype=wp.float32)
    packed_texcoords1: wp.array(dtype=wp.float32)
    packed_prev_positions: wp.array(dtype=wp.float32)
    packed_material_ids: wp.array(dtype=wp.uint32)
    material_count: wp.uint32
    textures: wp.array(dtype=wp.Texture2D)
    texture_count: wp.uint32

    color_output: wp.array2d(dtype=wp.vec4)
    normal_roughness_output: wp.array2d(dtype=wp.vec4)
    motion_output: wp.array2d(dtype=wp.vec2)
    depth_output: wp.array2d(dtype=wp.float32)
    diffuse_output: wp.array2d(dtype=wp.vec4)
    specular_output: wp.array2d(dtype=wp.vec4)
    spec_hit_dist_output: wp.array2d(dtype=wp.float32)


@wp.func
def _unpack_unit_from_u8(x: wp.uint32) -> wp.float32:
    return (wp.float32(x) / 255.0) * 2.0 - 1.0


@wp.func
def _decode_u8(x: wp.uint32) -> wp.float32:
    return wp.float32(x) / 255.0


@wp.func
def _decode_u16_norm(x: wp.uint32, max_value: wp.float32) -> wp.float32:
    return (wp.float32(x) / 65535.0) * max_value


@wp.func
def _fetch_vec2(flat: wp.array(dtype=wp.float32), idx: wp.int32) -> wp.vec2:
    base = idx * 2
    return wp.vec2(flat[base], flat[base + 1])


@wp.func
def _fetch_vec3(flat: wp.array(dtype=wp.float32), idx: wp.int32) -> wp.vec3:
    base = idx * 3
    return wp.vec3(flat[base], flat[base + 1], flat[base + 2])


@wp.func
def _fetch_vec4(flat: wp.array(dtype=wp.float32), idx: wp.int32) -> wp.vec4:
    base = idx * 4
    return wp.vec4(flat[base], flat[base + 1], flat[base + 2], flat[base + 3])


@wp.func
def _apply_uv_transform(uv: wp.vec2, m: Vec6f) -> wp.vec2:
    return wp.vec2(m[0] * uv[0] + m[1] * uv[1] + m[2], m[3] * uv[0] + m[4] * uv[1] + m[5])


@wp.func
def _select_uv(tex_coord: wp.int32, uv0: wp.vec2, uv1: wp.vec2) -> wp.vec2:
    if tex_coord == 1:
        return uv1
    return uv0


@wp.func
def _wrap_repeat_uv(uv: wp.vec2) -> wp.vec2:
    return wp.vec2(uv[0] - wp.floor(uv[0]), uv[1] - wp.floor(uv[1]))


@wp.func
def _wrap_repeat_index(i: wp.int32, size: wp.int32) -> wp.int32:
    if size <= 0:
        return 0
    x = i % size
    if x < 0:
        x = x + size
    return x


@wp.func
def _sample_texture_rgba(
    params: PathtraceLaunchParams,
    tex_index: wp.int32,
    uv: wp.vec2,
    lod: wp.float32,
) -> wp.vec4:
    if tex_index < 0 or params.texture_count == wp.uint32(0):
        return wp.vec4(1.0, 1.0, 1.0, 1.0)
    if tex_index >= int(params.texture_count):
        return wp.vec4(1.0, 1.0, 1.0, 1.0)

    return wp.texture_sample(
        params.textures[tex_index],
        _wrap_repeat_uv(uv),
        dtype=wp.vec4,
        lod=lod,
    )


@wp.func
def _compute_pixel_center(params: PathtraceLaunchParams, px: wp.int32, py: wp.int32) -> wp.vec2:
    return wp.vec2(wp.float32(px) + 0.5 + params.jitter[0], wp.float32(py) + 0.5 + params.jitter[1])


@wp.func
def _compute_pixel_center_unjittered(px: wp.int32, py: wp.int32) -> wp.vec2:
    return wp.vec2(wp.float32(px) + 0.5, wp.float32(py) + 0.5)


@wp.func
def _mul_mat3x3_cm(m: Mat16f, v: wp.vec3) -> wp.vec3:
    return wp.vec3(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
    )


@wp.func
def _compute_ray_dir(params: PathtraceLaunchParams, px: wp.int32, py: wp.int32) -> wp.vec3:
    w = wp.float32(params.width)
    h = wp.float32(params.height)
    pixel_center = _compute_pixel_center(params, px, py)
    in_uv = wp.vec2(pixel_center[0] / w, pixel_center[1] / h)
    d = wp.vec2(in_uv[0] * 2.0 - 1.0, in_uv[1] * 2.0 - 1.0)
    if params.device_camera.shape[0] > 0:
        camera = params.device_camera[0]
        return wp.normalize(
            camera.forward
            + camera.right * (d[0] * camera.tan_half_fov * camera.aspect)
            - camera.up * (d[1] * camera.tan_half_fov)
        )
    target = _mul_mat4_cm(params.proj_inv, wp.vec4(d[0], d[1], 0.01, 1.0))
    view_dir = wp.normalize(wp.vec3(target[0], target[1], target[2]))
    return wp.normalize(_mul_mat3x3_cm(params.view_inv, view_dir))


@wp.func
def _compute_ray_origin(params: PathtraceLaunchParams) -> wp.vec3:
    if params.device_camera.shape[0] > 0:
        return params.device_camera[0].position
    eye = _mul_mat4_cm(params.view_inv, wp.vec4(0.0, 0.0, 0.0, 1.0))
    return wp.vec3(eye[0], eye[1], eye[2])


# ---------------------------------------------------------------------------
# Physical sky model — 1:1 translation of Newton sky_common.h
# ---------------------------------------------------------------------------


@wp.func
def _sky_rgb_luminance(rgb: wp.vec3) -> wp.float32:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


@wp.func
def _sky_local_coords_to_dir(main_vec: wp.vec3, lx: wp.float32, ly: wp.float32, lz: wp.float32) -> wp.vec3:
    u = (
        wp.vec3(0.0, -main_vec[2], main_vec[1])
        if wp.abs(main_vec[0]) < wp.abs(main_vec[1])
        else wp.vec3(main_vec[2], 0.0, -main_vec[0])
    )
    u = wp.normalize(u)
    v = wp.cross(main_vec, u)
    return u * lx + v * ly + main_vec * lz


@wp.func
def _sky_square_to_disk(in_x: wp.float32, in_y: wp.float32) -> wp.vec2:
    lx = 2.0 * in_x - 1.0
    ly = 2.0 * in_y - 1.0
    if lx == 0.0 and ly == 0.0:
        return wp.vec2(0.0, 0.0)
    r = wp.float32(0.0)
    phi = wp.float32(0.0)
    if lx > -ly:
        if lx > ly:
            r = lx
            phi = (wp.pi / 4.0) * (1.0 + ly / lx)
        else:
            r = ly
            phi = (wp.pi / 4.0) * (3.0 - lx / ly)
    else:
        if lx < ly:
            r = -lx
            phi = (wp.pi / 4.0) * (5.0 + ly / lx)
        else:
            r = -ly
            phi = (wp.pi / 4.0) * (7.0 - lx / ly)
    return wp.vec2(r, phi)


@wp.func
def _sky_reflection_dir_diffuse(n: wp.vec3, s0: wp.float32, s1: wp.float32) -> wp.vec3:
    rp = _sky_square_to_disk(s0, s1)
    lx = rp[0] * wp.cos(rp[1])
    ly = rp[0] * wp.sin(rp[1])
    lz = wp.sqrt(wp.max(0.0, 1.0 - lx * lx - ly * ly))
    return _sky_local_coords_to_dir(n, lx, ly, lz)


@wp.func
def _sky_calc_sun_color(sun_dir_z: wp.float32, turbidity: wp.float32) -> wp.vec3:
    if sun_dir_z <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    ko = wp.vec3(12.0, 8.5, 0.9)
    wl = wp.vec3(0.610, 0.550, 0.470)
    sol_rad = wp.vec3(1.0, 0.992, 0.911) * (127500.0 / 0.9878)
    m = 1.0 / (sun_dir_z + 0.15 * wp.pow(93.885 - (wp.acos(sun_dir_z) * 180.0 / wp.pi), -1.253))
    beta = 0.04608 * turbidity - 0.04586
    ta = wp.vec3(
        wp.exp(-m * beta * wp.pow(wl[0], -1.3)),
        wp.exp(-m * beta * wp.pow(wl[1], -1.3)),
        wp.exp(-m * beta * wp.pow(wl[2], -1.3)),
    )
    to = wp.vec3(
        wp.exp(-m * ko[0] * 0.0035),
        wp.exp(-m * ko[1] * 0.0035),
        wp.exp(-m * ko[2] * 0.0035),
    )
    tr = wp.vec3(
        wp.exp(-m * 0.008735 * wp.pow(wl[0], -4.08)),
        wp.exp(-m * 0.008735 * wp.pow(wl[1], -4.08)),
        wp.exp(-m * 0.008735 * wp.pow(wl[2], -4.08)),
    )
    return wp.vec3(
        tr[0] * ta[0] * to[0] * sol_rad[0],
        tr[1] * ta[1] * to[1] * sol_rad[1],
        tr[2] * ta[2] * to[2] * sol_rad[2],
    )


@wp.func
def _sky_luminance(d: wp.vec3, sun_dir: wp.vec3, turbidity: wp.float32) -> wp.float32:
    """skyLuminance (sky_common.h line 198) — recomputes its own clamped cosGamma."""
    cos_gamma = wp.clamp(wp.dot(sun_dir, d), 0.0, 1.0)
    gamma = wp.acos(cos_gamma)
    cos_theta = d[2]
    cos_theta_sun = sun_dir[2]
    theta_sun = wp.acos(cos_theta_sun)
    A = 0.178721 * turbidity - 1.463037
    B = -0.355402 * turbidity + 0.427494
    C = -0.022669 * turbidity + 5.325056
    D = 0.120647 * turbidity - 2.577052
    E = -0.066967 * turbidity + 0.370275
    num = (1.0 + A * wp.exp(B / cos_theta)) * (1.0 + C * wp.exp(D * gamma) + E * cos_gamma * cos_gamma)
    den = (1.0 + A * wp.exp(B)) * (1.0 + C * wp.exp(D * theta_sun) + E * cos_theta_sun * cos_theta_sun)
    return num / wp.max(den, 1.0e-10)


@wp.func
def _sky_color_xyz(
    d: wp.vec3,
    sun_pos: wp.vec3,
    turbidity: wp.float32,
    luminance: wp.float32,
) -> wp.vec3:
    """skyColorXyz (sky_common.h line 141) — takes full float3 vectors like C++."""
    cos_gamma = wp.dot(sun_pos, d)
    if cos_gamma > 1.0:
        cos_gamma = 2.0 - cos_gamma
    gamma = wp.acos(cos_gamma)
    cos_theta = d[2]
    cos_theta_sun = sun_pos[2]
    theta_sun = wp.acos(cos_theta_sun)
    t2 = turbidity * turbidity
    ts2 = theta_sun * theta_sun
    ts3 = ts2 * theta_sun

    zenith_x = (
        (0.001650 * ts3 - 0.003742 * ts2 + 0.002088 * theta_sun + 0.0) * t2
        + (-0.029028 * ts3 + 0.063773 * ts2 - 0.032020 * theta_sun + 0.003948) * turbidity
        + (0.116936 * ts3 - 0.211960 * ts2 + 0.060523 * theta_sun + 0.258852)
    )
    zenith_y = (
        (0.002759 * ts3 - 0.006105 * ts2 + 0.003162 * theta_sun + 0.0) * t2
        + (-0.042149 * ts3 + 0.089701 * ts2 - 0.041536 * theta_sun + 0.005158) * turbidity
        + (0.153467 * ts3 - 0.267568 * ts2 + 0.066698 * theta_sun + 0.266881)
    )

    A = -0.019257 * turbidity - (0.29 - wp.pow(cos_theta_sun, 0.5) * 0.09)
    B = -0.066513 * turbidity + 0.000818
    C = -0.000417 * turbidity + 0.212479
    D = -0.064097 * turbidity - 0.898875
    E = -0.003251 * turbidity + 0.045178
    x_val = ((1.0 + A * wp.exp(B / cos_theta)) * (1.0 + C * wp.exp(D * gamma) + E * cos_gamma * cos_gamma)) / (
        (1.0 + A * wp.exp(B)) * (1.0 + C * wp.exp(D * theta_sun) + E * cos_theta_sun * cos_theta_sun)
    )

    A2 = -0.016698 * turbidity - 0.260787
    B2 = -0.094958 * turbidity + 0.009213
    C2 = -0.007928 * turbidity + 0.210230
    D2 = -0.044050 * turbidity - 1.653694
    E2 = -0.010922 * turbidity + 0.052919
    y_val = ((1.0 + A2 * wp.exp(B2 / cos_theta)) * (1.0 + C2 * wp.exp(D2 * gamma) + E2 * cos_gamma * cos_gamma)) / (
        (1.0 + A2 * wp.exp(B2)) * (1.0 + C2 * wp.exp(D2 * theta_sun) + E2 * cos_theta_sun * cos_theta_sun)
    )

    local_saturation = 1.0
    cx = zenith_x * (x_val * local_saturation + (1.0 - local_saturation))
    cy = zenith_y * (y_val * local_saturation + (1.0 - local_saturation))
    xyz_y = luminance
    xyz_x = (cx / cy) * xyz_y
    xyz_z = ((1.0 - cx - cy) / cy) * xyz_y
    return wp.vec3(xyz_x, xyz_y, xyz_z)


@wp.func
def _sky_calc_sky_color(sun_dir: wp.vec3, d: wp.vec3, turbidity: wp.float32) -> wp.vec3:
    """calcSkyColor (sky_common.h line 218)."""
    theta_sun = wp.acos(sun_dir[2])
    chi = (4.0 / 9.0 - turbidity / 120.0) * (wp.pi - 2.0 * theta_sun)
    lum = 1000.0 * ((4.0453 * turbidity - 4.9710) * wp.tan(chi) - 0.2155 * turbidity + 2.4192)
    lum = lum * _sky_luminance(d, sun_dir, turbidity)
    xyz = _sky_color_xyz(d, sun_dir, turbidity, lum)
    env_color = wp.vec3(
        3.241 * xyz[0] - 1.537 * xyz[1] - 0.499 * xyz[2],
        -0.969 * xyz[0] + 1.876 * xyz[1] + 0.042 * xyz[2],
        0.056 * xyz[0] - 0.204 * xyz[1] + 1.057 * xyz[2],
    )
    return env_color * wp.pi


@wp.func
def _sky_calc_irradiance(sun_dir: wp.vec3, haze: wp.float32) -> wp.vec3:
    col_sum = wp.vec3(0.0, 0.0, 0.0)
    n_up = wp.vec3(0.0, 0.0, 1.0)
    u = wp.float32(0.1)
    while u < 1.0:
        v = wp.float32(0.1)
        while v < 1.0:
            diff = _sky_reflection_dir_diffuse(n_up, u, v)
            sc = _sky_calc_sky_color(sun_dir, diff, haze)
            col_sum = col_sum + sc
            v = v + 0.2
        u = u + 0.2
    return col_sum / 25.0


@wp.func
def _sky_tweak_saturation(sat: wp.float32, haze: wp.float32) -> wp.float32:
    if sat > 1.0:
        return 1.0
    low_sat = sat * sat * sat
    local_haze = wp.clamp((haze - 2.0) / 15.0, 0.0, 1.0)
    local_haze = local_haze * local_haze * local_haze
    return sat * (1.0 - local_haze) + low_sat * local_haze


@wp.func
def _sky_tweak_vector(d: wp.vec3, y_is_up: wp.int32, horiz_height: wp.float32) -> wp.vec3:
    out = d
    if y_is_up == 1:
        out = wp.vec3(d[0], d[2], d[1])
    if horiz_height != 0.0:
        out = wp.vec3(out[0], out[1], out[2] - horiz_height)
        out = wp.normalize(out)
    return out


@wp.func
def _sky_tweak_color(tint: wp.vec3, saturation: wp.float32, redness: wp.float32) -> wp.vec3:
    intensity = _sky_rgb_luminance(tint)
    gray = wp.vec3(intensity, intensity, intensity)
    out = gray * (1.0 - saturation) + tint * saturation if saturation > 0.0 else gray
    out = wp.vec3(out[0] * (1.0 + redness), out[1], out[2] * (1.0 - redness))
    return wp.vec3(wp.max(out[0], 0.0), wp.max(out[1], 0.0), wp.max(out[2], 0.0))


@wp.func
def _sky_night_brightness(sun_dir_z: wp.float32) -> wp.float32:
    lmt = 0.30901699437494742
    if sun_dir_z <= -lmt:
        return 0.0
    factor = (sun_dir_z + lmt) / lmt
    factor = factor * factor
    factor = factor * factor
    return factor


@wp.func
def _sky_calc_physical_scale(
    sun_disk_scale: wp.float32,
    sun_glow_intensity: wp.float32,
    sun_disk_intensity: wp.float32,
) -> wp.vec2:
    sun_angular_radius = 0.00465
    sun_disk_radius = sun_angular_radius * sun_disk_scale
    sun_glow_radius = sun_disk_radius * 10.0
    glow_func_integral = sun_glow_intensity * (
        4.0 * wp.pi
        - 24.0 * wp.pi / (sun_glow_radius * sun_glow_radius)
        + 24.0 * wp.pi * wp.sin(sun_glow_radius) / (sun_glow_radius * sun_glow_radius * sun_glow_radius)
    )
    target_sundisk_integral = sun_disk_intensity * wp.pi
    sky_sunglow_scale = 1.0
    max_glow_integral = 0.5 * target_sundisk_integral
    if glow_func_integral > max_glow_integral:
        sky_sunglow_scale = max_glow_integral / glow_func_integral
        target_sundisk_integral = target_sundisk_integral - max_glow_integral
    else:
        target_sundisk_integral = target_sundisk_integral - glow_func_integral
    sundisk_area = 2.0 * wp.pi * (1.0 - wp.cos(sun_disk_radius))
    target_sundisk_intensity = target_sundisk_integral / wp.max(sundisk_area, 1.0e-20)
    actual_sundisk_integral = 1.0 * sundisk_area
    actual_sundisk_intensity = sun_disk_intensity * 100.0 * actual_sundisk_integral / wp.max(sundisk_area, 1.0e-20)
    disk_scale = (
        target_sundisk_intensity / wp.max(actual_sundisk_intensity, 1.0e-20) if target_sundisk_intensity != 0.0 else 0.0
    )
    return wp.vec2(disk_scale, sky_sunglow_scale)


@wp.func
def _sky_hash_cell(x: wp.uint32, y: wp.uint32, z: wp.uint32) -> wp.uint32:
    h = x * wp.uint32(0x8DA6B343)
    h = h ^ (y * wp.uint32(0xD8163841))
    h = h ^ (z * wp.uint32(0xCB1AB31F))
    h = h ^ (h >> wp.uint32(16))
    h = h * wp.uint32(0x7FEB352D)
    h = h ^ (h >> wp.uint32(15))
    return h


@wp.func
def _sky_star_radiance(direction: wp.vec3) -> wp.vec3:
    """Return a stable sparse star sample for an internal Z-up direction."""
    if direction[2] <= 0.0:
        return wp.vec3(0.0)
    horizon_visibility = wp.clamp((direction[2] - 0.12) / 0.18, 0.0, 1.0)
    horizon_visibility = horizon_visibility * horizon_visibility * (3.0 - 2.0 * horizon_visibility)
    grid = 768.0
    cell_x = wp.uint32(wp.floor((direction[0] + 1.0) * grid))
    cell_y = wp.uint32(wp.floor((direction[1] + 1.0) * grid))
    cell_z = wp.uint32(wp.floor((direction[2] + 1.0) * grid))
    h = _sky_hash_cell(cell_x, cell_y, cell_z)
    sample = wp.float32(h & wp.uint32(0x00FFFFFF)) * (1.0 / 16777216.0)
    threshold = 0.9996
    if sample <= threshold:
        return wp.vec3(0.0)
    relative = (sample - threshold) / (1.0 - threshold)
    magnitude = relative * relative * relative
    brightness = 0.006 + 0.07 * magnitude * magnitude
    tint_hash = wp.float32((h >> wp.uint32(24)) & wp.uint32(0xFF)) / 255.0
    tint = wp.vec3(1.0, 0.78 + 0.22 * tint_hash, 0.62 + 0.38 * tint_hash)
    return tint * brightness * horizon_visibility


@wp.func
def _eval_physical_sky_celestial(ss: PhysicalSkyParams, in_direction: wp.vec3) -> wp.vec3:
    """Evaluate high-frequency sun, moon, halo, and star radiance."""
    if ss.multiplier <= 0.0:
        return wp.vec3(0.0)

    celestial_dir = wp.normalize(_sky_tweak_vector(in_direction, ss.y_is_up, 0.0))
    real_sun_dir = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, 0.0))
    disk_direction = real_sun_dir
    is_moon = real_sun_dir[2] < 0.0
    if is_moon:
        disk_direction = -real_sun_dir

    result = wp.vec3(0.0)
    if ss.sun_disk_intensity > 0.0 and ss.sun_disk_scale > 0.0:
        sun_angle = wp.acos(wp.clamp(wp.dot(celestial_dir, disk_direction), -1.0, 1.0))
        glow_scale = 25.0 if is_moon else 10.0
        glow_radius = 0.00465 * ss.sun_disk_scale * glow_scale
        if sun_angle < glow_radius:
            center_proximity = 1.0 - sun_angle / glow_radius
            if is_moon:
                disk_radius = 0.00465 * ss.sun_disk_scale
                disk_edge = wp.clamp(
                    (disk_radius - sun_angle) / wp.max(disk_radius * 0.08, 1.0e-6),
                    0.0,
                    1.0,
                )
                disk_edge = disk_edge * disk_edge * (3.0 - 2.0 * disk_edge)
                halo = wp.pow(center_proximity, 3.0) * 0.06 * ss.sun_glow_intensity
                result = wp.vec3(0.040, 0.044, 0.050) * ss.sun_disk_intensity * (disk_edge + halo)
            else:
                height_adjusted = (ss.horizon_height + ss.horizon_blur) / 10.0
                sun_dir = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, height_adjusted))
                if sun_dir[2] < 0.001:
                    sun_dir = wp.normalize(wp.vec3(sun_dir[0], sun_dir[1], 0.001))
                local_haze = wp.max(2.0, 2.0 + ss.haze)
                data_sun_color = _sky_calc_sun_color(sun_dir[2], local_haze)
                scales = _sky_calc_physical_scale(ss.sun_disk_scale, ss.sun_glow_intensity, ss.sun_disk_intensity)
                glow_factor = wp.pow(center_proximity, 3.0) * 2.0 * ss.sun_glow_intensity * scales[1]
                smooth_edge = 0.95 + local_haze / 500.0
                t_ss = wp.clamp((center_proximity - 0.85) / (smooth_edge - 0.85), 0.0, 1.0)
                disk_factor = (t_ss * t_ss * (3.0 - 2.0 * t_ss)) * 100.0 * ss.sun_disk_intensity * scales[0]
                rgb_scale = ss.rgb_unit_conversion * ss.multiplier
                disk_color = wp.vec3(
                    data_sun_color[0] * (glow_factor + disk_factor) * rgb_scale[0],
                    data_sun_color[1] * (glow_factor + disk_factor) * rgb_scale[1],
                    data_sun_color[2] * (glow_factor + disk_factor) * rgb_scale[2],
                )
                local_saturation = _sky_tweak_saturation(ss.saturation, local_haze)
                result = _sky_tweak_color(disk_color, local_saturation, ss.redblueshift) * wp.pi

    star_visibility = wp.clamp((-real_sun_dir[2] - 0.05) / 0.2, 0.0, 1.0)
    if star_visibility > 0.0 and celestial_dir[2] > 0.0:
        result = result + _sky_star_radiance(celestial_dir) * star_visibility
    return result


@wp.func
def _eval_physical_sky(
    ss: PhysicalSkyParams,
    in_direction: wp.vec3,
    include_celestial: wp.bool,
    apply_grayscale: wp.bool,
) -> wp.vec3:
    """1:1 translation of C++ evalPhysicalSky (sky_common.h line 323)."""
    if ss.multiplier <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    factor = 1.0
    night_factor = 1.0
    rgb_scale = ss.rgb_unit_conversion * ss.multiplier
    height_adjusted = (ss.horizon_height + ss.horizon_blur) / 10.0
    d = _sky_tweak_vector(in_direction, ss.y_is_up, height_adjusted)
    local_haze = wp.max(2.0, 2.0 + ss.haze)
    local_saturation = _sky_tweak_saturation(ss.saturation, local_haze)

    downness = d[2]
    if d[2] < 0.001:
        d = wp.normalize(wp.vec3(d[0], d[1], 0.001))

    sun_dir = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, height_adjusted))
    if sun_dir[2] < 0.001:
        factor = _sky_night_brightness(sun_dir[2])
        sun_dir = wp.normalize(wp.vec3(sun_dir[0], sun_dir[1], 0.001))

    tint = _sky_calc_sky_color(sun_dir, d, local_haze) * factor if factor > 0.0 else wp.vec3(0.0, 0.0, 0.0)
    data_sun_color = _sky_calc_sun_color(sun_dir[2], local_haze if downness > 0.0 else 2.0)

    out_color = wp.vec3(tint[0] * rgb_scale[0], tint[1] * rgb_scale[1], tint[2] * rgb_scale[2])

    if downness <= 0.0:
        irrad = _sky_calc_irradiance(sun_dir, 2.0)
        down_color = wp.vec3(
            ss.ground_color[0] * (irrad[0] + data_sun_color[0] * sun_dir[2]) * rgb_scale[0],
            ss.ground_color[1] * (irrad[1] + data_sun_color[1] * sun_dir[2]) * rgb_scale[1],
            ss.ground_color[2] * (irrad[2] + data_sun_color[2] * sun_dir[2]) * rgb_scale[2],
        )
        down_color = down_color * factor
        hor_blur = ss.horizon_blur / 10.0
        if hor_blur > 0.0:
            dness = wp.clamp(-downness / hor_blur, 0.0, 1.0)
            dness = dness * dness * (3.0 - 2.0 * dness)
            out_color = out_color * (1.0 - dness) + down_color * dness
            night_factor = 1.0 - dness
        else:
            out_color = down_color
            night_factor = 0.0

    out_color = _sky_tweak_color(out_color, local_saturation, ss.redblueshift)
    result = out_color * wp.pi

    if night_factor > 0.0:
        night = ss.night_color * night_factor
        result = wp.vec3(
            wp.max(result[0], night[0]),
            wp.max(result[1], night[1]),
            wp.max(result[2], night[2]),
        )
    if include_celestial:
        result = result + _eval_physical_sky_celestial(ss, in_direction)

    grayscale = wp.clamp(ss.grayscale, 0.0, 1.0)
    if apply_grayscale and grayscale > 0.0:
        gray = wp.max(result[0], wp.max(result[1], result[2]))
        result = result * (1.0 - grayscale) + wp.vec3(
            gray * grayscale,
            gray * grayscale,
            gray * grayscale,
        )

    return result


@wp.func
def _sky_sun_probability(ss: PhysicalSkyParams) -> wp.float32:
    """physicalSkySunProbability (sky_common.h line 400)."""
    if ss.sun_disk_scale <= 1.0e-5:
        return 0.0
    sun_direction = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, 0.0))
    if sun_direction[2] < 0.0:
        sun_direction = -sun_direction
    sun_elevation = sun_direction[2]
    return wp.clamp(ss.sun_disk_intensity * sun_elevation * 0.5 + 0.5, 0.1, 0.9)


@wp.func
def _sky_sample_pdf(ss: PhysicalSkyParams, in_direction: wp.vec3) -> wp.float32:
    """samplePhysicalSkyPDF (sky_common.h line 408)."""
    sun_angular_radius = 0.00465 * ss.sun_disk_scale
    internal_direction = _sky_tweak_vector(in_direction, ss.y_is_up, 0.0)
    sky_pdf = 1.0 / (2.0 * wp.pi) if internal_direction[2] >= 0.0 else 0.0
    sun_direction = wp.normalize(ss.sun_direction)
    internal_sun_direction = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, 0.0))
    if internal_sun_direction[2] < 0.0:
        sun_direction = -sun_direction
    sun_sample_angular_radius = 1.5 * sun_angular_radius
    sun_sample_solid_angle = (
        wp.pi * sun_sample_angular_radius * sun_sample_angular_radius
        if sun_sample_angular_radius < 0.001
        else 2.0 * wp.pi * (1.0 - wp.cos(sun_sample_angular_radius))
    )
    sun_pdf = (
        1.0 / wp.max(sun_sample_solid_angle, 1.0e-20)
        if wp.dot(in_direction, sun_direction) >= wp.cos(sun_sample_angular_radius)
        else 0.0
    )
    p_sun = _sky_sun_probability(ss)
    return sky_pdf * (1.0 - p_sun) + sun_pdf * p_sun


@wp.func
def _sky_sample_spherical_cap(z_min: wp.float32, xi0: wp.float32, xi1: wp.float32) -> wp.vec3:
    """sampleSphericalCap (sky_common.h line 422)."""
    z = 1.0 * (1.0 - xi1) + z_min * xi1
    r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    phi = 2.0 * wp.pi * xi0
    return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), z)


@wp.struct
class SkySamplingResult:
    direction: wp.vec3
    pdf: wp.float32


@wp.func
def _sample_physical_sky(ss: PhysicalSkyParams, xi0: wp.float32, xi1: wp.float32) -> SkySamplingResult:
    """samplePhysicalSky (sky_common.h line 433)."""
    result = SkySamplingResult()
    sun_prob = _sky_sun_probability(ss)
    z_min = 0.0
    sample_sun = wp.bool(xi0 < sun_prob)
    rx = xi0
    if sample_sun:
        rx = xi0 / wp.max(sun_prob, 1.0e-10)
        sun_sample_angular_radius = 1.5 * 0.00465 * ss.sun_disk_scale
        z_min = wp.cos(sun_sample_angular_radius)
    else:
        rx = (xi0 - sun_prob) / wp.max(1.0 - sun_prob, 1.0e-10)

    d = _sky_sample_spherical_cap(z_min, rx, xi1)

    if sample_sun:
        sun_direction = wp.normalize(_sky_tweak_vector(ss.sun_direction, ss.y_is_up, 0.0))
        if sun_direction[2] < 0.0:
            sun_direction = -sun_direction
        up = wp.vec3(1.0, 0.0, 0.0) if wp.abs(sun_direction[2]) > 0.999 else wp.vec3(0.0, 0.0, 1.0)
        right = wp.normalize(wp.cross(up, sun_direction))
        up = wp.cross(sun_direction, right)
        d = right * d[0] + up * d[1] + sun_direction * d[2]

    if ss.y_is_up == 1:
        d = wp.vec3(d[0], d[2], d[1])

    result.direction = wp.normalize(d)
    result.pdf = _sky_sample_pdf(ss, result.direction)
    return result


# ---------------------------------------------------------------------------
# Environment helpers (env map path kept, physical sky path added)
# ---------------------------------------------------------------------------


@wp.func
def _rotate_y(v: wp.vec3, angle: wp.float32) -> wp.vec3:
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec3(c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


@wp.func
def _eval_env_map(params: PathtraceLaunchParams, rd: wp.vec3) -> wp.vec3:
    """Bilinear HDR env-map lookup (unchanged from before)."""
    d = _rotate_y(rd, -params.env_rotation)
    phi = wp.atan2(d[2], d[0])
    theta = wp.acos(wp.clamp(d[1], -1.0, 1.0))
    u = (phi + wp.pi) * (0.5 / wp.pi)
    v = theta / wp.pi

    w = int(params.env_map_width)
    h = int(params.env_map_height)
    if w <= 0 or h <= 0:
        return wp.vec3(0.0, 0.0, 0.0)

    uu = u - wp.floor(u)
    vv = wp.clamp(v, 0.0, 1.0)
    fx = uu * wp.float32(w) - 0.5
    fy = vv * wp.float32(h) - 0.5
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    tx = fx - wp.float32(x0)
    ty = fy - wp.float32(y0)
    x1 = x0 + 1
    y1 = y0 + 1

    ix0 = _wrap_repeat_index(x0, w)
    ix1 = _wrap_repeat_index(x1, w)
    iy0 = wp.clamp(y0, 0, h - 1)
    iy1 = wp.clamp(y1, 0, h - 1)

    texels = params.env_map
    p00 = (iy0 * w + ix0) * 4
    p10 = (iy0 * w + ix1) * 4
    p01 = (iy1 * w + ix0) * 4
    p11 = (iy1 * w + ix1) * 4
    if (
        p00 + 2 >= int(params.env_map_length)
        or p10 + 2 >= int(params.env_map_length)
        or p01 + 2 >= int(params.env_map_length)
        or p11 + 2 >= int(params.env_map_length)
    ):
        return wp.vec3(0.0, 0.0, 0.0)

    c00 = wp.vec3(texels[p00], texels[p00 + 1], texels[p00 + 2])
    c10 = wp.vec3(texels[p10], texels[p10 + 1], texels[p10 + 2])
    c01 = wp.vec3(texels[p01], texels[p01 + 1], texels[p01 + 2])
    c11 = wp.vec3(texels[p11], texels[p11 + 1], texels[p11 + 2])
    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    c = c0 * (1.0 - ty) + c1 * ty
    return wp.vec3(
        c[0] * params.env_intensity[0],
        c[1] * params.env_intensity[1],
        c[2] * params.env_intensity[2],
    )


@wp.func
def _eval_sky_lut(params: PathtraceLaunchParams, rd: wp.vec3) -> wp.vec3:
    phi = wp.atan2(rd[2], rd[0])
    theta = wp.acos(wp.clamp(rd[1], -1.0, 1.0))
    u = (phi + wp.pi) * (0.5 / wp.pi)
    v = theta / wp.pi
    w = int(params.sky_lut_width)
    h = int(params.sky_lut_height)
    if w <= 0 or h <= 0:
        return _eval_physical_sky(params.sky, rd, wp.bool(True), wp.bool(True))
    fx = (u - wp.floor(u)) * wp.float32(w) - 0.5
    fy = wp.clamp(v, 0.0, 1.0) * wp.float32(h) - 0.5
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    tx = fx - wp.float32(x0)
    ty = fy - wp.float32(y0)
    x1 = x0 + 1
    y1 = y0 + 1
    ix0 = _wrap_repeat_index(x0, w)
    ix1 = _wrap_repeat_index(x1, w)
    iy0 = wp.clamp(y0, 0, h - 1)
    iy1 = wp.clamp(y1, 0, h - 1)
    lut = params.sky_lut
    c00 = lut[iy0, ix0]
    c10 = lut[iy0, ix1]
    c01 = lut[iy1, ix0]
    c11 = lut[iy1, ix1]
    c0 = wp.vec3(c00[0], c00[1], c00[2]) * (1.0 - tx) + wp.vec3(c10[0], c10[1], c10[2]) * tx
    c1 = wp.vec3(c01[0], c01[1], c01[2]) * (1.0 - tx) + wp.vec3(c11[0], c11[1], c11[2]) * tx
    result = c0 * (1.0 - ty) + c1 * ty
    result = result + _eval_physical_sky_celestial(params.sky, rd)
    grayscale = wp.clamp(params.sky.grayscale, 0.0, 1.0)
    if grayscale > 0.0:
        gray = wp.max(result[0], wp.max(result[1], result[2]))
        result = result * (1.0 - grayscale) + wp.vec3(gray * grayscale)
    return result


@wp.func
def _sample_environment(params: PathtraceLaunchParams, rd: wp.vec3) -> wp.vec3:
    """Matches C++ eval_environment / evalPhysicalSky dispatch."""
    if params.use_procedural_sky == wp.uint32(1):
        env = _eval_sky_lut(params, rd)
        return wp.vec3(
            env[0] * params.env_intensity[0],
            env[1] * params.env_intensity[1],
            env[2] * params.env_intensity[2],
        )
    if params.env_map_length > wp.uint32(0):
        return _eval_env_map(params, rd)
    env = _eval_physical_sky(params.sky, rd, wp.bool(True), wp.bool(True))
    return wp.vec3(
        env[0] * params.env_intensity[0],
        env[1] * params.env_intensity[1],
        env[2] * params.env_intensity[2],
    )


@wp.func
def _environment_pdf_for_direction(params: PathtraceLaunchParams, rd: wp.vec3) -> wp.float32:
    """Matches C++ samplePhysicalSkyPDF / env-map PDF dispatch."""
    if params.use_procedural_sky == wp.uint32(1):
        return _sky_sample_pdf(params.sky, rd)

    if params.env_map_length == wp.uint32(0):
        return _sky_sample_pdf(params.sky, rd)

    d = _rotate_y(rd, -params.env_rotation)
    phi = wp.atan2(d[2], d[0])
    theta = wp.acos(wp.clamp(d[1], -1.0, 1.0))
    u = (phi + wp.pi) * (0.5 / wp.pi)
    v = theta / wp.pi

    w = int(params.env_map_width)
    h = int(params.env_map_height)
    if w <= 0 or h <= 0:
        return 1.0 / (4.0 * wp.pi)

    uu = u - wp.floor(u)
    vv = wp.clamp(v, 0.0, 1.0)
    x = wp.min(int(uu * wp.float32(w)), w - 1)
    y = wp.min(int(vv * wp.float32(h)), h - 1)

    if params.env_accel_count > wp.uint32(0):
        size = int(params.env_accel_count)
        idx = y * w + x
        if idx >= 0 and idx < size:
            accel = params.env_accel
            return wp.max(accel[idx].pdf, 1.0e-6)

    texels = params.env_map
    base = (y * w + x) * 4
    if base + 3 >= int(params.env_map_length):
        return 1.0 / (4.0 * wp.pi)
    return wp.max(texels[base + 3], 1.0e-6)


@wp.func
def _power_heuristic(pdf_a: wp.float32, pdf_b: wp.float32) -> wp.float32:
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / wp.max(a2 + b2, 1.0e-8)


@wp.func
def _sample_cosine_hemisphere(n: wp.vec3, xi0: wp.float32, xi1: wp.float32) -> wp.vec3:
    up = wp.vec3(0.0, 0.0, 1.0) if wp.abs(n[2]) < 0.999 else wp.vec3(1.0, 0.0, 0.0)
    t = wp.normalize(wp.cross(up, n))
    b = wp.cross(n, t)
    r = wp.sqrt(wp.clamp(xi0, 0.0, 1.0))
    phi = 2.0 * wp.pi * xi1
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(1.0 - xi0, 0.0))
    return wp.normalize(t * x + b * y + n * z)


@wp.struct
class LightSample:
    direction: wp.vec3
    radiance: wp.vec3
    pdf: wp.float32


@wp.struct
class SphereLightSample:
    direction: wp.vec3
    radiance: wp.vec3
    pdf: wp.float32
    distance: wp.float32


@wp.func
def _sample_sphere_light(
    params: PathtraceLaunchParams,
    position: wp.vec3,
    xi0: wp.float32,
    xi1: wp.float32,
    xi2: wp.float32,
) -> SphereLightSample:
    """Sample one analytic sphere light uniformly over its visible solid angle."""
    sample = SphereLightSample()
    sample.direction = wp.vec3(0.0, 0.0, 1.0)
    sample.radiance = wp.vec3(0.0, 0.0, 0.0)
    sample.pdf = 0.0
    sample.distance = 0.0
    count = int(params.sphere_light_count)
    if count <= 0:
        return sample

    scaled_index = xi0 * wp.float32(count)
    index = wp.min(int(scaled_index), count - 1)
    light = params.sphere_lights[index]
    center = wp.vec3(
        light.position_radius[0],
        light.position_radius[1],
        light.position_radius[2],
    )
    radius = wp.max(light.position_radius[3], 1.0e-5)
    to_center = center - position
    distance_squared = wp.dot(to_center, to_center)
    radius_squared = radius * radius
    if distance_squared <= radius_squared * 1.0001:
        z = xi1 * 2.0 - 1.0
        phi = xi2 * 2.0 * wp.pi
        radial = wp.sqrt(wp.max(1.0 - z * z, 0.0))
        sample.direction = wp.vec3(radial * wp.cos(phi), z, radial * wp.sin(phi))
        sample.distance = radius
        sample.pdf = 1.0 / (4.0 * wp.pi * wp.float32(count))
    else:
        distance = wp.sqrt(distance_squared)
        axis = to_center / distance
        cos_theta_max = wp.sqrt(wp.max(1.0 - radius_squared / distance_squared, 0.0))
        one_minus_cos = radius_squared / (distance_squared * wp.max(1.0 + cos_theta_max, 1.0e-6))
        cos_theta = 1.0 - xi1 * one_minus_cos
        sin_theta = wp.sqrt(wp.max(1.0 - cos_theta * cos_theta, 0.0))
        phi = xi2 * 2.0 * wp.pi
        helper = wp.vec3(0.0, 0.0, 1.0) if wp.abs(axis[2]) < 0.999 else wp.vec3(1.0, 0.0, 0.0)
        tangent = wp.normalize(wp.cross(helper, axis))
        bitangent = wp.cross(axis, tangent)
        sample.direction = wp.normalize(
            tangent * (sin_theta * wp.cos(phi)) + bitangent * (sin_theta * wp.sin(phi)) + axis * cos_theta
        )
        center_t = wp.dot(to_center, sample.direction)
        discriminant = radius_squared - (distance_squared - center_t * center_t)
        sample.distance = center_t - wp.sqrt(wp.max(discriminant, 0.0))
        sample.pdf = 1.0 / (2.0 * wp.pi * wp.max(one_minus_cos, 1.0e-8) * wp.float32(count))
    sample.radiance = light.radiance * params.analytic_light_intensity
    return sample


@wp.struct
class SphereLightRayHit:
    radiance: wp.vec3
    pdf: wp.float32
    distance: wp.float32


@wp.func
def _intersect_sphere_lights(
    params: PathtraceLaunchParams,
    origin: wp.vec3,
    direction: wp.vec3,
    max_distance: wp.float32,
) -> SphereLightRayHit:
    """Return the nearest finite analytic emitter along a secondary ray."""
    hit = SphereLightRayHit()
    hit.radiance = wp.vec3(0.0)
    hit.pdf = 0.0
    hit.distance = max_distance
    count = int(params.sphere_light_count)
    if count <= 0 or params.analytic_light_intensity <= 0.0:
        return hit

    index = wp.int32(0)
    while index < count:
        light = params.sphere_lights[index]
        center = wp.vec3(
            light.position_radius[0],
            light.position_radius[1],
            light.position_radius[2],
        )
        radius = wp.max(light.position_radius[3], 1.0e-5)
        offset = origin - center
        half_b = wp.dot(offset, direction)
        c = wp.dot(offset, offset) - radius * radius
        discriminant = half_b * half_b - c
        if discriminant >= 0.0:
            root = wp.sqrt(discriminant)
            distance = -half_b - root
            if distance <= 0.001:
                distance = -half_b + root
            if distance > 0.001 and distance < hit.distance:
                center_distance_sq = wp.dot(offset, offset)
                if center_distance_sq <= radius * radius * 1.0001:
                    hit.pdf = 1.0 / (4.0 * wp.pi * wp.float32(count))
                else:
                    cos_theta_max = wp.sqrt(wp.max(1.0 - radius * radius / center_distance_sq, 0.0))
                    one_minus_cos = (radius * radius) / (center_distance_sq * wp.max(1.0 + cos_theta_max, 1.0e-6))
                    hit.pdf = 1.0 / (2.0 * wp.pi * wp.max(one_minus_cos, 1.0e-8) * wp.float32(count))
                hit.distance = distance
                hit.radiance = light.radiance * params.analytic_light_intensity
        index = index + wp.int32(1)
    return hit


@wp.func
def _sample_environment_light(
    params: PathtraceLaunchParams, xi0: wp.float32, xi1: wp.float32, xi2: wp.float32
) -> LightSample:
    """Matches C++ Step 2 light sampling: physical sky or env-map importance sampling."""
    s = LightSample()

    # Physical sky path (matches C++ FLAGS_ENVMAP_SKY branch).
    if params.use_procedural_sky == wp.uint32(1) or params.env_map_length == wp.uint32(0):
        sky_sample = _sample_physical_sky(params.sky, xi0, xi1)
        s.direction = sky_sample.direction
        s.radiance = _sample_environment(params, s.direction)
        s.pdf = sky_sample.pdf
        return s

    # Env-map importance sampling path (matches C++ sample_environment_importance).
    if params.env_accel_count == wp.uint32(0):
        z = xi0 * 2.0 - 1.0
        a = xi1 * 2.0 * wp.pi
        r = wp.sqrt(wp.max(1.0 - z * z, 0.0))
        s.direction = wp.normalize(wp.vec3(r * wp.cos(a), z, r * wp.sin(a)))
        s.radiance = _sample_environment(params, s.direction)
        s.pdf = 1.0 / (4.0 * wp.pi)
        return s

    size = int(params.env_accel_count)
    w = int(params.env_map_width)
    h = int(params.env_map_height)
    accel = params.env_accel
    texels = params.env_map

    u = xi0 * wp.float32(size)
    idx = wp.min(int(u), size - 1)
    entry = accel[idx]

    chosen = idx
    local_y = xi1
    if xi1 >= entry.q:
        chosen = int(entry.alias)
        local_y = (xi1 - entry.q) / wp.max(1.0 - entry.q, 1.0e-6)
    else:
        local_y = xi1 / wp.max(entry.q, 1.0e-6)

    px = chosen % w
    py = chosen // w
    uu = (wp.float32(px) + xi2) / wp.float32(w)
    vv = (wp.float32(py) + wp.clamp(local_y, 0.0, 1.0)) / wp.float32(h)
    phi = uu * 2.0 * wp.pi - wp.pi
    theta = vv * wp.pi
    st = wp.sin(theta)

    local_dir = wp.vec3(wp.cos(phi) * st, wp.cos(theta), wp.sin(phi) * st)
    s.direction = wp.normalize(_rotate_y(local_dir, params.env_rotation))

    base = (py * w + px) * 4
    if base + 2 < int(params.env_map_length):
        c = wp.vec3(texels[base], texels[base + 1], texels[base + 2])
        s.radiance = wp.vec3(
            c[0] * params.env_intensity[0],
            c[1] * params.env_intensity[1],
            c[2] * params.env_intensity[2],
        )
        s.pdf = wp.max(accel[chosen].pdf, 1.0e-6)
    else:
        s.radiance = _sample_environment(params, s.direction)
        s.pdf = 1.0 / (4.0 * wp.pi)

    return s


@wp.func
def _mul_mat4_cm(m: Mat16f, v: wp.vec4) -> wp.vec4:
    # Column-major 4x4 multiply compatible with C++ mul_cm helper.
    return wp.vec4(
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    )


DLSS_INF_DISTANCE = 65504.0
FLAGS_USE_PATH_REGULARIZATION = wp.uint32(4)
FLAGS_CULL_BACK_FACES = wp.uint32(8)


@wp.func
def _backface_culling_ray_flag(params: PathtraceLaunchParams) -> wp.uint32:
    """Return triangle culling flags shared by camera, bounce, and shadow rays."""
    result = wp.uint32(0)
    if (params.flags & FLAGS_CULL_BACK_FACES) != wp.uint32(0):
        result = wp.uint32(OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES)
    return result


@wp.func
def _compute_view_z(view: Mat16f, world_pos: wp.vec3) -> wp.float32:
    p = wp.vec4(world_pos[0], world_pos[1], world_pos[2], 1.0)
    view_space = _mul_mat4_cm(view, p)
    return -view_space[2]


@wp.func
def _reinhard_max(color: wp.vec3) -> wp.vec3:
    lum = wp.max(1.0e-7, wp.max(wp.max(color[0], color[1]), color[2]))
    reinhard = lum / (lum + 1.0)
    return color * (reinhard / lum)


@wp.func
def _positive_rcp(x: wp.float32) -> wp.float32:
    return 1.0 / wp.max(x, 1.0e-15)


@wp.func
def _environment_term_rtg(rf0: wp.vec3, n_dot_v: wp.float32, alpha_roughness: wp.float32) -> wp.vec3:
    xx = 1.0
    xy = n_dot_v
    xz = n_dot_v * n_dot_v
    xw = n_dot_v * xz

    yx = 1.0
    yy = alpha_roughness
    yz = alpha_roughness * alpha_roughness
    yw = alpha_roughness * yz

    m1x_x = 0.99044 * xx + 1.29678 * xy
    m1x_y = -1.28514 * xx + (-0.755907) * xy

    m2x_x = 1.0 * xx + 20.3225 * xy + 121.563 * xw
    m2x_y = 2.92338 * xx + (-27.0302) * xy + 626.13 * xw
    m2x_z = 59.4188 * xx + 222.592 * xy + 316.627 * xw

    m3x_x = 0.0365463 * xx + 9.0632 * xy
    m3x_y = 3.32707 * xx + (-9.04756) * xy

    m4x_x = 1.0 * xx + 9.04401 * xz + 5.56589 * xw
    m4x_y = 3.59685 * xx + (-16.3174) * xz + 19.7886 * xw
    m4x_z = -1.36772 * xx + 9.22949 * xz + (-20.2123) * xw

    dot_m1_y = m1x_x * yx + m1x_y * yy
    dot_m2_y = m2x_x * yx + m2x_y * yy + m2x_z * yw
    dot_m3_y = m3x_x * yx + m3x_y * yy
    dot_m4_y = m4x_x * yx + m4x_y * yy + m4x_z * yw

    bias = dot_m1_y * _positive_rcp(dot_m2_y)
    scale = dot_m3_y * _positive_rcp(dot_m4_y)

    result = rf0 * scale + wp.vec3(bias, bias, bias)
    return wp.vec3(
        wp.clamp(result[0], 0.0, 1.0),
        wp.clamp(result[1], 0.0, 1.0),
        wp.clamp(result[2], 0.0, 1.0),
    )


@wp.func
def _compute_camera_motion_vector(
    params: PathtraceLaunchParams,
    pixel_center: wp.vec2,
    motion_origin: wp.vec4,
    dim: wp.vec2,
) -> wp.vec2:
    if params.device_camera.shape[0] > 0:
        camera = params.device_camera[0]
        relative = wp.vec3(motion_origin[0], motion_origin[1], motion_origin[2])
        if motion_origin[3] != 0.0:
            relative = relative - camera.previous_position
        depth = wp.dot(relative, camera.previous_forward)
        if wp.abs(depth) < 1.0e-8:
            depth = wp.where(depth >= 0.0, 1.0e-8, -1.0e-8)
        horizontal = depth * camera.previous_tan_half_fov * camera.previous_aspect
        vertical = depth * camera.previous_tan_half_fov
        ndc_x = wp.dot(relative, camera.previous_right) / horizontal
        ndc_y = -wp.dot(relative, camera.previous_up) / vertical
        ox = (ndc_x * 0.5 + 0.5) * dim[0]
        oy = (ndc_y * 0.5 + 0.5) * dim[1]
        return wp.vec2(ox - pixel_center[0], oy - pixel_center[1])
    old = _mul_mat4_cm(params.prev_mvp, motion_origin)
    inv_w = 1.0 / old[3]
    ox = ((old[0] * inv_w) * 0.5 + 0.5) * dim[0]
    oy = ((old[1] * inv_w) * 0.5 + 0.5) * dim[1]
    return wp.vec2(ox - pixel_center[0], oy - pixel_center[1])


@wp.func
def _transform_point(m: TransformMatrix3x4, p: wp.vec3) -> wp.vec3:
    return wp.vec3(
        m.row0[0] * p[0] + m.row0[1] * p[1] + m.row0[2] * p[2] + m.row0[3],
        m.row1[0] * p[0] + m.row1[1] * p[1] + m.row1[2] * p[2] + m.row1[3],
        m.row2[0] * p[0] + m.row2[1] * p[1] + m.row2[2] * p[2] + m.row2[3],
    )


@wp.func
def _inverse_transform_point(m: TransformMatrix3x4, world_pos: wp.vec3) -> wp.vec3:
    a00 = m.row0[0]
    a01 = m.row0[1]
    a02 = m.row0[2]
    a10 = m.row1[0]
    a11 = m.row1[1]
    a12 = m.row1[2]
    a20 = m.row2[0]
    a21 = m.row2[1]
    a22 = m.row2[2]
    det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    if wp.abs(det) < 1.0e-12:
        return world_pos
    inv = 1.0 / det
    i00 = (a11 * a22 - a12 * a21) * inv
    i01 = (a02 * a21 - a01 * a22) * inv
    i02 = (a01 * a12 - a02 * a11) * inv
    i10 = (a12 * a20 - a10 * a22) * inv
    i11 = (a00 * a22 - a02 * a20) * inv
    i12 = (a02 * a10 - a00 * a12) * inv
    i20 = (a10 * a21 - a11 * a20) * inv
    i21 = (a01 * a20 - a00 * a21) * inv
    i22 = (a00 * a11 - a01 * a10) * inv
    d = wp.vec3(world_pos[0] - m.row0[3], world_pos[1] - m.row1[3], world_pos[2] - m.row2[3])
    return wp.vec3(
        i00 * d[0] + i01 * d[1] + i02 * d[2],
        i10 * d[0] + i11 * d[1] + i12 * d[2],
        i20 * d[0] + i21 * d[1] + i22 * d[2],
    )


@wp.func
def _transforms_equal(a: TransformMatrix3x4, b: TransformMatrix3x4) -> wp.bool:
    return (
        a.row0[0] == b.row0[0]
        and a.row0[1] == b.row0[1]
        and a.row0[2] == b.row0[2]
        and a.row0[3] == b.row0[3]
        and a.row1[0] == b.row1[0]
        and a.row1[1] == b.row1[1]
        and a.row1[2] == b.row1[2]
        and a.row1[3] == b.row1[3]
        and a.row2[0] == b.row2[0]
        and a.row2[1] == b.row2[1]
        and a.row2[2] == b.row2[2]
        and a.row2[3] == b.row2[3]
    )


@wp.func
def _compute_object_motion_vector(
    params: PathtraceLaunchParams,
    pixel_center: wp.vec2,
    world_pos: wp.vec3,
    instance_id: wp.int32,
    tri_id: wp.int32,
    bary_b1: wp.float32,
    bary_b2: wp.float32,
    dim: wp.vec2,
) -> wp.vec2:
    if (
        instance_id < 0
        or params.instance_transforms.shape[0] == 0
        or params.prev_instance_transforms.shape[0] == 0
        or params.instance_transforms.shape[0] <= instance_id
        or params.prev_instance_transforms.shape[0] <= instance_id
        or instance_id >= int(params.instance_count)
    ):
        return _compute_camera_motion_vector(
            params,
            pixel_center,
            wp.vec4(world_pos[0], world_pos[1], world_pos[2], 1.0),
            dim,
        )

    curr = params.instance_transforms
    prev = params.prev_instance_transforms
    curr_t = curr[instance_id]
    prev_t = prev[instance_id]

    prev_world = (
        world_pos
        if _transforms_equal(curr_t, prev_t)
        else _transform_point(prev_t, _inverse_transform_point(curr_t, world_pos))
    )

    # Deformable mesh path: use previous-frame vertex positions when available.
    if tri_id >= 0 and params.instance_render_prim_ids.shape[0] > 0 and params.render_primitives.shape[0] > 0:
        inst_to_prim = params.instance_render_prim_ids
        prim_id = int(inst_to_prim[instance_id])
        if prim_id >= 0 and prim_id < int(params.render_prim_count):
            render_prims = params.render_primitives
            rp = render_prims[prim_id]
            if rp.vertex_buffer.has_prev_position != wp.uint32(0):
                tri_count = int(rp.num_indices) // 3
                if tri_id < tri_count:
                    idx = params.packed_indices
                    index_base = int(rp.index_offset)
                    i0 = int(idx[index_base + tri_id * 3 + 0])
                    i1 = int(idx[index_base + tri_id * 3 + 1])
                    i2 = int(idx[index_base + tri_id * 3 + 2])
                    prev_pos = params.packed_prev_positions
                    prev_base = int(rp.vertex_buffer.prev_position_offset)
                    b1 = bary_b1
                    b2 = bary_b2
                    b0 = 1.0 - b1 - b2
                    p0 = _fetch_vec3(prev_pos, prev_base // 3 + i0)
                    p1 = _fetch_vec3(prev_pos, prev_base // 3 + i1)
                    p2 = _fetch_vec3(prev_pos, prev_base // 3 + i2)
                    prev_local = p0 * b0 + p1 * b1 + p2 * b2
                    prev_world = _transform_point(prev_t, prev_local)

    return _compute_camera_motion_vector(
        params,
        pixel_center,
        wp.vec4(prev_world[0], prev_world[1], prev_world[2], 1.0),
        dim,
    )


@wp.func
def _xxhash32(px: wp.uint32, py: wp.uint32, pz: wp.uint32) -> wp.uint32:
    """Matches C++ xxhash32(uint3 p) from pbr_common.h."""
    PRIME1 = wp.uint32(2246822519)
    PRIME2 = wp.uint32(3266489917)
    PRIME3 = wp.uint32(668265263)
    PRIME4 = wp.uint32(374761393)
    h32 = pz + PRIME4 + px * PRIME2
    h32 = PRIME3 * ((h32 << wp.uint32(17)) | (h32 >> wp.uint32(15)))
    h32 = h32 + py * PRIME2
    h32 = PRIME3 * ((h32 << wp.uint32(17)) | (h32 >> wp.uint32(15)))
    h32 = PRIME1 * (h32 ^ (h32 >> wp.uint32(15)))
    h32 = PRIME2 * (h32 ^ (h32 >> wp.uint32(13)))
    return h32 ^ (h32 >> wp.uint32(16))


@wp.func
def _pcg_advance(state: wp.uint32) -> wp.uint32:
    """Advance PCG state by one step (LCG). Matches C++ ``prev = state * A + B``."""
    return state * wp.uint32(747796405) + wp.uint32(2891336453)


@wp.func
def _pcg_rand01(state: wp.uint32) -> wp.float32:
    """Extract a uniform float in [0,1) from a PCG state.

    Matches C++ ``rand01(unsigned int& state)`` from pbr_common.h.
    Usage pattern (mirrors C++ by-reference mutation)::

        rng = _pcg_advance(rng)  # state = state * A + B
        val = _pcg_rand01(rng)  # extract float from new state
    """
    word = ((state >> ((state >> wp.uint32(28)) + wp.uint32(4))) ^ state) * wp.uint32(277803737)
    r = (word >> wp.uint32(22)) ^ word
    return wp.float32(r & wp.uint32(0x00FFFFFF)) * (1.0 / 16777216.0)


@wp.func
def _fract(x: wp.float32) -> wp.float32:
    return x - wp.floor(x)


@wp.func
def _rand01(px: wp.int32, py: wp.int32, frame: wp.int32, bounce: wp.int32, channel: wp.int32) -> wp.float32:
    s = wp.float32(px * 1973 + py * 9277 + frame * 26699 + bounce * 104729 + channel * 13007)
    return _fract(wp.sin(s) * 43758.5453)


@wp.func
def _hash_u32(x: wp.uint32) -> wp.uint32:
    h = x
    h = h ^ (h >> wp.uint32(16))
    h = h * wp.uint32(0x7FEB352D)
    h = h ^ (h >> wp.uint32(15))
    h = h * wp.uint32(0x846CA68B)
    h = h ^ (h >> wp.uint32(16))
    return h


@wp.func
def _make_trace_seed(px: wp.int32, py: wp.int32, frame: wp.int32, bounce: wp.int32, channel: wp.int32) -> wp.uint32:
    s = wp.uint32(px * 1973 + py * 9277 + frame * 26699 + bounce * 104729 + channel * 13007 + 0x9E3779B9)
    return _hash_u32(s)


@wp.func
def _rand01_from_seed(seed: wp.uint32, salt: wp.uint32) -> wp.float32:
    h = _hash_u32(seed ^ salt)
    return wp.float32(h & wp.uint32(0x00FFFFFF)) * (1.0 / 16777216.0)


@wp.func
def _mul_vec3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])


@wp.func
def _max_vec3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def _init_primary_payload() -> PrimaryPayload:
    p = PrimaryPayload()
    p.hit_t = DLSS_INF_DISTANCE
    p.normal = wp.vec3(0.0, 0.0, 0.0)
    p.tangent = wp.vec3(0.0, 0.0, 0.0)
    p.uv = wp.vec2(0.0, 0.0)
    p.material_id = wp.uint32(0)
    p.bitangent_sign = 0.0
    p.instance_id = wp.int32(0)
    p.front_face = wp.uint32(1)
    p.primitive_id = wp.uint32(0)
    p.barycentrics = wp.vec3(0.0, 0.0, 0.0)
    p.uv1 = wp.vec2(0.0, 0.0)
    return p


@wp.func
def _payload_get_hitT(p: PrimaryPayload) -> wp.float32:
    return p.hit_t


@wp.func
def _payload_get_normal(p: PrimaryPayload) -> wp.vec3:
    return p.normal


@wp.func
def _payload_get_tangent(p: PrimaryPayload) -> wp.vec3:
    return p.tangent


@wp.func
def _payload_get_uv(p: PrimaryPayload) -> wp.vec2:
    return p.uv


@wp.func
def _payload_get_uv1(p: PrimaryPayload) -> wp.vec2:
    return p.uv1


@wp.func
def _payload_get_materialId(p: PrimaryPayload) -> wp.uint32:
    return p.material_id


@wp.func
def _payload_get_bitangentSign(p: PrimaryPayload) -> wp.float32:
    return p.bitangent_sign


@wp.func
def _payload_get_instanceId(p: PrimaryPayload) -> wp.int32:
    return p.instance_id


@wp.func
def _payload_get_front_face(p: PrimaryPayload) -> wp.uint32:
    return p.front_face


@wp.func
def _payload_get_primitiveId(p: PrimaryPayload) -> wp.uint32:
    return p.primitive_id


@wp.func
def _payload_get_barycentrics(p: PrimaryPayload) -> wp.vec3:
    return p.barycentrics


MICROFACET_MIN_ROUGHNESS = 0.0014142

# ---------------------------------------------------------------------------
# GGX microfacet BSDF — 1:1 translation of Newton pbr_common.h
# ---------------------------------------------------------------------------

BSDF_EVENT_ABSORB = 0
BSDF_EVENT_DIFFUSE = 1
BSDF_EVENT_GLOSSY = 2
BSDF_EVENT_IMPULSE = 4
BSDF_EVENT_REFLECTION = 8
BSDF_EVENT_TRANSMISSION = 16
BSDF_EVENT_DIFFUSE_REFLECTION = 9
BSDF_EVENT_DIFFUSE_TRANSMISSION = 17
BSDF_EVENT_GLOSSY_REFLECTION = 10
BSDF_EVENT_GLOSSY_TRANSMISSION = 18
DIRAC = -1.0

LOBE_DIFFUSE_REFLECTION = 0
LOBE_DIFFUSE_TRANSMISSION = 1
LOBE_SPECULAR_TRANSMISSION = 2
LOBE_SPECULAR_REFLECTION = 3
LOBE_METAL_REFLECTION = 4
LOBE_SHEEN_REFLECTION = 5
LOBE_CLEARCOAT_REFLECTION = 6
LOBE_COUNT = 7


# --- pbr_common.h line 135 ---
@wp.func
def _hvd_ggx_eval(inv_roughness: wp.float32, hx: wp.float32, hy: wp.float32, hz: wp.float32) -> wp.float32:
    x = hx * inv_roughness
    y = hy * inv_roughness
    aniso = x * x + y * y
    f = aniso + hz * hz
    return (1.0 / wp.pi) * inv_roughness * inv_roughness * hz / wp.max(f * f, 1.0e-30)


# --- pbr_common.h line 144 ---
@wp.func
def _hvd_ggx_sample_vndf(
    kx: wp.float32,
    ky: wp.float32,
    kz: wp.float32,
    roughness: wp.float32,
    xi0: wp.float32,
    xi1: wp.float32,
) -> wp.vec3:
    v = wp.normalize(wp.vec3(kx * roughness, ky * roughness, kz))
    t1 = wp.normalize(wp.cross(v, wp.vec3(0.0, 0.0, 1.0))) if v[2] < 0.99999 else wp.vec3(1.0, 0.0, 0.0)
    t2 = wp.cross(t1, v)
    a = 1.0 / (1.0 + v[2])
    r = wp.sqrt(xi0)
    phi = wp.pi * xi1 / a if xi1 < a else wp.pi + wp.pi * (xi1 - a) / (1.0 - a)
    sp = wp.sin(phi)
    cp = wp.cos(phi)
    p1 = r * cp
    p2 = r * sp * (1.0 if xi1 < a else v[2])
    h = t1 * p1 + t2 * p2 + v * wp.sqrt(wp.max(0.0, 1.0 - p1 * p1 - p2 * p2))
    h = wp.vec3(h[0] * roughness, h[1] * roughness, wp.max(0.0, h[2]))
    return wp.normalize(h)


# --- pbr_common.h line 164 ---
@wp.func
def _smith_shadow_or_mask(kx: wp.float32, ky: wp.float32, kz: wp.float32, roughness: wp.float32) -> wp.float32:
    kz2 = kz * kz
    if kz2 == 0.0:
        return 0.0
    ax = kx * roughness
    ay = ky * roughness
    inv_a2 = (ax * ax + ay * ay) / kz2
    return 2.0 / (1.0 + wp.sqrt(1.0 + inv_a2))


@wp.func
def _hvd_sheen_eval(inv_roughness: wp.float32, nh: wp.float32) -> wp.float32:
    sin_theta2 = wp.max(0.0, 1.0 - nh * nh)
    sin_theta = wp.sqrt(sin_theta2)
    return (inv_roughness + 2.0) * wp.pow(sin_theta, inv_roughness) * 0.5 * (1.0 / wp.pi) * nh


@wp.func
def _vcavities_mask(nh: wp.float32, kh: wp.float32, nk: wp.float32) -> wp.float32:
    return wp.min(2.0 * nh * nk / wp.max(kh, 1.0e-12), 1.0)


@wp.func
def _vcavities_shadow_mask(
    nh: wp.float32,
    k1z: wp.float32,
    k1h: wp.float32,
    k2z: wp.float32,
    k2h: wp.float32,
) -> wp.vec3:
    g1 = _vcavities_mask(nh, k1h, k1z)
    g2 = _vcavities_mask(nh, k2h, k2z)
    return wp.vec3(g1, g2, wp.min(g1, g2))


@wp.func
def _hvd_sheen_sample(x0: wp.float32, x1: wp.float32, inv_roughness: wp.float32) -> wp.vec3:
    phi = 2.0 * wp.pi * x0
    sin_phi = wp.sin(phi)
    cos_phi = wp.cos(phi)
    sin_theta = wp.pow(1.0 - x1, 1.0 / (inv_roughness + 2.0))
    cos_theta = wp.sqrt(wp.max(1.0 - sin_theta * sin_theta, 0.0))
    return wp.normalize(wp.vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta))


@wp.func
def _flip_half_vector(h: wp.vec3, k: wp.vec3, xi: wp.float32) -> wp.vec3:
    a = h[2] * k[2]
    b = h[0] * k[0] + h[1] * k[1]
    kh = wp.max(0.0, a + b)
    kh_f = wp.max(0.0, a - b)
    p_flip = kh_f / wp.max(kh + kh_f, 1.0e-12)
    if xi < p_flip:
        return wp.vec3(-h[0], -h[1], h[2])
    return h


# --- pbr_common.h line 311 ---
@wp.func
def _ior_fresnel(eta: wp.float32, kh: wp.float32) -> wp.float32:
    costheta = 1.0 - (1.0 - kh * kh) / (eta * eta)
    if costheta <= 0.0:
        return 1.0
    costheta = wp.sqrt(costheta)
    n1t1 = kh
    n1t2 = costheta
    n2t1 = kh * eta
    n2t2 = costheta * eta
    r_p = (n1t2 - n2t1) / (n1t2 + n2t1)
    r_o = (n1t1 - n2t2) / (n1t1 + n2t2)
    fres = 0.5 * (r_p * r_p + r_o * r_o)
    return wp.clamp(fres, 0.0, 1.0)


@wp.func
def _material_f0(
    base_color: wp.vec3,
    specular_color: wp.vec3,
    specular_scalar: wp.float32,
    metallic: wp.float32,
    ior1: wp.float32,
    ior2: wp.float32,
    clearcoat: wp.float32,
) -> wp.vec3:
    """Return physical normal-incidence reflectance for reconstruction guides."""
    dielectric = specular_color * (_ior_fresnel(ior2 / wp.max(ior1, 1.0e-6), 1.0) * specular_scalar)
    metalness = wp.clamp(metallic, 0.0, 1.0)
    base_f0 = dielectric * (1.0 - metalness) + base_color * metalness
    coat_f0 = wp.clamp(clearcoat, 0.0, 1.0) * _ior_fresnel(1.5 / wp.max(ior1, 1.0e-6), 1.0)
    return wp.vec3(coat_f0) + base_f0 * (1.0 - coat_f0)


# --- pbr_common.h line 522 ---
@wp.func
def _fresnel_cosine_approximation(v_dot_n: wp.float32, roughness: wp.float32) -> wp.float32:
    sr = wp.sqrt(roughness)
    return v_dot_n * (1.0 - sr) + wp.sqrt(0.5 + 0.5 * v_dot_n) * sr


@wp.func
def _build_tangent_frame(n: wp.vec3) -> wp.mat33:
    up = wp.vec3(0.0, 0.0, 1.0) if wp.abs(n[1]) < 0.999 else wp.vec3(1.0, 0.0, 0.0)
    t = wp.normalize(wp.cross(up, n))
    b = wp.cross(n, t)
    return wp.mat33(t[0], t[1], t[2], b[0], b[1], b[2], n[0], n[1], n[2])


@wp.func
def _offset_ray(p: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Matches C++ offsetRay (pbr_common.h line 70)."""
    epsilon = 1.0 / 65536.0
    magnitude = wp.length(p)
    offset = epsilon * magnitude
    return p + n * offset


@wp.func
def _offset_ray_for_direction(p: wp.vec3, n: wp.vec3, direction: wp.vec3) -> wp.vec3:
    """Offset toward the side of the surface containing the outgoing ray."""
    offset_normal = n if wp.dot(direction, n) >= 0.0 else -n
    return _offset_ray(p, offset_normal)


# --- computeLobeWeights + findLobe (pbr_common.h lines 528-593) ---
# 1:1 translation using actual material properties (except iridescence/dispersion).


@wp.func
def _find_lobe(
    v_dot_n: wp.float32,
    roughness: wp.float32,
    metallic: wp.float32,
    specular_scalar: wp.float32,
    ior1: wp.float32,
    ior2: wp.float32,
    transmission: wp.float32,
    diffuse_transmission_factor: wp.float32,
    clearcoat: wp.float32,
    clearcoat_roughness: wp.float32,
    sheen_roughness: wp.float32,
    sheen_color: wp.vec3,
    rnd: wp.float32,
) -> wp.int32:
    # C++ computeLobeWeights lines 531-534: clearcoat
    fr_coat = float(0.0)  # noqa: UP018
    if clearcoat > 0.0:
        fr_cosine_cc = _fresnel_cosine_approximation(v_dot_n, clearcoat_roughness)
        fr_coat = clearcoat * _ior_fresnel(1.5 / ior1, fr_cosine_cc)

    # C++ lines 537-542: dielectric specular
    fr_dielectric = float(0.0)  # noqa: UP018
    if specular_scalar > 0.0:
        fr_cosine_di = _fresnel_cosine_approximation(v_dot_n, roughness)
        fr_dielectric = _ior_fresnel(ior2 / ior1, fr_cosine_di)
        fr_dielectric = fr_dielectric * specular_scalar

    sheen = float(0.0)  # noqa: UP018
    if sheen_color[0] != 0.0 or sheen_color[1] != 0.0 or sheen_color[2] != 0.0:
        sheen = wp.pow(1.0 - wp.abs(v_dot_n), wp.max(sheen_roughness, 0.0))
        sheen = sheen / (sheen + 0.5)

    # C++ lines 566-578: weight accumulation
    weight_base = 1.0 - fr_coat
    w_clearcoat = fr_coat
    w_sheen = weight_base * sheen
    weight_base = weight_base * (1.0 - sheen)
    w_metal = weight_base * metallic
    weight_base = weight_base * (1.0 - metallic)
    w_specular = weight_base * fr_dielectric
    weight_base = weight_base * (1.0 - fr_dielectric)
    w_transmission = weight_base * transmission
    remaining_weight = weight_base * (1.0 - transmission)
    w_diffuse_tx = remaining_weight * diffuse_transmission_factor

    # C++ findLobe lines 585-592: iterate from LOBE_COUNT-1 down to 0
    # LOBE order: DIFFUSE=0, DIFFUSE_TX=1, SPEC_TX=2, SPEC_REFL=3, METAL=4, SHEEN=5, CLEARCOAT=6
    weight = float(0.0)  # noqa: UP018
    weight = weight + w_clearcoat
    if rnd < weight:
        return LOBE_CLEARCOAT_REFLECTION
    weight = weight + w_sheen
    if rnd < weight:
        return LOBE_SHEEN_REFLECTION
    weight = weight + w_metal
    if rnd < weight:
        return LOBE_METAL_REFLECTION
    weight = weight + w_specular
    if rnd < weight:
        return LOBE_SPECULAR_REFLECTION
    weight = weight + w_transmission
    if rnd < weight:
        return LOBE_SPECULAR_TRANSMISSION
    weight = weight + w_diffuse_tx
    if rnd < weight:
        return LOBE_DIFFUSE_TRANSMISSION
    return LOBE_DIFFUSE_REFLECTION


@wp.struct
class BsdfEvalResult:
    bsdf_diffuse: wp.vec3
    bsdf_glossy: wp.vec3
    pdf: wp.float32


# --- bsdfEvaluate (pbr_common.h line 960) ---
# 1:1 translation using material T, B and full lobe weights.


@wp.func
def _bsdf_evaluate(
    to_eye: wp.vec3,
    to_light: wp.vec3,
    normal: wp.vec3,
    mat_Ng: wp.vec3,
    mat_Nc: wp.vec3,
    mat_T: wp.vec3,
    mat_B: wp.vec3,
    base_color: wp.vec3,
    transmission_color: wp.vec3,
    specular_color: wp.vec3,
    roughness: wp.float32,
    metallic: wp.float32,
    specular_scalar: wp.float32,
    sheen_roughness: wp.float32,
    sheen_color: wp.vec3,
    ior1: wp.float32,
    ior2: wp.float32,
    transmission: wp.float32,
    diffuse_transmission_factor: wp.float32,
    diffuse_transmission_color: wp.vec3,
    clearcoat: wp.float32,
    clearcoat_roughness: wp.float32,
    occlusion: wp.float32,
    is_thin_walled: wp.int32,
    xi_z: wp.float32,
) -> BsdfEvalResult:
    result = BsdfEvalResult()
    result.bsdf_diffuse = wp.vec3(0.0, 0.0, 0.0)
    result.bsdf_glossy = wp.vec3(0.0, 0.0, 0.0)
    result.pdf = 0.0

    v_dot_n = wp.dot(to_eye, normal)
    lobe = _find_lobe(
        v_dot_n,
        roughness,
        metallic,
        specular_scalar,
        ior1,
        ior2,
        transmission,
        diffuse_transmission_factor,
        clearcoat,
        clearcoat_roughness,
        sheen_roughness,
        sheen_color,
        xi_z,
    )

    if lobe == LOBE_DIFFUSE_REFLECTION:
        if wp.dot(to_light, mat_Ng) <= 0.0:
            return result
        n_dot_l = wp.dot(to_light, normal)
        tint = base_color
        result.pdf = wp.max(0.0, n_dot_l / wp.pi)
        result.bsdf_diffuse = tint * result.pdf
    elif lobe == LOBE_DIFFUSE_TRANSMISSION:
        n_dot_l = wp.dot(to_light, -normal)
        if n_dot_l <= 0.0 or wp.dot(to_light, -mat_Ng) <= 0.0:
            return result
        tint = wp.vec3(
            base_color[0] * diffuse_transmission_color[0],
            base_color[1] * diffuse_transmission_color[1],
            base_color[2] * diffuse_transmission_color[2],
        )
        result.pdf = wp.max(0.0, n_dot_l / wp.pi)
        result.bsdf_diffuse = tint * result.pdf
    elif lobe == LOBE_SHEEN_REFLECTION:
        if wp.dot(to_light, mat_Ng) <= 0.0:
            return result
        nk1 = wp.abs(wp.dot(to_eye, normal))
        nk2 = wp.abs(wp.dot(to_light, normal))
        h = wp.normalize(to_eye + to_light)
        nh = wp.dot(normal, h)
        k1h = wp.dot(to_eye, h)
        k2h = wp.dot(to_light, h)
        if nk1 <= 0.0 or nk2 <= 0.0 or nh <= 0.0 or k1h < 0.0 or k2h < 0.0:
            return result
        inv_sheen_roughness = 1.0 / wp.max(sheen_roughness * sheen_roughness, 1.0e-6)
        g = _vcavities_shadow_mask(nh, nk1, k1h, nk2, k2h)
        g1 = g[0]
        g12 = g[2]
        result.pdf = _hvd_sheen_eval(inv_sheen_roughness, nh)
        result.pdf = result.pdf * 0.25 / wp.max(nk1 * nh, 1.0e-12)
        bsdf = g12 * result.pdf
        result.pdf = result.pdf * g1
        result.bsdf_glossy = wp.vec3(bsdf * sheen_color[0], bsdf * sheen_color[1], bsdf * sheen_color[2])
    elif lobe == LOBE_SPECULAR_REFLECTION or lobe == LOBE_METAL_REFLECTION or lobe == LOBE_CLEARCOAT_REFLECTION:
        N = normal
        alpha = wp.max(roughness, MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS)
        tint = base_color if lobe == LOBE_METAL_REFLECTION else specular_color
        if lobe == LOBE_CLEARCOAT_REFLECTION:
            N = mat_Nc
            cc_r = wp.max(clearcoat_roughness, 0.001)
            alpha = cc_r * cc_r
            tint = wp.vec3(1.0, 1.0, 1.0)

        nk1 = wp.abs(wp.dot(to_eye, N))
        nk2 = wp.abs(wp.dot(to_light, N))
        h = wp.normalize(to_eye + to_light)
        nh = wp.dot(N, h)
        k1h = wp.dot(to_eye, h)
        k2h = wp.dot(to_light, h)
        if nk1 <= 0.0 or nh <= 0.0 or k1h < 0.0 or k2h < 0.0:
            return result

        inv_alpha = 1.0 / alpha

        h0x = wp.dot(mat_T, h)
        h0y = wp.dot(mat_B, h)
        h0z = nh

        D = _hvd_ggx_eval(inv_alpha, h0x, h0y, h0z)

        k1x = wp.dot(mat_T, to_eye)
        k1y = wp.dot(mat_B, to_eye)
        k2x = wp.dot(mat_T, to_light)
        k2y = wp.dot(mat_B, to_light)

        G1 = _smith_shadow_or_mask(k1x, k1y, nk1, alpha)
        G2 = _smith_shadow_or_mask(k2x, k2y, nk2, alpha)
        G12 = G1 * G2

        result.pdf = D * 0.25 / wp.max(nk1 * nh, 1.0e-12)
        bsdf_scalar = G12 * result.pdf
        result.pdf = result.pdf * G1

        result.bsdf_glossy = tint * bsdf_scalar
    elif lobe == LOBE_SPECULAR_TRANSMISSION:
        tint = transmission_color
        N = normal
        nk1 = wp.abs(wp.dot(to_eye, N))
        nk2 = wp.abs(wp.dot(to_light, N))
        backside = wp.dot(to_light, mat_Ng) < 0.0

        # C++ compute_half_vector: thin-walled uses reflect-flip half-vector
        if is_thin_walled != 0:
            h = wp.normalize(to_eye + (N * (nk2 + nk2) + to_light))
        else:
            h = wp.normalize(to_light * ior2 + to_eye * ior1)
            if ior2 > ior1:
                h = -h
        nh = wp.dot(N, h)
        k1h = wp.dot(to_eye, h)
        k2h = wp.dot(to_light, h) * (-1.0 if backside else 1.0)
        if nk1 <= 0.0 or nh <= 0.0 or k1h < 0.0 or k2h < 0.0:
            return result

        fr = float(0.0)  # noqa: UP018
        if not backside:
            eta = ior1 / wp.max(ior2, 1.0e-6)
            tir = 1.0 < (eta * eta * (1.0 - k1h * k1h))
            if not tir:
                return result
            fr = 1.0

        alpha = wp.max(roughness, MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS)
        inv_alpha = 1.0 / alpha
        h0x = wp.dot(mat_T, h)
        h0y = wp.dot(mat_B, h)
        h0z = nh
        D = _hvd_ggx_eval(inv_alpha, h0x, h0y, h0z)

        k1x = wp.dot(mat_T, to_eye)
        k1y = wp.dot(mat_B, to_eye)
        k2x = wp.dot(mat_T, to_light)
        k2y = wp.dot(mat_B, to_light)
        G1 = _smith_shadow_or_mask(k1x, k1y, nk1, alpha)
        G2 = _smith_shadow_or_mask(k2x, k2y, nk2, alpha)
        G12 = G1 * G2

        result.pdf = D
        if is_thin_walled == 0 and backside:
            tmp = k1h * ior1 - k2h * ior2
            result.pdf = result.pdf * k1h * k2h / wp.max(nk1 * nh * tmp * tmp, 1.0e-12)
        else:
            result.pdf = result.pdf * 0.25 / wp.max(nk1 * nh, 1.0e-12)

        prob = 1.0 - fr if backside else fr
        bsdf_scalar = prob * G12 * result.pdf
        result.pdf = result.pdf * prob * G1
        result.bsdf_glossy = tint * bsdf_scalar

    # C++ bsdfEvaluate lines 1001-1002: apply occlusion
    result.bsdf_diffuse = result.bsdf_diffuse * occlusion
    result.bsdf_glossy = result.bsdf_glossy * occlusion

    return result


@wp.struct
class BsdfSampleResult:
    direction: wp.vec3
    bsdf_over_pdf: wp.vec3
    pdf: wp.float32
    event_type: wp.int32


# --- bsdfSample (pbr_common.h line 1006) ---
# 1:1 translation: findLobe selects lobe, then dispatch to
# brdf_diffuse_sample or brdf_ggx_smith_sample.
# Uses material T, B instead of building tangent frame.


@wp.func
def _bsdf_sample(
    to_eye: wp.vec3,
    normal: wp.vec3,
    mat_Ng: wp.vec3,
    mat_Nc: wp.vec3,
    mat_T: wp.vec3,
    mat_B: wp.vec3,
    base_color: wp.vec3,
    transmission_color: wp.vec3,
    specular_color: wp.vec3,
    roughness: wp.float32,
    metallic: wp.float32,
    specular_scalar: wp.float32,
    sheen_roughness: wp.float32,
    sheen_color: wp.vec3,
    ior1: wp.float32,
    ior2: wp.float32,
    transmission: wp.float32,
    diffuse_transmission_factor: wp.float32,
    diffuse_transmission_color: wp.vec3,
    clearcoat: wp.float32,
    clearcoat_roughness: wp.float32,
    is_thin_walled: wp.int32,
    xi0: wp.float32,
    xi1: wp.float32,
    xi2: wp.float32,
) -> BsdfSampleResult:
    result = BsdfSampleResult()
    result.direction = wp.vec3(0.0, 0.0, 0.0)
    result.bsdf_over_pdf = wp.vec3(0.0, 0.0, 0.0)
    result.pdf = 0.0
    result.event_type = BSDF_EVENT_ABSORB

    v_dot_n = wp.dot(to_eye, normal)
    lobe = _find_lobe(
        v_dot_n,
        roughness,
        metallic,
        specular_scalar,
        ior1,
        ior2,
        transmission,
        diffuse_transmission_factor,
        clearcoat,
        clearcoat_roughness,
        sheen_roughness,
        sheen_color,
        xi2,
    )

    T = mat_T
    B = mat_B
    N = normal

    if lobe == LOBE_DIFFUSE_REFLECTION:
        local_dir = _sample_cosine_hemisphere_local(xi0, xi1)
        L = T * local_dir[0] + B * local_dir[1] + N * local_dir[2]
        L = wp.normalize(L)
        result.pdf = wp.dot(L, N) / wp.pi
        tint = base_color
        result.bsdf_over_pdf = tint
        result.direction = L
        result.event_type = BSDF_EVENT_DIFFUSE_REFLECTION if wp.dot(L, mat_Ng) > 0.0 else BSDF_EVENT_ABSORB
    elif lobe == LOBE_DIFFUSE_TRANSMISSION:
        local_dir = _sample_cosine_hemisphere_local(xi0, xi1)
        L = T * local_dir[0] + B * local_dir[1] - N * local_dir[2]
        L = wp.normalize(L)
        result.pdf = wp.max(0.0, wp.dot(L, -N) / wp.pi)
        result.bsdf_over_pdf = wp.vec3(
            base_color[0] * diffuse_transmission_color[0],
            base_color[1] * diffuse_transmission_color[1],
            base_color[2] * diffuse_transmission_color[2],
        )
        result.direction = L
        result.event_type = BSDF_EVENT_DIFFUSE_TRANSMISSION if wp.dot(L, -mat_Ng) > 0.0 else BSDF_EVENT_ABSORB
    elif lobe == LOBE_SHEEN_REFLECTION:
        inv_sheen_roughness = 1.0 / wp.max(sheen_roughness * sheen_roughness, 1.0e-6)
        nk1 = wp.abs(wp.dot(to_eye, N))
        k10 = wp.vec3(wp.dot(to_eye, T), wp.dot(to_eye, B), nk1)
        h0 = _hvd_sheen_sample(xi0, xi1, inv_sheen_roughness)
        h0 = _flip_half_vector(h0, k10, xi2)
        if wp.abs(h0[2]) == 0.0:
            return result
        H = T * h0[0] + B * h0[1] + N * h0[2]
        k1h = wp.dot(to_eye, H)
        if k1h <= 0.0:
            return result
        L = 2.0 * k1h * H - to_eye
        gnk2 = wp.dot(L, mat_Ng)
        if gnk2 <= 0.0:
            return result
        nk2 = wp.abs(wp.dot(L, N))
        k2h = wp.abs(wp.dot(L, H))
        g = _vcavities_shadow_mask(h0[2], nk1, k1h, nk2, k2h)
        g1 = g[0]
        g12 = g[2]
        if g12 <= 0.0:
            return result
        result.bsdf_over_pdf = wp.vec3(
            g12 / wp.max(g1, 1.0e-12),
            g12 / wp.max(g1, 1.0e-12),
            g12 / wp.max(g1, 1.0e-12),
        )
        result.pdf = _hvd_sheen_eval(inv_sheen_roughness, h0[2]) * g1
        result.pdf = result.pdf * 0.25 / wp.max(nk1 * h0[2], 1.0e-12)
        result.bsdf_over_pdf = wp.vec3(
            result.bsdf_over_pdf[0] * sheen_color[0],
            result.bsdf_over_pdf[1] * sheen_color[1],
            result.bsdf_over_pdf[2] * sheen_color[2],
        )
        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION
        result.direction = L
    elif lobe == LOBE_SPECULAR_REFLECTION or lobe == LOBE_METAL_REFLECTION or lobe == LOBE_CLEARCOAT_REFLECTION:
        if lobe == LOBE_CLEARCOAT_REFLECTION:
            N = mat_Nc
            B = wp.normalize(wp.cross(N, T))
            T = wp.cross(B, N)
            cc_r = wp.max(clearcoat_roughness, 0.001)
            alpha = cc_r * cc_r
            tint = wp.vec3(1.0, 1.0, 1.0)
        else:
            alpha = wp.max(roughness, MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS)
            tint = base_color if lobe == LOBE_METAL_REFLECTION else specular_color

        nk1 = wp.dot(to_eye, N)
        if nk1 <= 0.0:
            return result

        k1x = wp.dot(to_eye, T)
        k1y = wp.dot(to_eye, B)

        h0 = _hvd_ggx_sample_vndf(k1x, k1y, nk1, alpha, xi0, xi1)
        if h0[2] == 0.0:
            return result

        H = T * h0[0] + B * h0[1] + N * h0[2]
        kh = wp.dot(to_eye, H)
        if kh <= 0.0:
            return result

        L = 2.0 * kh * H - to_eye
        gnk2 = wp.dot(L, mat_Ng)
        if gnk2 <= 0.0:
            return result

        nk2 = wp.abs(wp.dot(L, N))
        k2x = wp.dot(L, T)
        k2y = wp.dot(L, B)

        G1 = _smith_shadow_or_mask(k1x, k1y, nk1, alpha)
        G2 = _smith_shadow_or_mask(k2x, k2y, nk2, alpha)
        G12 = G1 * G2
        if G12 <= 0.0:
            return result

        result.bsdf_over_pdf = tint * G2
        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION

        inv_alpha = 1.0 / alpha
        result.pdf = _hvd_ggx_eval(inv_alpha, h0[0], h0[1], h0[2]) * G1 * 0.25 / wp.max(nk1 * h0[2], 1.0e-12)

        result.direction = L
    elif lobe == LOBE_SPECULAR_TRANSMISSION:
        nk1 = wp.abs(wp.dot(to_eye, N))
        if nk1 <= 0.0:
            return result

        alpha = wp.max(roughness, MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS)
        k1x = wp.dot(to_eye, T)
        k1y = wp.dot(to_eye, B)
        # A smooth zero-thickness sheet represents parallel entry/exit
        # interfaces, so its delta transmission remains straight. Rough thin
        # surfaces retain the reference GGX pseudo-BTDF below.
        is_smooth = roughness <= (MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS)
        if is_thin_walled != 0 and is_smooth:
            result.direction = -to_eye
            result.bsdf_over_pdf = transmission_color
            result.pdf = 1.0

            result.event_type = BSDF_EVENT_GLOSSY_TRANSMISSION
            return result
        h0 = _hvd_ggx_sample_vndf(k1x, k1y, nk1, alpha, xi0, xi1)
        if h0[2] == 0.0:
            return result

        H = T * h0[0] + B * h0[1] + N * h0[2]
        kh = wp.dot(to_eye, H)
        if kh <= 0.0:
            return result

        # C++ btdf_ggx_smith_sample: thin-walled reflect-then-flip vs real refraction
        tir = wp.bool(False)
        if is_thin_walled != 0:
            L = 2.0 * kh * H - to_eye
            L = wp.normalize(L - 2.0 * N * wp.dot(L, N))
        else:
            eta = ior1 / wp.max(ior2, 1.0e-6)
            refraction = eta * eta * (1.0 - kh * kh)
            tir = refraction >= 1.0
            if tir:
                L = 2.0 * kh * H - to_eye
            else:
                L = wp.normalize((-to_eye * eta) + H * (eta * kh - wp.sqrt(wp.max(1.0 - refraction, 0.0))))

        result.event_type = BSDF_EVENT_GLOSSY_REFLECTION if tir else BSDF_EVENT_GLOSSY_TRANSMISSION

        gnk2 = wp.dot(L, mat_Ng) * (1.0 if result.event_type == BSDF_EVENT_GLOSSY_REFLECTION else -1.0)
        if gnk2 <= 0.0 or L[0] != L[0]:
            result.event_type = BSDF_EVENT_ABSORB
            return result

        nk2 = wp.abs(wp.dot(L, N))
        k2h = wp.abs(wp.dot(L, H))
        k2x = wp.dot(L, T)
        k2y = wp.dot(L, B)
        G1 = _smith_shadow_or_mask(k1x, k1y, nk1, alpha)
        G2 = _smith_shadow_or_mask(k2x, k2y, nk2, alpha)
        G12 = G1 * G2
        if G12 <= 0.0:
            result.event_type = BSDF_EVENT_ABSORB
            return result

        result.bsdf_over_pdf = transmission_color * G2
        inv_alpha = 1.0 / alpha
        result.pdf = _hvd_ggx_eval(inv_alpha, h0[0], h0[1], h0[2]) * G1
        if is_thin_walled == 0 and result.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION:
            tmp = kh * ior1 - k2h * ior2
            if tmp > 0.0:
                result.pdf = result.pdf * kh * k2h / wp.max(nk1 * h0[2] * tmp * tmp, 1.0e-12)
        else:
            result.pdf = result.pdf * 0.25 / wp.max(nk1 * h0[2], 1.0e-12)
        result.direction = L

    # C++ bsdfSample lines 1037-1042: post-sample absorb/DIRAC guard
    bop = result.bsdf_over_pdf
    if result.pdf <= 0.00001 or bop[0] != bop[0] or bop[1] != bop[1] or bop[2] != bop[2]:
        result.event_type = BSDF_EVENT_ABSORB
    if (result.pdf != result.pdf or wp.abs(result.pdf) > 1.0e30) and result.event_type != BSDF_EVENT_ABSORB:
        result.event_type = (result.event_type & (~BSDF_EVENT_GLOSSY)) | BSDF_EVENT_IMPULSE
        result.pdf = DIRAC

    return result


# cosineSampleHemisphere (func_common.h line 428) — returns local tangent-space direction
@wp.func
def _sample_cosine_hemisphere_local(r1: wp.float32, r2: wp.float32) -> wp.vec3:
    r = wp.sqrt(r1)
    phi = 2.0 * wp.pi * r2
    return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), wp.sqrt(wp.max(1.0 - r1, 0.0)))


@wp.struct
class ShadedHitData:
    """Mirrors C++ PbrMaterial fields needed for rendering."""

    valid: wp.uint32
    color: wp.vec3  # emissive
    normal: wp.vec3  # N (shading normal, possibly from normal map)
    Ng: wp.vec3  # geometric normal (for ray offset)
    T: wp.vec3  # tangent
    B: wp.vec3  # bitangent
    roughness: wp.float32
    diffuse: wp.vec3  # baseColor
    specular: wp.vec3  # specularColor from material (NOT F0)
    specular_scalar: wp.float32  # mat.specular
    sheen_roughness: wp.float32
    diffuse_transmission_factor: wp.float32
    t_hit: wp.float32
    transmission_color: wp.vec3
    spec_hit_dist: wp.float32
    metallic: wp.float32
    opacity: wp.float32
    opacity_fresnel_low: wp.float32
    opacity_fresnel_high: wp.float32
    opacity_fresnel_falloff: wp.float32
    transmission: wp.float32
    ior1: wp.float32
    ior2: wp.float32
    clearcoat: wp.float32
    clearcoat_roughness: wp.float32
    occlusion: wp.float32
    is_thin_walled: wp.int32
    sheen_color: wp.vec3
    diffuse_transmission_color: wp.vec3
    Nc: wp.vec3  # clearcoat normal


@wp.func
def _make_invalid_shaded_hit() -> ShadedHitData:
    d = ShadedHitData()
    d.valid = wp.uint32(0)
    d.color = wp.vec3(0.0, 0.0, 0.0)
    d.normal = wp.vec3(0.0, 0.0, 1.0)
    d.Ng = wp.vec3(0.0, 0.0, 1.0)
    d.T = wp.vec3(1.0, 0.0, 0.0)
    d.B = wp.vec3(0.0, 1.0, 0.0)
    d.roughness = 1.0
    d.diffuse = wp.vec3(0.0, 0.0, 0.0)
    d.specular = wp.vec3(1.0, 1.0, 1.0)
    d.specular_scalar = 1.0
    d.sheen_roughness = 0.0
    d.diffuse_transmission_factor = 0.0
    d.t_hit = 65504.0
    d.spec_hit_dist = 0.0
    d.transmission_color = wp.vec3(1.0, 1.0, 1.0)
    d.metallic = 0.0
    d.opacity = 1.0
    d.opacity_fresnel_low = -1.0
    d.opacity_fresnel_high = 1.0
    d.opacity_fresnel_falloff = 1.0
    d.transmission = 0.0
    d.ior1 = 1.0
    d.ior2 = 1.5
    d.clearcoat = 0.0
    d.clearcoat_roughness = 0.01
    d.occlusion = 1.0
    d.is_thin_walled = 1
    d.sheen_color = wp.vec3(0.0, 0.0, 0.0)
    d.diffuse_transmission_color = wp.vec3(1.0, 1.0, 1.0)
    d.Nc = wp.vec3(0.0, 0.0, 1.0)
    return d


@wp.func
def _set_incident_medium_for_hit(hit: ShadedHitData, front_face: wp.uint32) -> ShadedHitData:
    """Select the incident medium after geometry has face-forwarded its frame."""
    if hit.valid != wp.uint32(0) and front_face == wp.uint32(0) and hit.is_thin_walled == 0:
        exterior_ior = hit.ior1
        hit.ior1 = hit.ior2
        hit.ior2 = exterior_ior
    return hit


@wp.func
def _sample_sphere_light_contribution(
    params: PathtraceLaunchParams,
    position: wp.vec3,
    material: ShadedHitData,
    to_eye: wp.vec3,
    base_color: wp.vec3,
    specular_color: wp.vec3,
    xi0: wp.float32,
    xi1: wp.float32,
    xi2: wp.float32,
    xi_lobe: wp.float32,
) -> wp.vec3:
    """Evaluate one shadowed next-event sample from the analytic light set."""
    contribution = wp.vec3(0.0, 0.0, 0.0)
    if params.sphere_light_count == wp.uint32(0) or params.analytic_light_intensity <= 0.0:
        return contribution
    sample = _sample_sphere_light(params, position, xi0, xi1, xi2)
    if (
        sample.pdf <= 1.0e-8
        or sample.distance <= 0.002
        or (wp.dot(sample.direction, material.Ng) <= 0.0 and material.transmission <= 0.0)
    ):
        return contribution

    evaluated = _bsdf_evaluate(
        to_eye,
        sample.direction,
        material.normal,
        material.Ng,
        material.Nc,
        material.T,
        material.B,
        base_color,
        material.transmission_color,
        specular_color,
        material.roughness,
        material.metallic,
        material.specular_scalar,
        material.sheen_roughness,
        material.sheen_color,
        material.ior1,
        material.ior2,
        material.transmission,
        material.diffuse_transmission_factor,
        material.diffuse_transmission_color,
        material.clearcoat,
        material.clearcoat_roughness,
        material.occlusion,
        material.is_thin_walled,
        xi_lobe,
    )
    if evaluated.pdf <= 1.0e-8:
        return contribution

    shadow = ShadowPayload()
    shadow.visible = wp.uint32(0)
    shadow.seed = wp.uint32(0)
    wp.optix_trace(
        params.tlas,
        _offset_ray_for_direction(position, material.Ng, sample.direction),
        sample.direction,
        0.001,
        wp.max(sample.distance - 0.001, 0.001),
        0.0,
        wp.uint32(255),
        wp.uint32(OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT),
        wp.uint32(1),
        wp.uint32(2),
        wp.uint32(1),
        shadow,
    )
    if shadow.visible == wp.uint32(1):
        bsdf = evaluated.bsdf_diffuse + evaluated.bsdf_glossy
        mis_weight = _power_heuristic(sample.pdf, evaluated.pdf)
        contribution = _mul_vec3(bsdf, sample.radiance) * (mis_weight / sample.pdf)
    return contribution


@wp.func
def _filter_roughness_for_normal_map(microfacet_roughness: wp.float32, tangent_normal: wp.vec3) -> wp.float32:
    """Broaden the GGX lobe by unresolved tangent-normal variance."""
    normal_length = wp.clamp(wp.length(tangent_normal), 0.0, 1.0)
    variance = 1.0 - normal_length
    return wp.clamp(microfacet_roughness + variance, 0.0, 1.0)


@wp.func
def _evaluate_material_from_payload(
    params: PathtraceLaunchParams,
    material_id: wp.int32,
    normal: wp.vec3,
    tangent: wp.vec3,
    bitangent_sign: wp.float32,
    uv: wp.vec2,
    uv1: wp.vec2,
    texture_lod: wp.float32,
) -> ShadedHitData:
    """Evaluate PBR material from geometry payload data (matches C++ evaluate_pbr_from_payload 1:1)."""
    out = _make_invalid_shaded_hit()
    if material_id < 0 or material_id >= int(params.material_count):
        return out

    # C++ lines 51-60: build tangent frame from geometry
    n = wp.normalize(normal)
    # Keep the unperturbed surface normal separate from the normal-mapped
    # shading normal.  Ng is used for geometric hemisphere tests and ray
    # offsets; replacing it with the tangent-space normal can reject valid
    # lighting samples and self-intersect strongly normal-mapped surfaces.
    # This matches mat_eval_common.glsl in the reference C# renderer, which
    # assigns pbrMat.Ng from state.Ng before changing only pbrMat.N.
    ng = n
    t = tangent - n * wp.dot(n, tangent)
    t_len_sq = wp.dot(t, t)
    if t_len_sq < 1.0e-12:
        up = wp.vec3(0.0, 0.0, 1.0) if wp.abs(n[2]) < 0.999 else wp.vec3(0.0, 1.0, 0.0)
        t = wp.cross(up, n)
    t = wp.normalize(t)
    bs = bitangent_sign
    if bs == 0.0:
        bs = 1.0
    b = wp.normalize(wp.cross(n, t)) * bs

    materials = params.compact_materials
    mat = materials[material_id]

    # C++ line 124: outMat = defaultPbrMaterial(baseColor, metallic, roughness, n, n)
    # defaultPbrMaterial squares roughness: roughness * roughness
    base_color = mat.base_color
    metallic = wp.clamp(mat.metallic, 0.0, 1.0)
    raw_roughness = wp.max(wp.clamp(mat.roughness, 0.0, 1.0), MICROFACET_MIN_ROUGHNESS)
    roughness_sq = raw_roughness * raw_roughness
    opacity = wp.clamp(mat.opacity, 0.0, 1.0)
    emissive = (
        wp.vec3(
            wp.max(mat.emissive[0], 0.0),
            wp.max(mat.emissive[1], 0.0),
            wp.max(mat.emissive[2], 0.0),
        )
        * params.emissive_material_intensity
    )

    # C++ lines 125-136: set material fields from compact material
    transmission = wp.clamp(mat.transmission, 0.0, 1.0)
    ior2 = wp.max(mat.ior, 1.0)
    specular_scalar = wp.max(mat.specular, 0.0)
    specular_color = mat.specular_color
    clearcoat = wp.max(mat.clearcoat, 0.0)
    clearcoat_roughness = wp.max(mat.clearcoat_roughness, 0.001)
    # Strict template parity: compact-material path does not carry these fields.
    sheen_roughness = 0.0
    sheen_color = wp.vec3(0.0, 0.0, 0.0)
    diffuse_transmission_factor = 0.0
    diffuse_transmission_color = wp.vec3(1.0, 1.0, 1.0)

    # C++ lines 143-151: apply base color texture
    uv_base = _apply_uv_transform(_select_uv(mat.base_color_tex_coord, uv, uv1), mat.base_color_uv_transform)
    if mat.base_color_tex_index >= 0:
        base_tex = _sample_texture_rgba(params, mat.base_color_tex_index, uv_base, texture_lod)
        base_color = wp.vec3(
            base_color[0] * base_tex[0],
            base_color[1] * base_tex[1],
            base_color[2] * base_tex[2],
        )
        opacity = opacity * base_tex[3]

    if mat.base_color_desaturation > 0.0:
        luminance = 0.2126 * base_color[0] + 0.7152 * base_color[1] + 0.0722 * base_color[2]
        amount = wp.clamp(mat.base_color_desaturation, 0.0, 1.0)
        base_color = base_color * (1.0 - amount) + wp.vec3(luminance) * amount
    if mat.base_color_add != 0.0:
        base_color = base_color + wp.vec3(mat.base_color_add)
        base_color = wp.vec3(
            wp.max(base_color[0], 0.0),
            wp.max(base_color[1], 0.0),
            wp.max(base_color[2], 0.0),
        )

    transmission_color = mat.transmission_color
    if transmission_color[0] < 0.0:
        transmission_color = base_color

    if mat.alpha_mode == wp.int32(0):
        opacity = 1.0
    elif mat.alpha_mode == wp.int32(1):
        opacity = 1.0 if opacity >= mat.alpha_cutoff else 0.0

    if mat.u_subdiv > 0.0 and mat.v_subdiv > 0.0:
        checker_u = wp.floor(uv_base[0] * mat.u_subdiv)
        checker_v = wp.floor(uv_base[1] * mat.v_subdiv)
        checker_sum = checker_u + checker_v
        checker = checker_sum - wp.floor(checker_sum * 0.5) * 2.0
        if checker >= 1.0:
            base_color = base_color * wp.clamp(mat.base_color_scale, 0.0, 1.0)

    # C++ lines 152-161: apply metallic-roughness texture
    uv_mr = _apply_uv_transform(
        _select_uv(mat.metallic_roughness_tex_coord, uv, uv1),
        mat.metallic_roughness_uv_transform,
    )
    mr_tex = wp.vec4(1.0)
    if mat.metallic_roughness_tex_index >= 0:
        mr_tex = _sample_texture_rgba(params, mat.metallic_roughness_tex_index, uv_mr, texture_lod)
        # C++ line 158: r = max(sqrt(max(outMat.roughness.x, 0)) * mr.y, MIN)
        # outMat.roughness.x is roughness_sq at this point
        r_linear = wp.max(wp.sqrt(wp.max(roughness_sq, 0.0)) * mr_tex[1], MICROFACET_MIN_ROUGHNESS)
        roughness_sq = r_linear * r_linear
        metallic = wp.clamp(metallic * mr_tex[2], 0.0, 1.0)

    occlusion = 1.0
    if mat.occlusion_tex_index >= 0:
        ao = mr_tex[0]
        if mat.occlusion_tex_index != mat.metallic_roughness_tex_index:
            uv_ao = _apply_uv_transform(_select_uv(mat.occlusion_tex_coord, uv, uv1), mat.occlusion_uv_transform)
            ao = _sample_texture_rgba(params, mat.occlusion_tex_index, uv_ao, texture_lod)[0]
        occlusion = 1.0 + wp.clamp(mat.occlusion, 0.0, 1.0) * (ao - 1.0)

    if params.override_roughness > 0.0:
        r_override = wp.clamp(params.override_roughness, MICROFACET_MIN_ROUGHNESS, 1.0)
        roughness_sq = r_override * r_override
    if params.override_metallic > 0.0:
        metallic = wp.clamp(params.override_metallic, 0.0, 1.0)

    # C++ lines 163-174: apply the normal map to N; Ng remains geometric.
    needs_tangent_update = wp.bool(False)
    uv_n = _apply_uv_transform(_select_uv(mat.normal_tex_coord, uv, uv1), mat.normal_uv_transform)
    if mat.normal_tex_index >= 0:
        n_tex = _sample_texture_rgba(params, mat.normal_tex_index, uv_n, texture_lod)
        n_tan = wp.vec3(n_tex[0] * 2.0 - 1.0, n_tex[1] * 2.0 - 1.0, n_tex[2] * 2.0 - 1.0)
        n_tan = wp.vec3(
            n_tan[0] * mat.normal_scale[0],
            n_tan[1] * mat.normal_scale[1],
            n_tan[2],
        )
        roughness_sq = _filter_roughness_for_normal_map(roughness_sq, n_tan)
        n = wp.normalize(t * n_tan[0] + b * n_tan[1] + n * n_tan[2])
        needs_tangent_update = wp.bool(True)

    # C++ lines 176-184: apply clearcoat normal map
    Nc = n
    uv_cc = _apply_uv_transform(
        _select_uv(mat.clearcoat_normal_tex_coord, uv, uv1),
        mat.clearcoat_normal_uv_transform,
    )
    if mat.clearcoat_normal_tex_index >= 0:
        cc_tex = _sample_texture_rgba(params, mat.clearcoat_normal_tex_index, uv_cc, texture_lod)
        cc_tan = wp.vec3(cc_tex[0] * 2.0 - 1.0, cc_tex[1] * 2.0 - 1.0, cc_tex[2] * 2.0 - 1.0)
        Nc = wp.normalize(t * cc_tan[0] + b * cc_tan[1] + Nc * cc_tan[2])

    # C++ lines 185-191: re-orthogonalize tangent frame after normal map
    if needs_tangent_update:
        b_new = wp.cross(n, t)
        bsign_new = -1.0 if wp.dot(b, b_new) < 0.0 else 1.0
        b = b_new * bsign_new
        t = wp.cross(b, n) * bsign_new

    # C++ lines 192-199: apply emissive texture
    if params.emissive_material_intensity > 0.0 and mat.emissive_tex_index >= 0:
        uv_e = _apply_uv_transform(_select_uv(mat.emissive_tex_coord, uv, uv1), mat.emissive_uv_transform)
        e_tex = _sample_texture_rgba(params, mat.emissive_tex_index, uv_e, texture_lod)
        emissive = wp.vec3(
            emissive[0] * e_tex[0],
            emissive[1] * e_tex[1],
            emissive[2] * e_tex[2],
        )
    out.valid = wp.uint32(1)
    out.color = emissive
    out.normal = n
    out.Ng = ng
    out.T = t
    out.B = b
    out.roughness = roughness_sq
    out.diffuse = base_color
    out.specular = specular_color
    out.specular_scalar = specular_scalar
    out.sheen_roughness = sheen_roughness
    out.diffuse_transmission_factor = diffuse_transmission_factor
    out.transmission_color = transmission_color
    out.t_hit = 0.0
    out.spec_hit_dist = 0.0
    out.metallic = metallic
    out.opacity = opacity
    out.opacity_fresnel_low = mat.opacity_fresnel_low
    out.opacity_fresnel_high = mat.opacity_fresnel_high
    out.opacity_fresnel_falloff = mat.opacity_fresnel_falloff
    out.transmission = transmission
    out.ior1 = 1.0
    out.ior2 = ior2
    out.clearcoat = clearcoat
    out.clearcoat_roughness = clearcoat_roughness
    out.occlusion = occlusion
    out.is_thin_walled = mat.is_thin_walled
    out.sheen_color = sheen_color
    out.diffuse_transmission_color = diffuse_transmission_color
    out.Nc = Nc
    return out


@wp.kernel(module="unique")
def primary_raygen(params: PathtraceLaunchParams):
    launch_idx = wp.optix_get_launch_index()
    x = int(launch_idx[0])
    y = int(launch_idx[1])

    width = int(params.width)
    height = int(params.height)
    if x >= width or y >= height:
        return

    rng = _xxhash32(wp.uint32(x), wp.uint32(y), params.frame_index)

    ray_origin = _compute_ray_origin(params)
    org_dir = _compute_ray_dir(params, x, y)

    origin = ray_origin
    direction = org_dir

    ray_flags = _backface_culling_ray_flag(params)
    payload = _init_primary_payload()
    wp.optix_trace(
        params.tlas,
        origin,
        direction,
        0.01,
        1.0e32,
        0.0,
        wp.uint32(255),
        ray_flags,
        wp.uint32(0),
        wp.uint32(2),
        wp.uint32(0),
        payload,
    )

    hitT = _payload_get_hitT(payload)
    hit_sky = wp.bool(hitT >= DLSS_INF_DISTANCE)

    # PSR state tracking (matches C++ psrMirror / psrHitDist / isPsr / psrThroughput / psrDirectRadiance).
    is_psr = wp.bool(False)
    psr_hit_dist = wp.float32(0.0)
    found_opaque_hit = wp.bool(False)
    psr_mirror = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    psr_throughput = wp.vec3(1.0, 1.0, 1.0)
    psr_direct_radiance = wp.vec3(0.0, 0.0, 0.0)
    hit_pos = wp.vec3(0.0, 0.0, 0.0)
    hit_instance_id = wp.int32(-1)
    hit_primitive_id = wp.uint32(0)
    hit_barycentrics = wp.vec3(0.0, 0.0, 0.0)
    pbr = _make_invalid_shaded_hit()

    psr_depth = wp.int32(0)
    max_psr_depth = wp.int32(5)
    while psr_depth < max_psr_depth:
        hitT = _payload_get_hitT(payload)
        hit_sky = wp.bool(hitT >= DLSS_INF_DISTANCE)

        if hit_sky:
            sky_color = _payload_get_normal(payload)
            psr_direct_radiance = psr_direct_radiance + _mul_vec3(psr_throughput, sky_color)
            break

        psr_hit_dist = psr_hit_dist + hitT

        hit_normal = _payload_get_normal(payload)
        hit_tangent = _payload_get_tangent(payload)
        hit_uv = _payload_get_uv(payload)
        hit_uv1 = _payload_get_uv1(payload)
        hit_mat_id = wp.int32(_payload_get_materialId(payload))
        hit_bsign = _payload_get_bitangentSign(payload)
        hit_instance_id = _payload_get_instanceId(payload)
        hit_primitive_id = _payload_get_primitiveId(payload)
        hit_barycentrics = _payload_get_barycentrics(payload)
        hit_pos = origin + direction * hitT

        pbr = _evaluate_material_from_payload(
            params,
            hit_mat_id,
            hit_normal,
            hit_tangent,
            hit_bsign,
            hit_uv,
            hit_uv1,
            hit_barycentrics[0],
        )
        pbr = _set_incident_medium_for_hit(pbr, _payload_get_front_face(payload))

        # C++ line 253: origin = offsetRay(hitPos, pbrMat.Ng) — use geometric normal
        origin = _offset_ray(hit_pos, pbr.Ng)

        # C++ lines 273-277: non-mirror surface check
        use_psr = (params.flags & wp.uint32(2)) != wp.uint32(0)
        is_mirror_surface = wp.bool(False)
        if pbr.valid == wp.uint32(1) and use_psr:
            is_mirror_surface = wp.bool(
                pbr.roughness <= (MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS + 0.001) and pbr.metallic >= 1.0
            )

        if not is_mirror_surface or pbr.valid == wp.uint32(0):
            found_opaque_hit = wp.bool(True)
            break

        # C++ lines 280-308: mirror hit — accumulate PSR chain
        is_psr = wp.bool(True)
        found_opaque_hit = wp.bool(True)
        psr_direct_radiance = psr_direct_radiance + _mul_vec3(psr_throughput, pbr.color)

        rng = _pcg_advance(rng)
        psr_xi0 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        psr_xi1 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        psr_xi2 = _pcg_rand01(rng)
        spec_sample = _bsdf_sample(
            -direction,
            pbr.normal,
            pbr.Ng,
            pbr.Nc,
            pbr.T,
            pbr.B,
            pbr.diffuse,
            pbr.transmission_color,
            pbr.specular,
            pbr.roughness,
            pbr.metallic,
            pbr.specular_scalar,
            pbr.sheen_roughness,
            pbr.sheen_color,
            pbr.ior1,
            pbr.ior2,
            pbr.transmission,
            pbr.diffuse_transmission_factor,
            pbr.diffuse_transmission_color,
            pbr.clearcoat,
            pbr.clearcoat_roughness,
            pbr.is_thin_walled,
            psr_xi0,
            psr_xi1,
            psr_xi2,
        )
        if spec_sample.event_type != BSDF_EVENT_GLOSSY_REFLECTION:
            break
        bop = spec_sample.bsdf_over_pdf
        if bop[0] != bop[0] or bop[1] != bop[1] or bop[2] != bop[2]:
            break

        psr_throughput = _mul_vec3(psr_throughput, bop)

        n_psr = pbr.normal
        nx = n_psr[0]
        ny = n_psr[1]
        nz = n_psr[2]
        mirror = wp.mat33(
            1.0 - 2.0 * nx * nx,
            -2.0 * nx * ny,
            -2.0 * nx * nz,
            -2.0 * ny * nx,
            1.0 - 2.0 * ny * ny,
            -2.0 * ny * nz,
            -2.0 * nz * nx,
            -2.0 * nz * ny,
            1.0 - 2.0 * nz * nz,
        )
        psr_mirror = psr_mirror * mirror

        direction = wp.normalize(spec_sample.direction)
        psr_depth = psr_depth + wp.int32(1)

        payload = _init_primary_payload()
        wp.optix_trace(
            params.tlas,
            origin,
            direction,
            0.01,
            1.0e32,
            0.0,
            wp.uint32(255),
            ray_flags,
            wp.uint32(0),
            wp.uint32(2),
            wp.uint32(0),
            payload,
        )

    # Virtual origin for PSR depth computation (matches C++ reference).
    virtual_origin = ray_origin + org_dir * psr_hit_dist
    view_depth = _compute_view_z(params.view, virtual_origin)

    # DLSS auxiliary defaults.
    aux_view_z = wp.float32(DLSS_INF_DISTANCE)
    aux_motion = wp.vec2(0.0, 0.0)
    aux_normal_roughness = wp.vec4(0.0, 0.0, 0.0, 0.0)
    aux_diffuse_albedo = wp.vec4(0.0, 0.0, 0.0, 0.0)
    aux_specular_albedo = wp.vec4(0.0, 0.0, 0.0, 0.0)
    aux_spec_hit_dist = wp.float32(0.0)

    pixel_center = _compute_pixel_center(params, x, y)
    dim = wp.vec2(wp.float32(width), wp.float32(height))

    # Sky hit path (matches C++ early-out for hitSky).
    if hit_sky:
        sky_guide = _reinhard_max(psr_direct_radiance)
        aux_diffuse_albedo = wp.vec4(sky_guide[0], sky_guide[1], sky_guide[2], 0.0)

        if is_psr:
            aux_view_z = view_depth
            motion_origin = wp.vec4(virtual_origin[0], virtual_origin[1], virtual_origin[2], 1.0)
        else:
            aux_view_z = wp.float32(DLSS_INF_DISTANCE)
            motion_origin = wp.vec4(org_dir[0], org_dir[1], org_dir[2], 0.0)
        aux_motion = _compute_camera_motion_vector(params, pixel_center, motion_origin, dim)

        params.color_output[y, x] = wp.vec4(psr_direct_radiance[0], psr_direct_radiance[1], psr_direct_radiance[2], 1.0)
        params.normal_roughness_output[y, x] = aux_normal_roughness
        params.motion_output[y, x] = aux_motion
        params.depth_output[y, x] = aux_view_z
        params.diffuse_output[y, x] = aux_diffuse_albedo
        params.specular_output[y, x] = aux_specular_albedo
        params.spec_hit_dist_output[y, x] = aux_spec_hit_dist
        return

    # No opaque hit found after PSR chain (matches C++ !foundOpaqueHit path).
    if not found_opaque_hit:
        params.color_output[y, x] = wp.vec4(psr_direct_radiance[0], psr_direct_radiance[1], psr_direct_radiance[2], 1.0)
        params.normal_roughness_output[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        params.motion_output[y, x] = wp.vec2(0.0, 0.0)
        params.depth_output[y, x] = wp.float32(DLSS_INF_DISTANCE)
        params.diffuse_output[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        params.specular_output[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        params.spec_hit_dist_output[y, x] = wp.float32(0.0)
        return

    # The PSR traversal already evaluated the terminal surface. Reuse it here;
    # evaluating again repeats every material texture fetch for every pixel.
    pbr_mat = pbr

    # Guides describe the first camera-visible surface. Background geometry
    # behind moving glass has different motion and causes flowing history.
    # Keep the glass surface's stable depth, normal, and object motion.
    guide_pbr = pbr_mat
    guide_hit_pos = hit_pos
    guide_instance_id = hit_instance_id
    guide_primitive_id = hit_primitive_id
    guide_barycentrics = hit_barycentrics
    guide_view_z = view_depth

    aux_view_z = guide_view_z

    # Normal/Roughness buffer - transform through PSR mirror chain.
    world_normal = wp.normalize(psr_mirror * guide_pbr.normal)
    aux_normal_roughness = wp.vec4(
        world_normal[0],
        world_normal[1],
        world_normal[2],
        wp.sqrt(wp.max(guide_pbr.roughness, 0.0)),
    )

    # Tint material by accumulated PSR mirror throughput (matches C++ lines 388-391).
    base_color = _mul_vec3(pbr_mat.diffuse, psr_throughput)
    guide_base_color = _mul_vec3(guide_pbr.diffuse, psr_throughput)
    specular_color = _mul_vec3(pbr_mat.specular, psr_throughput)
    emissive_tinted = _mul_vec3(pbr_mat.color, psr_throughput) + psr_direct_radiance

    # Motion Vector Buffer (matches C++ reference behavior).
    if is_psr:
        aux_motion = _compute_camera_motion_vector(
            params,
            pixel_center,
            wp.vec4(virtual_origin[0], virtual_origin[1], virtual_origin[2], 1.0),
            dim,
        )
    else:
        aux_motion = _compute_object_motion_vector(
            params,
            pixel_center,
            guide_hit_pos,
            guide_instance_id,
            int(guide_primitive_id),
            guide_barycentrics[1],
            guide_barycentrics[2],
            dim,
        )

    # BaseColor/Metalness buffer (matches C++ auxDiffuseAlbedo = baseColor + metallic).
    aux_diffuse_albedo = wp.vec4(
        guide_base_color[0],
        guide_base_color[1],
        guide_base_color[2],
        guide_pbr.metallic,
    )

    # Direct lighting at primary hit (Step 2 - matches C++ HdrContrib with GGX BSDF).
    to_eye = -direction
    # Omniverse's sceneDb ambient light is diffuse irradiance, separate from
    # the environment/DomeLight. AO modulates it and metals have no diffuse
    # lobe in the glTF metallic-roughness model.
    hdr_radiance = _mul_vec3(base_color, params.ambient_light) * ((1.0 - pbr_mat.metallic) * pbr_mat.occlusion)
    rng = _pcg_advance(rng)
    xi0_l = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    xi1_l = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    xi2_l = _pcg_rand01(rng)
    ls = _sample_environment_light(params, xi0_l, xi1_l, xi2_l)
    if ls.pdf > 1.0e-6 and (wp.dot(ls.direction, pbr_mat.normal) > 0.0 or pbr_mat.transmission > 0.0):
        rng = _pcg_advance(rng)
        eval_xi_z = _pcg_rand01(rng)
        bsdf_eval = _bsdf_evaluate(
            to_eye,
            ls.direction,
            pbr_mat.normal,
            pbr_mat.Ng,
            pbr_mat.Nc,
            pbr_mat.T,
            pbr_mat.B,
            base_color,
            pbr_mat.transmission_color,
            specular_color,
            pbr_mat.roughness,
            pbr_mat.metallic,
            pbr_mat.specular_scalar,
            pbr_mat.sheen_roughness,
            pbr_mat.sheen_color,
            pbr_mat.ior1,
            pbr_mat.ior2,
            pbr_mat.transmission,
            pbr_mat.diffuse_transmission_factor,
            pbr_mat.diffuse_transmission_color,
            pbr_mat.clearcoat,
            pbr_mat.clearcoat_roughness,
            pbr_mat.occlusion,
            pbr_mat.is_thin_walled,
            eval_xi_z,
        )
        if bsdf_eval.pdf > 1.0e-6:
            shadow = ShadowPayload()
            shadow.visible = wp.uint32(0)
            shadow.seed = wp.uint32(0)
            shadow_origin = _offset_ray(hit_pos, pbr_mat.Ng)
            wp.optix_trace(
                params.tlas,
                shadow_origin,
                ls.direction,
                0.001,
                DLSS_INF_DISTANCE,
                0.0,
                wp.uint32(255),
                wp.uint32(
                    wp.uint32(OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT)
                    | _backface_culling_ray_flag(params)
                ),
                wp.uint32(1),
                wp.uint32(2),
                wp.uint32(1),
                shadow,
            )
            if shadow.visible == wp.uint32(1):
                mis_weight = _power_heuristic(ls.pdf, bsdf_eval.pdf)
                bsdf_sum = bsdf_eval.bsdf_diffuse + bsdf_eval.bsdf_glossy
                hdr_radiance = hdr_radiance + _mul_vec3(bsdf_sum, ls.radiance) * (mis_weight / ls.pdf)

    rng = _pcg_advance(rng)
    sphere_xi0 = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    sphere_xi1 = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    sphere_xi2 = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    sphere_lobe_xi = _pcg_rand01(rng)
    hdr_radiance = hdr_radiance + _sample_sphere_light_contribution(
        params,
        hit_pos,
        pbr_mat,
        to_eye,
        base_color,
        specular_color,
        sphere_xi0,
        sphere_xi1,
        sphere_xi2,
        sphere_lobe_xi,
    )

    if hdr_radiance[0] != hdr_radiance[0] or hdr_radiance[1] != hdr_radiance[1] or hdr_radiance[2] != hdr_radiance[2]:
        hdr_radiance = wp.vec3(0.0, 0.0, 0.0)

    # directLum = psrDirectRadiance + pbrMat.emissive (matches C++ line 508, where
    # pbrMat.emissive was already set to emissive*psrThroughput + psrDirectRadiance on line 391).
    direct_lum = psr_direct_radiance + emissive_tinted

    # Step 3 - Indirect contribution (path tracing from primary surface).
    radiance = hdr_radiance
    path_length = wp.float32(0.0)

    # Step 3.1 - Sample BSDF direction using GGX VNDF.
    rng = _pcg_advance(rng)
    xi_bsdf0 = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    xi_bsdf1 = _pcg_rand01(rng)
    rng = _pcg_advance(rng)
    xi_bsdf2 = _pcg_rand01(rng)
    bsdf_sample = _bsdf_sample(
        to_eye,
        pbr_mat.normal,
        pbr_mat.Ng,
        pbr_mat.Nc,
        pbr_mat.T,
        pbr_mat.B,
        base_color,
        pbr_mat.transmission_color,
        specular_color,
        pbr_mat.roughness,
        pbr_mat.metallic,
        pbr_mat.specular_scalar,
        pbr_mat.sheen_roughness,
        pbr_mat.sheen_color,
        pbr_mat.ior1,
        pbr_mat.ior2,
        pbr_mat.transmission,
        pbr_mat.diffuse_transmission_factor,
        pbr_mat.diffuse_transmission_color,
        pbr_mat.clearcoat,
        pbr_mat.clearcoat_roughness,
        pbr_mat.is_thin_walled,
        xi_bsdf0,
        xi_bsdf1,
        xi_bsdf2,
    )

    # NaN/Inf guard on bsdf_over_pdf (matches C++ line 526-527).
    bop0 = bsdf_sample.bsdf_over_pdf
    if bop0[0] != bop0[0] or bop0[1] != bop0[1] or bop0[2] != bop0[2]:
        bsdf_sample.event_type = BSDF_EVENT_ABSORB

    sec_direction = bsdf_sample.direction
    sec_throughput = bsdf_sample.bsdf_over_pdf
    bsdf_pdf = wp.max(bsdf_sample.pdf, 0.0001)
    sec_event_type = bsdf_sample.event_type
    is_glossy_reflection = wp.bool(bsdf_sample.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
    bsdf_absorbed = wp.bool(bsdf_sample.event_type == BSDF_EVENT_ABSORB)

    sec_origin = _offset_ray_for_direction(hit_pos, pbr_mat.Ng, sec_direction)
    max_depth = wp.int32(params.max_bounces) if params.max_bounces > wp.uint32(0) else wp.int32(1)
    depth = wp.int32(1)
    use_path_reg = (params.flags & FLAGS_USE_PATH_REGULARIZATION) != wp.uint32(0)
    max_roughness = float(0.0)  # noqa: UP018
    while depth < max_depth:
        if bsdf_absorbed:
            break

        sec_payload = _init_primary_payload()
        wp.optix_trace(
            params.tlas,
            sec_origin,
            sec_direction,
            0.001,
            1.0e16,
            0.0,
            wp.uint32(255),
            wp.uint32(0) if (sec_event_type & BSDF_EVENT_TRANSMISSION) != 0 else ray_flags,
            wp.uint32(0),
            wp.uint32(2),
            wp.uint32(0),
            sec_payload,
        )

        t_sec = _payload_get_hitT(sec_payload)
        sphere_hit = _intersect_sphere_lights(params, sec_origin, sec_direction, t_sec)
        if sphere_hit.pdf > 0.0:
            sphere_mis = _power_heuristic(bsdf_pdf, sphere_hit.pdf)
            radiance = radiance + _mul_vec3(sec_throughput, sphere_hit.radiance) * sphere_mis
            break

        miss = wp.bool(t_sec >= DLSS_INF_DISTANCE * 0.99)

        if depth == 1 and is_glossy_reflection:
            path_length = wp.abs(t_sec)

        if miss:
            env_color = _sample_environment(params, sec_direction)
            env_pdf = _environment_pdf_for_direction(params, sec_direction)
            mis_weight = _power_heuristic(bsdf_pdf, wp.max(env_pdf, 0.0001))
            if mis_weight != mis_weight:
                mis_weight = 0.0
            radiance = radiance + _mul_vec3(sec_throughput, env_color) * mis_weight
            break

        sec_normal = wp.normalize(_payload_get_normal(sec_payload))
        sec_tangent = _payload_get_tangent(sec_payload)
        sec_uv = _payload_get_uv(sec_payload)
        sec_uv1 = _payload_get_uv1(sec_payload)
        sec_mat_id = wp.int32(_payload_get_materialId(sec_payload))
        sec_bsign = _payload_get_bitangentSign(sec_payload)

        sec_pbr = _evaluate_material_from_payload(
            params,
            sec_mat_id,
            sec_normal,
            sec_tangent,
            sec_bsign,
            sec_uv,
            sec_uv1,
            _payload_get_barycentrics(sec_payload)[0],
        )
        sec_pbr = _set_incident_medium_for_hit(sec_pbr, _payload_get_front_face(sec_payload))

        # C++ secondary_rchit.h lines 278-283: path regularization
        if use_path_reg:
            max_roughness = wp.max(sec_pbr.roughness, max_roughness)
            sec_pbr.roughness = max_roughness

        sec_hit_pos = sec_origin + sec_direction * t_sec
        radiance = radiance + _mul_vec3(sec_throughput, sec_pbr.color)

        sec_base_color = sec_pbr.diffuse
        sec_specular_color = sec_pbr.specular
        sec_to_eye = -sec_direction

        sec_ambient = _mul_vec3(sec_base_color, params.ambient_light) * ((1.0 - sec_pbr.metallic) * sec_pbr.occlusion)
        radiance = radiance + _mul_vec3(sec_throughput, sec_ambient)

        # Direct lighting at secondary hit (GGX BSDF).
        rng = _pcg_advance(rng)
        xi0_s = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        xi1_s = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        xi2_s = _pcg_rand01(rng)
        sec_ls = _sample_environment_light(params, xi0_s, xi1_s, xi2_s)
        if sec_ls.pdf > 1.0e-6 and (wp.dot(sec_ls.direction, sec_pbr.normal) > 0.0 or sec_pbr.transmission > 0.0):
            rng = _pcg_advance(rng)
            sec_eval_xi_z = _pcg_rand01(rng)
            sec_bsdf_eval = _bsdf_evaluate(
                sec_to_eye,
                sec_ls.direction,
                sec_pbr.normal,
                sec_pbr.Ng,
                sec_pbr.Nc,
                sec_pbr.T,
                sec_pbr.B,
                sec_base_color,
                sec_pbr.transmission_color,
                sec_specular_color,
                sec_pbr.roughness,
                sec_pbr.metallic,
                sec_pbr.specular_scalar,
                sec_pbr.sheen_roughness,
                sec_pbr.sheen_color,
                sec_pbr.ior1,
                sec_pbr.ior2,
                sec_pbr.transmission,
                sec_pbr.diffuse_transmission_factor,
                sec_pbr.diffuse_transmission_color,
                sec_pbr.clearcoat,
                sec_pbr.clearcoat_roughness,
                sec_pbr.occlusion,
                sec_pbr.is_thin_walled,
                sec_eval_xi_z,
            )
            if sec_bsdf_eval.pdf > 1.0e-6:
                sec_shadow = ShadowPayload()
                sec_shadow.visible = wp.uint32(0)
                sec_shadow.seed = wp.uint32(0)
                wp.optix_trace(
                    params.tlas,
                    _offset_ray(sec_hit_pos, sec_pbr.Ng),
                    sec_ls.direction,
                    0.001,
                    DLSS_INF_DISTANCE,
                    0.0,
                    wp.uint32(255),
                    wp.uint32(
                        wp.uint32(OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT)
                        | _backface_culling_ray_flag(params)
                    ),
                    wp.uint32(1),
                    wp.uint32(2),
                    wp.uint32(1),
                    sec_shadow,
                )
                if sec_shadow.visible == wp.uint32(1):
                    sec_mis = _power_heuristic(sec_ls.pdf, sec_bsdf_eval.pdf)
                    sec_bsdf_sum = sec_bsdf_eval.bsdf_diffuse + sec_bsdf_eval.bsdf_glossy
                    sec_light_contrib = _mul_vec3(sec_bsdf_sum, sec_ls.radiance) * (sec_mis / sec_ls.pdf)
                    radiance = radiance + _mul_vec3(sec_throughput, sec_light_contrib)

        rng = _pcg_advance(rng)
        sec_sphere_xi0 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        sec_sphere_xi1 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        sec_sphere_xi2 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        sec_sphere_lobe_xi = _pcg_rand01(rng)
        sec_sphere_contrib = _sample_sphere_light_contribution(
            params,
            sec_hit_pos,
            sec_pbr,
            sec_to_eye,
            sec_base_color,
            sec_specular_color,
            sec_sphere_xi0,
            sec_sphere_xi1,
            sec_sphere_xi2,
            sec_sphere_lobe_xi,
        )
        radiance = radiance + _mul_vec3(sec_throughput, sec_sphere_contrib)

        # Sample next bounce direction (GGX VNDF).
        rng = _pcg_advance(rng)
        sxi0 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        sxi1 = _pcg_rand01(rng)
        rng = _pcg_advance(rng)
        sxi2 = _pcg_rand01(rng)
        sec_sample = _bsdf_sample(
            sec_to_eye,
            sec_pbr.normal,
            sec_pbr.Ng,
            sec_pbr.Nc,
            sec_pbr.T,
            sec_pbr.B,
            sec_base_color,
            sec_pbr.transmission_color,
            sec_specular_color,
            sec_pbr.roughness,
            sec_pbr.metallic,
            sec_pbr.specular_scalar,
            sec_pbr.sheen_roughness,
            sec_pbr.sheen_color,
            sec_pbr.ior1,
            sec_pbr.ior2,
            sec_pbr.transmission,
            sec_pbr.diffuse_transmission_factor,
            sec_pbr.diffuse_transmission_color,
            sec_pbr.clearcoat,
            sec_pbr.clearcoat_roughness,
            sec_pbr.is_thin_walled,
            sxi0,
            sxi1,
            sxi2,
        )

        # C++ lines 655-661: absorb or NaN/Inf guard on bsdf_over_pdf and throughput.
        bop = sec_sample.bsdf_over_pdf
        if sec_sample.event_type == BSDF_EVENT_ABSORB:
            break
        if bop[0] != bop[0] or bop[1] != bop[1] or bop[2] != bop[2]:
            break
        sec_throughput = _mul_vec3(sec_throughput, bop)
        if (
            sec_throughput[0] != sec_throughput[0]
            or sec_throughput[1] != sec_throughput[1]
            or sec_throughput[2] != sec_throughput[2]
        ):
            break

        completed_thick_exit = wp.bool(
            sec_pbr.is_thin_walled == 0
            and _payload_get_front_face(sec_payload) == wp.uint32(0)
            and (sec_sample.event_type & BSDF_EVENT_TRANSMISSION) != 0
        )

        # Russian roulette.
        if depth >= int(params.russian_roulette_start_bounce) and not completed_thick_exit:
            max_comp = wp.max(sec_throughput[0], wp.max(sec_throughput[1], sec_throughput[2]))
            rr_prob = wp.clamp(wp.max(max_comp, 0.05), 0.05, 0.99)
            rng = _pcg_advance(rng)
            if _pcg_rand01(rng) > rr_prob:
                break
            sec_throughput = sec_throughput / rr_prob

        sec_origin = _offset_ray_for_direction(sec_hit_pos, sec_pbr.Ng, sec_sample.direction)
        sec_direction = wp.normalize(sec_sample.direction)
        sec_event_type = sec_sample.event_type
        bsdf_pdf = wp.max(sec_sample.pdf, 0.0001)
        if not completed_thick_exit:
            depth = depth + wp.int32(1)

    # Specular albedo (pre-integrated environment term).
    guide_f0 = _mul_vec3(
        _material_f0(
            guide_pbr.diffuse,
            guide_pbr.specular,
            guide_pbr.specular_scalar,
            guide_pbr.metallic,
            guide_pbr.ior1,
            guide_pbr.ior2,
            guide_pbr.clearcoat,
        ),
        psr_throughput,
    )
    v_dot_n = wp.max(wp.dot(to_eye, guide_pbr.normal), 0.0)
    f_env = _environment_term_rtg(guide_f0, v_dot_n, guide_pbr.roughness)
    aux_specular_albedo = wp.vec4(f_env[0], f_env[1], f_env[2], 0.0)
    aux_spec_hit_dist = path_length

    # Guard against NaN/Inf (matches C++ lines 688-693).
    if radiance[0] != radiance[0] or radiance[1] != radiance[1] or radiance[2] != radiance[2]:
        radiance = wp.vec3(0.0, 0.0, 0.0)
    if direct_lum[0] != direct_lum[0] or direct_lum[1] != direct_lum[1] or direct_lum[2] != direct_lum[2]:
        direct_lum = wp.vec3(0.0, 0.0, 0.0)
    color = radiance + direct_lum

    # Debug visualization modes (matches C++ outputMode enums).
    out_color = color
    if params.output_mode == 2:  # DEPTH
        depth_vis = 0.0 if aux_view_z >= DLSS_INF_DISTANCE else wp.exp(-0.075 * aux_view_z)
        out_color = wp.vec3(depth_vis, depth_vis, depth_vis)
    elif params.output_mode == 3:  # MOTION
        out_color = wp.vec3(
            wp.clamp(0.5 + aux_motion[0] * 8.0, 0.0, 1.0),
            wp.clamp(0.5 + aux_motion[1] * 8.0, 0.0, 1.0),
            0.0,
        )
    elif params.output_mode == 4:  # NORMAL
        out_color = wp.vec3(
            0.5 * (world_normal[0] + 1.0),
            0.5 * (world_normal[1] + 1.0),
            0.5 * (world_normal[2] + 1.0),
        )
    elif params.output_mode == 5:  # ROUGHNESS
        out_color = wp.vec3(aux_normal_roughness[3], aux_normal_roughness[3], aux_normal_roughness[3])
    elif params.output_mode == 6:  # DIFFUSE
        out_color = base_color
    elif params.output_mode == 7:  # SPECULAR (viewer uses metallic visualization)
        out_color = wp.vec3(pbr_mat.metallic, pbr_mat.metallic, pbr_mat.metallic)
    elif params.output_mode == 8:  # SPEC_HITDIST
        vis = 1.0 - wp.exp(-0.05 * wp.max(aux_spec_hit_dist, 0.0))
        out_color = wp.vec3(vis, vis, vis)

    params.color_output[y, x] = wp.vec4(out_color[0], out_color[1], out_color[2], pbr_mat.opacity)
    params.normal_roughness_output[y, x] = aux_normal_roughness
    params.motion_output[y, x] = aux_motion
    params.depth_output[y, x] = aux_view_z
    params.diffuse_output[y, x] = aux_diffuse_albedo
    params.specular_output[y, x] = aux_specular_albedo
    params.spec_hit_dist_output[y, x] = aux_spec_hit_dist


@wp.kernel(module="unique")
def primary_miss(params: PathtraceLaunchParams):
    """Miss shader: set hitT = DLSS_INF_DISTANCE and store sky color in normal/envRad fields."""
    rd = wp.normalize(wp.optix_get_world_ray_direction())
    sky = _sample_environment(params, rd)
    miss_payload = PrimaryMissPayload()
    miss_payload.hit_t = DLSS_INF_DISTANCE
    miss_payload.env_radiance = sky
    wp.optix_store_payload(miss_payload)


@wp.struct
class HitGeometry:
    """Geometry data computed at a hit point (matches C++ HitState + material ID)."""

    normal: wp.vec3
    tangent: wp.vec3
    uv: wp.vec2
    uv1: wp.vec2
    bitangent_sign: wp.float32
    texture_lod: wp.float32
    material_id: wp.int32


@wp.func
def _resolve_material_id(
    params: PathtraceLaunchParams,
    inst_id: wp.int32,
    render_prim_id: wp.int32,
    primitive_id: wp.int32,
    primitive_count: wp.int32,
) -> wp.int32:
    """Resolve geometry material and the existing per-instance override."""
    material_id = wp.int32(0)
    if (
        render_prim_id >= 0
        and render_prim_id < int(params.render_prim_count)
        and primitive_id >= 0
        and primitive_id < primitive_count
    ):
        render_primitives = params.render_primitives
        rp = render_primitives[render_prim_id]
        material_ids = params.packed_material_ids
        material_id = wp.int32(material_ids[int(rp.material_id_offset) + primitive_id])
    if inst_id >= 0 and inst_id < int(params.instance_count):
        instance_material_ids = params.instance_material_ids
        if instance_material_ids.shape[0] > 0:
            override_id = instance_material_ids[inst_id]
            if override_id != wp.uint32(4294967295):
                material_id = wp.int32(override_id)
    return wp.clamp(material_id, 0, int(params.material_count) - 1)


@wp.func
def _compute_curve_hit_geometry(params: PathtraceLaunchParams, inst_id: wp.int32, segment_id: wp.int32) -> HitGeometry:
    """Extract a native round-linear or cubic Bezier curve hit."""
    geo = HitGeometry()
    geo.texture_lod = 0.0
    geo.normal = wp.vec3(0.0, 0.0, 1.0)
    geo.tangent = wp.vec3(1.0, 0.0, 0.0)
    geo.uv = wp.vec2(0.0, 0.0)
    geo.uv1 = wp.vec2(0.0, 0.0)
    geo.bitangent_sign = 1.0
    geo.material_id = wp.int32(0)

    if inst_id < 0 or inst_id >= int(params.instance_count):
        return geo
    instance_render_prim_ids = params.instance_render_prim_ids
    if instance_render_prim_ids.shape[0] == 0:
        return geo
    render_prim_id = int(instance_render_prim_ids[inst_id])
    if render_prim_id < 0 or render_prim_id >= int(params.render_prim_count):
        return geo
    render_primitives = params.render_primitives
    rp = render_primitives[render_prim_id]
    segment_count = int(rp.num_indices)
    if segment_id < 0 or segment_id >= segment_count:
        return geo

    indices = params.packed_indices
    control_id = int(indices[int(rp.index_offset) + segment_id])
    position_base = int(rp.vertex_buffer.position_offset) // 3
    positions = params.packed_positions
    p0 = _fetch_vec3(positions, position_base + control_id)
    p1 = _fetch_vec3(positions, position_base + control_id + 1)
    radius_base = int(rp.vertex_buffer.radius_offset)
    radii = params.packed_radii
    r0 = radii[radius_base + control_id]
    r1 = radii[radius_base + control_id + 1]
    u = wp.optix_get_curve_parameter()
    one_minus_u = 1.0 - u
    center_object = p0 * one_minus_u + p1 * u
    axis_object = p1 - p0
    radius_derivative = r1 - r0
    instance_geometry_types = params.instance_geometry_types
    geometry_type = wp.uint32(1)
    if instance_geometry_types.shape[0] > 0:
        geometry_type = instance_geometry_types[inst_id]
    if geometry_type == wp.uint32(2):
        p2 = _fetch_vec3(positions, position_base + control_id + 2)
        p3 = _fetch_vec3(positions, position_base + control_id + 3)
        r2 = radii[radius_base + control_id + 2]
        r3 = radii[radius_base + control_id + 3]
        one_minus_u2 = one_minus_u * one_minus_u
        u2 = u * u
        center_object = (
            p0 * (one_minus_u2 * one_minus_u)
            + p1 * (3.0 * one_minus_u2 * u)
            + p2 * (3.0 * one_minus_u * u2)
            + p3 * (u2 * u)
        )
        axis_object = (p1 - p0) * (3.0 * one_minus_u2) + (p2 - p1) * (6.0 * one_minus_u * u) + (p3 - p2) * (3.0 * u2)
        radius_derivative = (
            (r1 - r0) * (3.0 * one_minus_u2) + (r2 - r1) * (6.0 * one_minus_u * u) + (r3 - r2) * (3.0 * u2)
        )
    hit_world = wp.optix_get_world_ray_origin() + wp.optix_get_world_ray_direction() * wp.optix_get_ray_tmax()
    hit_object = wp.optix_transform_point_from_world_to_object_space(hit_world)
    radial_object = wp.normalize(hit_object - center_object)
    axis_length = wp.length(axis_object)
    normal_object = radial_object
    # On the swept body, a changing radius tilts the normal like a cone.
    # End caps and spherical elbows retain their radial normal.
    if u > 1.0e-5 and u < 1.0 - 1.0e-5 and axis_length > 1.0e-8:
        normal_object = wp.normalize(radial_object - axis_object * (radius_derivative / (axis_length * axis_length)))
    geo.normal = wp.normalize(wp.optix_transform_normal_from_object_to_world_space(normal_object))
    tangent_world = wp.normalize(wp.optix_transform_vector_from_object_to_world_space(axis_object))
    geo.tangent = wp.normalize(tangent_world - geo.normal * wp.dot(geo.normal, tangent_world))
    to_eye = -wp.normalize(wp.optix_get_world_ray_direction())
    if wp.dot(geo.normal, to_eye) < 0.0:
        geo.normal = -geo.normal
        geo.bitangent_sign = -geo.bitangent_sign
    geo.uv = wp.vec2(u, 0.0)
    geo.uv1 = geo.uv
    geo.material_id = _resolve_material_id(params, inst_id, render_prim_id, segment_id, segment_count)
    return geo


@wp.func
def _compute_hit_geometry(params: PathtraceLaunchParams, inst_id: wp.int32, tri_id: wp.int32) -> HitGeometry:
    """Compute geometry at hit point (matching C++ GetHitStateOptiX + material lookup)."""
    geo = HitGeometry()
    geo.texture_lod = 0.0
    geo.normal = wp.vec3(0.0, 0.0, 1.0)
    geo.tangent = wp.vec3(1.0, 0.0, 0.0)
    geo.uv = wp.vec2(0.0, 0.0)
    geo.uv1 = wp.vec2(0.0, 0.0)
    geo.bitangent_sign = 1.0
    geo.material_id = wp.int32(0)

    if inst_id < 0 or inst_id >= int(params.instance_count):
        return geo
    if params.instance_render_prim_ids.shape[0] == 0 or params.render_primitives.shape[0] == 0:
        return geo

    instance_render_prim_ids = params.instance_render_prim_ids
    render_prim_id = int(instance_render_prim_ids[inst_id])
    if render_prim_id < 0 or render_prim_id >= int(params.render_prim_count):
        return geo

    render_primitives = params.render_primitives
    rp = render_primitives[render_prim_id]
    tri_count = int(rp.num_indices) // 3
    if tri_id < 0 or tri_id >= tri_count:
        return geo

    indices = params.packed_indices
    index_base = int(rp.index_offset)
    i0 = int(indices[index_base + tri_id * 3 + 0])
    i1 = int(indices[index_base + tri_id * 3 + 1])
    i2 = int(indices[index_base + tri_id * 3 + 2])

    bary = wp.optix_get_triangle_barycentrics()
    b1 = bary[0]
    b2 = bary[1]
    b0 = 1.0 - b1 - b2

    # Normal
    vnor = params.packed_normals
    normal_base = int(rp.vertex_buffer.normal_offset) // 3
    n0 = _fetch_vec3(vnor, normal_base + i0)
    n1 = _fetch_vec3(vnor, normal_base + i1)
    n2 = _fetch_vec3(vnor, normal_base + i2)
    n_obj = wp.normalize(n0 * b0 + n1 * b1 + n2 * b2)
    geo.normal = wp.normalize(wp.optix_transform_normal_from_object_to_world_space(n_obj))

    # Tangent
    vtng = params.packed_tangents
    tangent_base = int(rp.vertex_buffer.tangent_offset) // 4
    t0 = _fetch_vec4(vtng, tangent_base + i0)
    t1 = _fetch_vec4(vtng, tangent_base + i1)
    t2 = _fetch_vec4(vtng, tangent_base + i2)
    t_interp = t0 * b0 + t1 * b1 + t2 * b2
    t_world = wp.normalize(
        wp.optix_transform_vector_from_object_to_world_space(wp.vec3(t_interp[0], t_interp[1], t_interp[2]))
    )
    geo.tangent = wp.normalize(t_world - geo.normal * wp.dot(geo.normal, t_world))
    geo.bitangent_sign = t_interp[3] * params.bitangent_flip
    # Match C++ closest-hit behavior: face-forward normal against view direction.
    to_eye = -wp.normalize(wp.optix_get_world_ray_direction())
    if wp.dot(geo.normal, to_eye) < 0.0:
        geo.normal = -geo.normal
        geo.bitangent_sign = -geo.bitangent_sign

    # UV0
    vt0 = params.packed_texcoords0
    tex0_base = int(rp.vertex_buffer.texcoord0_offset) // 2
    uv0_0 = _fetch_vec2(vt0, tex0_base + i0)
    uv0_1 = _fetch_vec2(vt0, tex0_base + i1)
    uv0_2 = _fetch_vec2(vt0, tex0_base + i2)
    geo.uv = uv0_0 * b0 + uv0_1 * b1 + uv0_2 * b2

    # UV1
    uv1_0 = uv0_0
    uv1_1 = uv0_1
    uv1_2 = uv0_2
    geo.uv1 = geo.uv
    if rp.vertex_buffer.has_texcoord1 != wp.uint32(0):
        vt1 = params.packed_texcoords1
        tex1_base = int(rp.vertex_buffer.texcoord1_offset) // 2
        uv1_0 = _fetch_vec2(vt1, tex1_base + i0)
        uv1_1 = _fetch_vec2(vt1, tex1_base + i1)
        uv1_2 = _fetch_vec2(vt1, tex1_base + i2)
        geo.uv1 = uv1_0 * b0 + uv1_1 * b1 + uv1_2 * b2

    geo.material_id = _resolve_material_id(params, inst_id, render_prim_id, tri_id, tri_count)
    # Estimate an isotropic ray footprint from projected pixel width and the
    # triangle's texel density. This gives stable trilinear filtering without
    # tracing ray differentials.
    vertices = wp.optix_get_triangle_vertex_data()
    p0 = wp.optix_transform_point_from_object_to_world_space(wp.vec3(vertices[0, 0], vertices[0, 1], vertices[0, 2]))
    p1 = wp.optix_transform_point_from_object_to_world_space(wp.vec3(vertices[1, 0], vertices[1, 1], vertices[1, 2]))
    p2 = wp.optix_transform_point_from_object_to_world_space(wp.vec3(vertices[2, 0], vertices[2, 1], vertices[2, 2]))
    world_area = wp.length(wp.cross(p1 - p0, p2 - p0))
    duv0_1 = uv0_1 - uv0_0
    duv0_2 = uv0_2 - uv0_0
    duv1_1 = uv1_1 - uv1_0
    duv1_2 = uv1_2 - uv1_0
    uv0_area = wp.abs(duv0_1[0] * duv0_2[1] - duv0_1[1] * duv0_2[0])
    uv1_area = wp.abs(duv1_1[0] * duv1_2[1] - duv1_1[1] * duv1_2[0])
    uv_area = wp.max(uv0_area, uv1_area)
    materials = params.compact_materials
    material = materials[int(geo.material_id)]
    texture_size = material.texture_size
    incidence = wp.max(wp.abs(wp.dot(to_eye, geo.normal)), 0.25)
    pixel_width = (
        2.0 * wp.optix_get_ray_tmax() * params.cam_tan_half_fov / wp.max(wp.float32(params.height), 1.0) / incidence
    )
    footprint = pixel_width * texture_size * wp.sqrt(uv_area / wp.max(world_area, 1.0e-12))
    max_lod = wp.log(wp.max(texture_size, 1.0)) / wp.log(2.0)
    geo.texture_lod = wp.clamp(wp.log(wp.max(footprint, 1.0)) / wp.log(2.0), 0.0, max_lod)

    return geo


@wp.func
def _apply_opacity_fresnel(
    opacity: wp.float32,
    low: wp.float32,
    high: wp.float32,
    falloff: wp.float32,
    normal: wp.vec3,
    to_eye: wp.vec3,
) -> wp.float32:
    """Apply the OmniUe4Translucent facing-ratio opacity response."""
    if low < 0.0:
        return wp.clamp(opacity, 0.0, 1.0)
    facing_ratio = 1.0 - wp.abs(wp.dot(wp.normalize(normal), wp.normalize(to_eye)))
    weight = 1.0
    if falloff > 0.0:
        weight = wp.pow(wp.clamp(facing_ratio, 0.0, 1.0), falloff)
    angular_opacity = low + (high - low) * weight
    return wp.clamp(opacity * angular_opacity, 0.0, 1.0)


@wp.struct
class AnyHitAlphaResult:
    valid: wp.uint32
    inst_id: wp.int32
    tri_id: wp.int32
    alpha: wp.float32
    transmission: wp.float32
    coverage_seed: wp.uint32


@wp.func
def _compute_any_hit_alpha(params: PathtraceLaunchParams) -> AnyHitAlphaResult:
    """Evaluate coverage for triangles and curves through one material path."""
    out = AnyHitAlphaResult()
    out.valid = wp.uint32(0)
    out.inst_id = wp.int32(-1)
    out.tri_id = wp.int32(-1)
    out.alpha = wp.float32(1.0)
    out.transmission = wp.float32(0.0)
    out.coverage_seed = wp.uint32(0)

    inst_id = int(wp.optix_get_instance_id())
    if inst_id < 0 or inst_id >= int(params.instance_count):
        return out

    primitive_id = int(wp.optix_get_primitive_index())
    if params.instance_render_prim_ids.shape[0] == 0 or params.render_primitives.shape[0] == 0:
        return out
    instance_render_prim_ids = params.instance_render_prim_ids
    render_prim_id = int(instance_render_prim_ids[inst_id])
    if render_prim_id < 0 or render_prim_id >= int(params.render_prim_count):
        return out

    render_primitives = params.render_primitives
    rp = render_primitives[render_prim_id]
    geometry_types = params.instance_geometry_types
    is_curve = geometry_types.shape[0] > 0 and geometry_types[inst_id] != wp.uint32(0)
    primitive_count = int(rp.num_indices) if is_curve else int(rp.num_indices) // 3
    if primitive_id < 0 or primitive_id >= primitive_count:
        return out
    material_id = _resolve_material_id(params, inst_id, render_prim_id, primitive_id, primitive_count)
    materials = params.compact_materials
    mat = materials[material_id]
    out.valid = wp.uint32(1)
    out.inst_id = wp.int32(inst_id)
    out.tri_id = wp.int32(primitive_id)
    out.transmission = wp.clamp(mat.transmission, 0.0, 1.0)
    if mat.opacity_fresnel_low >= 0.0 and out.transmission > 0.0:
        out.alpha = 1.0
        return out

    if mat.alpha_mode == wp.int32(0):
        out.alpha = 1.0
        return out

    if is_curve:
        geo = _compute_curve_hit_geometry(params, inst_id, primitive_id)
    else:
        geo = _compute_hit_geometry(params, inst_id, primitive_id)
    uv_base = _apply_uv_transform(
        _select_uv(mat.base_color_tex_coord, geo.uv, geo.uv1),
        mat.base_color_uv_transform,
    )
    uv_seed_x = wp.uint32(int(wp.floor(uv_base[0] * 2048.0)))
    uv_seed_y = wp.uint32(int(wp.floor(uv_base[1] * 2048.0)))
    launch_index = wp.optix_get_launch_index()
    pixel_frame_seed = _xxhash32(wp.uint32(launch_index[0]), wp.uint32(launch_index[1]), params.frame_index)
    out.coverage_seed = _xxhash32(
        wp.uint32(inst_id),
        wp.uint32(primitive_id),
        _xxhash32(uv_seed_x, uv_seed_y, pixel_frame_seed),
    )
    base_tex = _sample_texture_rgba(params, mat.base_color_tex_index, uv_base, 0.0)

    alpha = _apply_opacity_fresnel(
        mat.opacity * base_tex[3],
        mat.opacity_fresnel_low,
        mat.opacity_fresnel_high,
        mat.opacity_fresnel_falloff,
        geo.normal,
        -wp.optix_get_world_ray_direction(),
    )
    if mat.alpha_mode == wp.int32(1):
        alpha = 1.0 if alpha >= mat.alpha_cutoff else 0.0
    out.alpha = alpha
    return out


@wp.func
def _store_primary_hit_payload(
    geo: HitGeometry,
    inst_id: wp.int32,
    primitive_id: wp.int32,
    front_face: wp.uint32,
    parameter_0: wp.float32,
    parameter_1: wp.float32,
):
    """Store the geometry-neutral payload consumed by the shared PBR raygen."""
    payload = PrimaryPayload()
    payload.hit_t = wp.optix_get_ray_tmax()
    payload.normal = geo.normal
    payload.tangent = geo.tangent
    payload.uv = geo.uv
    payload.material_id = wp.uint32(geo.material_id)
    payload.bitangent_sign = geo.bitangent_sign
    payload.instance_id = inst_id
    payload.front_face = front_face
    payload.primitive_id = wp.uint32(primitive_id)
    payload.barycentrics = wp.vec3(geo.texture_lod, parameter_0, parameter_1)
    payload.uv1 = geo.uv1
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def primary_closest_hit(params: PathtraceLaunchParams):
    """Closest-hit: pass geometry data only (matching C++ primary_rchit.h)."""
    inst_id = int(wp.optix_get_instance_id())
    tri_id = int(wp.optix_get_primitive_index())
    geo = _compute_hit_geometry(params, inst_id, tri_id)
    bary = wp.optix_get_triangle_barycentrics()
    front_face = wp.uint32(1) if wp.optix_is_front_face_hit() else wp.uint32(0)
    _store_primary_hit_payload(geo, wp.int32(inst_id), wp.int32(tri_id), front_face, bary[0], bary[1])


@wp.kernel(module="unique")
def curve_primary_closest_hit(params: PathtraceLaunchParams):
    """Adapt a native curve hit to the same PBR payload as a triangle hit."""
    inst_id = int(wp.optix_get_instance_id())
    segment_id = int(wp.optix_get_primitive_index())
    geo = _compute_curve_hit_geometry(params, inst_id, segment_id)
    _store_primary_hit_payload(
        geo,
        wp.int32(inst_id),
        wp.int32(segment_id),
        wp.uint32(1),
        wp.optix_get_curve_parameter(),
        0.0,
    )


@wp.func
def _any_hit_rng(coverage_seed: wp.uint32) -> wp.uint32:
    return _pcg_advance(coverage_seed)


@wp.kernel(module="unique")
def primary_any_hit(params: PathtraceLaunchParams):
    alpha_hit = _compute_any_hit_alpha(params)
    if alpha_hit.valid == wp.uint32(0):
        return

    if alpha_hit.alpha <= 0.001:
        wp.optix_ignore_intersection()
        return
    if alpha_hit.alpha < 0.999:
        ah_rng = _any_hit_rng(alpha_hit.coverage_seed)
        r = _pcg_rand01(ah_rng)
        if r > alpha_hit.alpha:
            wp.optix_ignore_intersection()
            return


@wp.kernel(module="unique")
def secondary_miss(params: PathtraceLaunchParams):
    """Matches C++ __miss__secondary: visible path for shadow ray."""
    wp.optix_set_payload_0(wp.uint32(1))


@wp.kernel(module="unique")
def secondary_closest_hit(params: PathtraceLaunchParams):
    """Matches C++ __closesthit__secondary: occluded path for shadow ray."""
    wp.optix_set_payload_0(wp.uint32(0))


@wp.kernel(module="unique")
def secondary_any_hit(params: PathtraceLaunchParams):
    alpha_hit = _compute_any_hit_alpha(params)
    if alpha_hit.valid == wp.uint32(1):
        blocking = alpha_hit.alpha * (1.0 - alpha_hit.transmission)
        if blocking <= 0.001:
            wp.optix_ignore_intersection()
            return
        if blocking < 0.999:
            rng = _any_hit_rng(alpha_hit.coverage_seed)
            if _pcg_rand01(rng) > blocking:
                wp.optix_ignore_intersection()
                return

    payload = ShadowPayload()
    wp.optix_load_payload(payload)

    payload.visible = wp.uint32(0)
    wp.optix_store_payload(payload)
    wp.optix_terminate_ray()
