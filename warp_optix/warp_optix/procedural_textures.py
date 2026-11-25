# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable material textures generated with standard Warp launches."""

from functools import lru_cache
from importlib import import_module

import numpy as np
import warp as wp

from warp_optix._src.procedural_texture_models import INDEX
from warp_optix._src.texture_coefficients import decode

__all__ = ["bake", "bake_gltf", "brick", "marble", "reference_ids", "reference_maps", "wood"]


@wp.func
def _texel_uv(row: int, col: int, height: int, width: int) -> wp.vec2:
    return wp.vec2((float(col) + 0.5) / float(width), (float(row) + 0.5) / float(height))


@wp.func
def _tile_noise(state: wp.uint32, uv: wp.vec2, scale: wp.vec2) -> float:
    # Put each UV axis on a circle so opposite edges sample the same noise field.
    x = uv[0] * (2.0 * wp.PI)
    y = uv[1] * (2.0 * wp.PI)
    return wp.noise(
        state,
        wp.vec4(
            scale[0] * wp.cos(x),
            scale[0] * wp.sin(x),
            scale[1] * wp.cos(y),
            scale[1] * wp.sin(y),
        ),
    )


@wp.kernel
def wood(
    light: wp.vec3,
    dark: wp.vec3,
    grain_count: int,
    seed: int,
    image: wp.array2d(dtype=wp.vec3),
):
    """Bake long wood grain along the row axis. Use a positive grain count."""
    row, col = wp.tid()
    uv = _texel_uv(row, col, image.shape[0], image.shape[1])
    state = wp.rand_init(seed)
    bend = _tile_noise(state, uv, wp.vec2(2.0, 0.6))
    fiber = _tile_noise(state, uv, wp.vec2(20.0, 1.5))
    grain = 0.5 + 0.5 * wp.sin(2.0 * wp.PI * (float(grain_count) * uv[0] + bend))
    tone = wp.clamp(grain * grain + 0.25 * fiber, 0.0, 1.0)
    image[row, col] = wp.lerp(light, dark, tone)


@wp.kernel
def marble(
    stone: wp.vec3,
    vein: wp.vec3,
    vein_count: int,
    seed: int,
    image: wp.array2d(dtype=wp.vec3),
):
    """Bake warped diagonal marble veins. Use a positive vein count."""
    row, col = wp.tid()
    uv = _texel_uv(row, col, image.shape[0], image.shape[1])
    state = wp.rand_init(seed)
    bend = _tile_noise(state, uv, wp.vec2(1.0, 1.0))
    detail = _tile_noise(state, uv, wp.vec2(3.0, 3.0))
    phase = float(vein_count) * (uv[0] + uv[1]) + 2.0 * bend + 0.35 * detail
    distance = wp.abs(wp.sin(wp.PI * phase))
    tone = 1.0 - wp.smoothstep(0.02, 0.35, distance)
    image[row, col] = wp.lerp(stone, vein, tone)


@wp.kernel
def brick(
    clay: wp.vec3,
    mortar: wp.vec3,
    columns: int,
    row_pairs: int,
    mortar_fraction: float,
    image: wp.array2d(dtype=wp.vec3),
):
    """Bake staggered brick. Use positive counts and a mortar fraction in [0, 1)."""
    row, col = wp.tid()
    uv = _texel_uv(row, col, image.shape[0], image.shape[1])
    rows = 2.0 * float(row_pairs)
    y = uv[1] * rows
    course = wp.floor(y)
    offset = 0.5 * float(int(course) % 2)
    x = uv[0] * float(columns) + offset
    fx = x - wp.floor(x)
    fy = y - course
    half_gap = 0.5 * mortar_fraction
    # Smooth each joint across approximately one texel at the output resolution.
    dx = 0.5 * float(columns) / float(image.shape[1])
    dy = 0.5 * rows / float(image.shape[0])
    bx = wp.smoothstep(half_gap - dx, half_gap + dx, wp.min(fx, 1.0 - fx))
    by = wp.smoothstep(half_gap - dy, half_gap + dy, wp.min(fy, 1.0 - fy))
    mask = bx * by
    if mortar_fraction == 0.0:
        mask = 1.0
    image[row, col] = wp.lerp(mortar, clay, mask)


@wp.func
def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return wp.pow((value + 0.055) / 1.055, 2.4)


@wp.func
def _value_noise(uv: wp.vec2, cells: int, seed: int) -> wp.vec3:
    p = uv * float(cells)
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    fx = p[0] - float(x)
    fy = p[1] - float(y)
    tx = wp.smoothstep(0.0, 1.0, fx)
    ty = wp.smoothstep(0.0, 1.0, fy)
    x1 = (x + 1) % cells
    y1 = (y + 1) % cells
    a = wp.randf(wp.rand_init(seed, x + y * cells))
    b = wp.randf(wp.rand_init(seed, x1 + y * cells))
    c = wp.randf(wp.rand_init(seed, x + y1 * cells))
    d = wp.randf(wp.rand_init(seed, x1 + y1 * cells))
    value = wp.lerp(wp.lerp(a, b, tx), wp.lerp(c, d, tx), ty) - 0.5
    du = 6.0 * fx * (1.0 - fx) * wp.lerp(b - a, d - c, ty)
    dv = 6.0 * fy * (1.0 - fy) * wp.lerp(c - a, d - b, tx)
    return wp.vec3(value, du, dv)


@wp.func
def _map_detail(noise: wp.vec3, amplitude: wp.vec3, normal_map: bool) -> wp.vec3:
    if normal_map:
        # Image rows run down. OpenGL tangent Y points up, so the height
        # gradient has opposite signs in the two output channels.
        return wp.cw_mul(amplitude, wp.vec3(-0.5 * noise[1], 0.5 * noise[2], 0.0))
    return amplitude * noise[0]


@wp.func
def _cubic_weights(t: float) -> wp.vec4:
    t2 = t * t
    t3 = t2 * t
    return wp.vec4(
        -0.5 * t + t2 - 0.5 * t3, 1.0 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2.0 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3
    )


@wp.func
def _sample_field(field: wp.array2d(dtype=wp.vec3), uv: wp.vec2) -> wp.vec3:
    size = field.shape[0]
    p = uv * float(size) - wp.vec2(0.5)
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    wx = _cubic_weights(p[0] - float(x))
    wy = _cubic_weights(p[1] - float(y))
    value = wp.vec3(0.0)
    for j in range(4):
        row = (y + j - 1) & (size - 1)
        for i in range(4):
            col = (x + i - 1) & (size - 1)
            value += (wx[i] * wy[j]) * field[row, col]
    return value


@wp.kernel
def _bake_reference(
    field: wp.array2d(dtype=wp.vec3),
    microdetail: wp.vec3,
    offset: wp.vec2,
    phase: wp.vec2,
    noise_seed: int,
    srgb: bool,
    normal_map: bool,
    image: wp.array2d(dtype=wp.vec3),
):
    row, col = wp.tid()
    uv = _texel_uv(row, col, image.shape[0], image.shape[1])
    shear_u = 0.0
    shear_v = 0.0
    if noise_seed != 0:
        # Every map uses the same periodic deformation. Keep the Jacobian
        # so slopes in the tangent-space normal follow the remapped surface.
        wave_u = 4.0 * wp.PI * uv[1] + phase[0]
        wave_v = 6.0 * wp.PI * uv[0] + phase[1]
        shear_u = 0.020 * 4.0 * wp.PI * wp.cos(wave_u)
        shear_v = 0.015 * 6.0 * wp.PI * wp.cos(wave_v)
        uv += offset + wp.vec2(0.020 * wp.sin(wave_u), 0.015 * wp.sin(wave_v))
        uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    value = _sample_field(field, uv)
    if wp.dot(microdetail, microdetail) > 0.0:
        value += _map_detail(_value_noise(uv, 256, noise_seed + 157), microdetail, normal_map)
    value = wp.vec3(wp.clamp(value[0], 0.0, 1.0), wp.clamp(value[1], 0.0, 1.0), wp.clamp(value[2], 0.0, 1.0))
    if srgb:
        value = wp.vec3(_srgb_to_linear(value[0]), _srgb_to_linear(value[1]), _srgb_to_linear(value[2]))
    if normal_map:
        xy = 2.0 * wp.vec2(value[0], value[1]) - wp.vec2(1.0)
        z = wp.sqrt(wp.max(1.0e-8, 1.0 - wp.dot(xy, xy)))
        n = wp.normalize(wp.vec3(xy[0] - shear_v * xy[1], xy[1] - shear_u * xy[0], z))
        value = 0.5 * n + wp.vec3(0.5)
    image[row, col] = value


def reference_ids() -> tuple[str, ...]:
    """List the fitted ambientCG material IDs."""
    return tuple(INDEX)


def reference_maps(material: str) -> tuple[str, ...]:
    """List the fitted maps for an ambientCG material ID."""
    return INDEX[material][1]


@lru_cache(maxsize=64)
def _field(material, map_type):
    shard, _ = INDEX[material]
    models = import_module(f"._src.procedural_texture_models._data{shard:02}", __package__)
    return decode(models.MODELS[material][map_type], map_type)


def bake(
    material: str,
    shape: tuple[int, int],
    *,
    map_type: str = "color",
    device: wp.DeviceLike = None,
    seed: int | None = None,
) -> wp.array:
    """Bake a fitted ambientCG texture into a Warp array of shape (height, width).

    Color is linear RGB. Normals are OpenGL tangent-space normals encoded
    in [0, 1]. Scalar maps repeat the normalized value in all three output
    channels. Displacement is unitless. Pass the same non-negative seed for
    every map of a variant. None gives the closest fit to the reference.
    """
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("Texture dimensions must be positive.")
    field, microdetail = _field(material, map_type)
    resolution = min(height, width)
    while field.shape[0] > 2 * resolution:
        size = field.shape[0] // 2
        field = field.reshape(size, 2, size, 2, 3).mean(axis=(1, 3))
    gain = min(1.0, max(0.0, (resolution - 256.0) / 256.0))
    offset, phase = wp.vec2(0.0), wp.vec2(0.0)
    noise_seed = 0
    if seed is not None:
        rng = np.random.default_rng(seed)
        offset = wp.vec2(*rng.random(2))
        phase = wp.vec2(*rng.uniform(0.0, 2.0 * np.pi, 2))
        noise_seed = int(rng.integers(1, 1 << 30))
    image = wp.empty(shape, dtype=wp.vec3, device=device)
    wp.launch(
        _bake_reference,
        dim=shape,
        inputs=[
            wp.array(field, dtype=wp.vec3, device=image.device),
            wp.vec3(*(value * gain for value in microdetail)),
            offset,
            phase,
            noise_seed,
            map_type == "color",
            map_type == "normal",
        ],
        outputs=[image],
        device=image.device,
    )
    return image


@wp.func
def _rgba8(rgb: wp.vec3, alpha: float) -> wp.vec4ub:
    return wp.vec4ub(
        wp.uint8(rgb[0] * 255.0 + 0.5),
        wp.uint8(rgb[1] * 255.0 + 0.5),
        wp.uint8(rgb[2] * 255.0 + 0.5),
        wp.uint8(alpha * 255.0 + 0.5),
    )


@wp.kernel
def _pack_gltf(
    base: wp.array2d(dtype=wp.vec3),
    normal: wp.array2d(dtype=wp.vec3),
    rough: wp.array2d(dtype=wp.vec3),
    occlusion: wp.array2d(dtype=wp.vec3),
    metal: wp.array2d(dtype=wp.vec3),
    opacity: wp.array2d(dtype=wp.vec3),
    has_occlusion: bool,
    has_metal: bool,
    has_opacity: bool,
    packed: wp.array3d(dtype=wp.vec4ub),
):
    row, col = wp.tid()
    red = 1.0
    blue = 0.0
    alpha = 1.0
    if has_occlusion:
        red = occlusion[row, col][0]
    if has_metal:
        blue = metal[row, col][0]
    if has_opacity:
        alpha = opacity[row, col][0]
    packed[0, row, col] = _rgba8(base[row, col], alpha)
    packed[1, row, col] = _rgba8(normal[row, col], 1.0)
    packed[2, row, col] = _rgba8(wp.vec3(red, rough[row, col][0], blue), 1.0)


def bake_gltf(
    material: str,
    shape: tuple[int, int],
    *,
    device: wp.DeviceLike = None,
    first_texture_index: int = 0,
    seed: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Return RGBA8 textures and keyword arguments for the path tracer.

    Pass the textures to Scene.set_gltf_textures without sRGB decoding.
    Pass the keywords to Scene.materials.add_gltf_material. When appending to
    a scene, set first_texture_index to the count of textures already present.
    A seed generates a consistent variant across all maps.
    """
    maps = reference_maps(material)

    def generate(map_type: str) -> wp.array:
        return bake(material, shape, map_type=map_type, device=device, seed=seed)

    base = generate("color")
    normal = generate("normal")
    rough = generate("roughness")
    has_occlusion = "occlusion" in maps
    has_metal = "metalness" in maps
    has_opacity = "opacity" in maps
    packed = wp.empty((3, *shape), dtype=wp.vec4ub, device=base.device)
    wp.launch(
        _pack_gltf,
        dim=shape,
        inputs=[
            base,
            normal,
            rough,
            generate("occlusion") if has_occlusion else rough,
            generate("metalness") if has_metal else rough,
            generate("opacity") if has_opacity else rough,
            has_occlusion,
            has_metal,
            has_opacity,
        ],
        outputs=[packed],
        device=base.device,
    )
    kwargs: dict[str, dict[str, int] | str] = {
        "base_color_texture": {"index": first_texture_index},
        "normal_texture": {"index": first_texture_index + 1},
        "metallic_roughness_texture": {"index": first_texture_index + 2},
    }
    if has_occlusion:
        kwargs["occlusion_texture"] = {"index": first_texture_index + 2}
    if has_opacity:
        kwargs["alpha_mode"] = "MASK"
    return list(packed.numpy()), kwargs
