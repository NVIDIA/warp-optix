# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable material textures generated with standard Warp launches."""

from functools import lru_cache
from pathlib import Path
from struct import unpack_from
from zlib import decompress

import numpy as np
import warp as wp

from warp_optix._src.texture_coefficients import decode, read_index, read_record

__all__ = ["bake", "bake_gltf", "brick", "marble", "reference_ids", "reference_maps", "wood"]

_package = Path(__file__).resolve().parent
_catalog = _package / "_src" / "material_catalog"
if not (_catalog / "models" / "index.json").is_file():
    _catalog = _package.parents[1] / "material_catalog"
_MODELS = _catalog / "models"
_DETAIL = _catalog / "detail"
INDEX = read_index(_MODELS) if (_MODELS / "index.json").is_file() else {}
DETAIL_INDEX = read_index(_DETAIL) if (_DETAIL / "index.json").is_file() else {}


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


@wp.func
def _sample_detail(tiles: wp.array3d(dtype=wp.vec3), band: int, point: wp.vec2) -> wp.vec3:
    x = int(wp.floor(point[0]))
    y = int(wp.floor(point[1]))
    fx = point[0] - float(x)
    fy = point[1] - float(y)
    mask = tiles.shape[1] - 1
    a = wp.lerp(tiles[band, y & mask, x & mask], tiles[band, y & mask, (x + 1) & mask], fx)
    b = wp.lerp(tiles[band, (y + 1) & mask, x & mask], tiles[band, (y + 1) & mask, (x + 1) & mask], fx)
    return wp.lerp(a, b, fy)


@wp.func
def _detail_amplitude(envelope: wp.array3d(dtype=float), band: int, uv: wp.vec2) -> float:
    size = envelope.shape[1]
    p = uv * float(size) - wp.vec2(0.5)
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    fx = p[0] - float(x)
    fy = p[1] - float(y)
    mask = size - 1
    a = wp.lerp(envelope[band, y & mask, x & mask], envelope[band, y & mask, (x + 1) & mask], fx)
    b = wp.lerp(envelope[band, (y + 1) & mask, x & mask], envelope[band, (y + 1) & mask, (x + 1) & mask], fx)
    return wp.lerp(a, b, fy)


@wp.func
def _translated_detail(tiles: wp.array3d(dtype=wp.vec3), band: int, uv: wp.vec2, level: int, seed: int) -> wp.vec3:
    # Blend translated samples near region boundaries. Translating the patch
    # instead of warping it keeps its measured grain direction and frequency.
    grid = uv * 8.0
    x = int(wp.floor(grid[0]))
    y = int(wp.floor(grid[1]))
    fx = wp.smoothstep(0.35, 0.65, grid[0] - float(x))
    fy = wp.smoothstep(0.35, 0.65, grid[1] - float(y))
    point = uv * float(level) - wp.vec2(0.5)
    value = wp.vec3(0.0)
    energy = float(0.0)
    for j in range(2):
        for i in range(2):
            weight = wp.lerp(1.0 - fx, fx, float(i)) * wp.lerp(1.0 - fy, fy, float(j))
            if weight > 0.0:
                corner = ((y + j) & 7) * 8 + ((x + i) & 7)
                state = wp.rand_init(seed + level, corner)
                size = tiles.shape[1]
                shift = wp.vec2(float(wp.randi(state, 0, size)), float(wp.randi(state, 0, size)))
                value += weight * _sample_detail(tiles, band, point + shift)
                energy += weight * weight
    return value / wp.sqrt(energy)


@wp.func
def _edge_texel(edges: wp.array2d(dtype=wp.vec2b), y: int, x: int) -> wp.vec2:
    mask = edges.shape[0] - 1
    delta = edges[y & mask, x & mask]
    return wp.vec2(float(delta[0]), float(delta[1]))


@wp.func
def _sample_edges(edges: wp.array2d(dtype=wp.vec2b), uv: wp.vec2) -> wp.vec2:
    p = uv * float(edges.shape[0]) - wp.vec2(0.5)
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    a = wp.lerp(_edge_texel(edges, y, x), _edge_texel(edges, y, x + 1), p[0] - float(x))
    b = wp.lerp(_edge_texel(edges, y + 1, x), _edge_texel(edges, y + 1, x + 1), p[0] - float(x))
    return wp.lerp(a, b, p[1] - float(y))


@wp.func
def _grain_modulation(uv: wp.vec2, seed: int, aggregate: bool) -> float:
    state = wp.rand_init(seed + 601)
    variation = 1.0 + 0.34 * _tile_noise(state, uv, wp.vec2(5.0, 5.0))
    if aggregate:
        # One jittered spot per cell, inset so no neighbor lookup is needed.
        p = uv * 128.0
        x = int(wp.floor(p[0]))
        y = int(wp.floor(p[1]))
        spot = wp.rand_init(seed + 991, (y & 127) * 128 + (x & 127))
        center = wp.vec2(wp.randf(spot, 0.30, 0.70), wp.randf(spot, 0.30, 0.70))
        delta = p - wp.vec2(float(x), float(y)) - center
        variation *= 0.94 + 0.45 * (1.0 - wp.smoothstep(0.005, 0.075, wp.dot(delta, delta)))
    return variation


@wp.func
def _sample_parquet(layout: wp.array3d(dtype=wp.vec3), uv: wp.vec2) -> wp.vec3:
    grid = uv * 10.0
    x = int(wp.floor(grid[0]))
    y = int(wp.floor(grid[1]))
    along = grid[0] - float(x)
    across = grid[1] - float(y)
    if ((x + y) & 1) != 0:
        along, across = across, along
    block = (y % 10) * 10 + (x % 10)
    across *= float(layout.shape[1])
    board = wp.min(int(across), layout.shape[1] - 1)
    p = wp.clamp(along * 8.0 - 0.5, 0.0, 7.0)
    q = wp.clamp((across - float(board)) * 8.0 - 0.5, 0.0, 7.0)
    i, j = int(wp.floor(p)), int(wp.floor(q))
    i1, j1 = wp.min(i + 1, 7), wp.min(j + 1, 7)
    a = wp.lerp(layout[block, board, j * 8 + i], layout[block, board, j * 8 + i1], p - float(i))
    b = wp.lerp(layout[block, board, j1 * 8 + i], layout[block, board, j1 * 8 + i1], p - float(i))
    return wp.lerp(a, b, q - float(j))


@wp.func
def _sample_repeat(layout: wp.array3d(dtype=wp.vec3), uv: wp.vec2) -> wp.vec3:
    size = layout.shape[1]
    p = uv * (5.0 * float(size)) - wp.vec2(0.5)
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    x0, y0 = (x + size) % size, (y + size) % size
    x1, y1 = (x0 + 1) % size, (y0 + 1) % size
    a = wp.lerp(layout[0, y0, x0], layout[0, y0, x1], p[0] - float(x))
    b = wp.lerp(layout[0, y1, x0], layout[0, y1, x1], p[0] - float(x))
    return wp.lerp(a, b, p[1] - float(y))


@wp.kernel
def _bake_reference(
    field: wp.array2d(dtype=wp.vec3),
    tiles: wp.array3d(dtype=wp.vec3),
    envelope: wp.array3d(dtype=float),
    edges: wp.array2d(dtype=wp.vec2b),
    edge_scale: float,
    layout: wp.array3d(dtype=wp.vec3),
    layout_kind: int,
    offset: wp.vec2,
    phase: wp.vec2,
    noise_seed: int,
    aggregate: bool,
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
        if layout_kind == 0:
            wave_u = 4.0 * wp.PI * uv[1] + phase[0]
            wave_v = 6.0 * wp.PI * uv[0] + phase[1]
            shear_u = 0.020 * 4.0 * wp.PI * wp.cos(wave_u)
            shear_v = 0.015 * 6.0 * wp.PI * wp.cos(wave_v)
            uv += wp.vec2(0.020 * wp.sin(wave_u), 0.015 * wp.sin(wave_v))
        uv += offset
        uv -= wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    resolution = float(wp.min(image.shape[0], image.shape[1]))
    layout_resolution = field.shape[0]
    if layout_kind == 1 and resolution >= 128.0:
        value = _sample_parquet(layout, uv)
        layout_resolution = 512
    elif layout_kind == 2 and resolution >= 128.0:
        value = _sample_repeat(layout, uv)
        layout_resolution = 1024
    else:
        value = _sample_field(field, uv)
    if edges.shape[0] > 0 and resolution >= 256.0:
        transition = wp.smoothstep(256.0, 512.0, resolution)
        delta = (edge_scale * transition) * _sample_edges(edges, uv)
        if normal_map:
            value += wp.vec3(delta[0], delta[1], 0.0)
        else:
            value += wp.vec3(delta[0])
        if transition > 0.5:
            layout_resolution = wp.max(layout_resolution, edges.shape[0] // 2)
    if tiles.shape[0] > 0:
        level = int(256)
        grain_modulation = float(1.0)
        if resolution > float(128 << tiles.shape[0]):
            grain_modulation = _grain_modulation(uv, noise_seed, aggregate)
        for band in range(5):
            if level > layout_resolution and resolution > 0.5 * float(level):
                gain = wp.smoothstep(0.5, 1.0, resolution / float(level))
                source = wp.min(band, tiles.shape[0] - 1)
                # Extend beyond the captured frequencies with diminishing
                # residual contrast, rather than magnifying a low-res bake.
                if band > source:
                    gain *= grain_modulation * wp.pow(0.55, float(band - source))
                amplitude = _detail_amplitude(envelope, source, uv)
                value += gain * amplitude * _translated_detail(tiles, source, uv, level, noise_seed)
            level *= 2
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
    return tuple(INDEX[material])


@lru_cache(maxsize=64)
def _field(material, map_type):
    return decode(read_record(_MODELS, INDEX[material][map_type]), map_type)


@lru_cache(maxsize=64)
def _detail_tiles(material, map_type):
    if map_type not in ("color", "normal", "roughness", "displacement"):
        return np.empty((0, 32, 32, 3), dtype=np.float32), np.empty((0, 16, 16), dtype=np.float32)
    packed = decompress(read_record(_DETAIL, DETAIL_INDEX[material][map_type]))
    bands, channels, size, envelope_size, scale = unpack_from("<BBBBf", packed)
    count = bands * size * size * channels
    values = np.frombuffer(packed, dtype=np.int8, count=count, offset=8).reshape(bands, size, size, channels)
    envelope = (
        np.frombuffer(packed, dtype=np.uint8, offset=8 + count)
        .reshape(bands, envelope_size, envelope_size)
        .astype(np.float32)
        / 85.0
    )
    values = values.astype(np.float32) * scale
    if channels == 1:
        values = np.repeat(values, 3, axis=3)
    if channels == 2:
        values = np.pad(values, ((0, 0), (0, 0), (0, 0), (0, 1)))
    return values, envelope


@lru_cache(maxsize=12)
def _edge_field(material, map_type):
    key = f"{map_type}_edges"
    if key in DETAIL_INDEX[material]:
        packed = decompress(read_record(_DETAIL, DETAIL_INDEX[material][key]))
        size, channels, scale = unpack_from("<HBf", packed)
        field = np.frombuffer(packed, dtype=np.int8, offset=7).reshape(size, size, channels)
        if channels == 1:
            field = np.repeat(field, 2, axis=2)
        return field, scale
    return np.empty((0, 0, 2), dtype=np.int8), 0.0


@lru_cache(maxsize=16)
def _layout_field(material, map_type):
    key = f"{map_type}_layout"
    if key not in DETAIL_INDEX[material]:
        return np.empty((0, 0, 0, 3), dtype=np.float32), 0
    packed = decompress(read_record(_DETAIL, DETAIL_INDEX[material][key]))
    mode, depth, height, width, channels = unpack_from("<BHHHB", packed)
    values = np.frombuffer(packed, dtype=np.uint8, offset=8).reshape(depth, height, width, channels)
    values = values.astype(np.float32) / 255.0
    if channels == 1:
        values = np.repeat(values, 3, axis=3)
    elif channels == 2:
        values = np.pad(values, ((0, 0), (0, 0), (0, 0), (0, 1)))
    return values, mode


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
    field, _ = _field(material, map_type)
    resolution = min(height, width)
    while field.shape[0] > 2 * resolution:
        size = field.shape[0] // 2
        field = field.reshape(size, 2, size, 2, 3).mean(axis=(1, 3))
    offset, phase = wp.vec2(0.0), wp.vec2(0.0)
    noise_seed = 0
    if seed is not None:
        rng = np.random.default_rng(seed)
        offset = wp.vec2(*rng.random(2))
        phase = wp.vec2(*rng.uniform(0.0, 2.0 * np.pi, 2))
        noise_seed = int(rng.integers(1, 1 << 30))
    image = wp.empty(shape, dtype=wp.vec3, device=device)
    tiles, envelope = _detail_tiles(material, map_type)
    edges, edge_scale = _edge_field(material, map_type)
    layout, layout_kind = _layout_field(material, map_type)
    wp.launch(
        _bake_reference,
        dim=shape,
        inputs=[
            wp.array(field, dtype=wp.vec3, device=image.device),
            wp.array(tiles, dtype=wp.vec3, device=image.device),
            wp.array(envelope, dtype=float, device=image.device),
            wp.array(edges, dtype=wp.vec2b, device=image.device),
            edge_scale,
            wp.array(layout, dtype=wp.vec3, device=image.device),
            layout_kind,
            offset,
            phase,
            noise_seed,
            material.startswith(("Asphalt", "Concrete", "Gravel", "Ground")),
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
