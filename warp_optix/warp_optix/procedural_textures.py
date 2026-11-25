# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable material textures generated with standard Warp launches."""

from base64 import b85decode
from struct import unpack_from
from zlib import decompress

import numpy as np
import warp as wp

from warp_optix._src.procedural_texture_data import REFERENCES

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


@wp.kernel
def _separate_waves(
    frequencies: wp.array(dtype=wp.vec2),
    cosine: wp.array(dtype=wp.vec3),
    sine: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    row_cosine: wp.array2d(dtype=wp.vec3),
    row_sine: wp.array2d(dtype=wp.vec3),
    columns: wp.array2d(dtype=wp.vec2),
):
    group, index = wp.tid()
    start = offsets[group]
    width = columns.shape[1]
    if index < width:
        u = (float(index) + 0.5) / float(width)
        phase = frequencies[start][0] * u
        columns[group, index] = wp.vec2(wp.cos(phase), wp.sin(phase))
    height = row_cosine.shape[0]
    if index < height:
        v = (float(index) + 0.5) / float(height)
        a = wp.vec3(0.0)
        b = wp.vec3(0.0)
        for i in range(start, offsets[group + 1]):
            phase = frequencies[i][1] * v
            c = wp.cos(phase)
            s = wp.sin(phase)
            a += cosine[i] * c + sine[i] * s
            b += sine[i] * c - cosine[i] * s
        row_cosine[index, group] = a
        row_sine[index, group] = b


@wp.kernel
def _bake_reference(
    row_cosine: wp.array2d(dtype=wp.vec3),
    row_sine: wp.array2d(dtype=wp.vec3),
    columns: wp.array2d(dtype=wp.vec2),
    mean: wp.vec3,
    detail: wp.vec3,
    grain: wp.vec3,
    noise_seed: int,
    srgb: bool,
    normal_map: bool,
    image: wp.array2d(dtype=wp.vec3),
):
    row, col = wp.tid()
    height, width = image.shape[0], image.shape[1]
    uv = _texel_uv(row, col, height, width)
    value = mean
    for i in range(columns.shape[0]):
        phase = columns[i, col]
        value += row_cosine[row, i] * phase[0] + row_sine[row, i] * phase[1]
    if wp.dot(detail, detail) > 0.0:
        value += _map_detail(_value_noise(uv, 48, noise_seed + 157), detail, normal_map)
    if wp.dot(grain, grain) > 0.0:
        value += _map_detail(_value_noise(uv, 128, noise_seed + 173), grain, normal_map)
    value = wp.vec3(wp.clamp(value[0], 0.0, 1.0), wp.clamp(value[1], 0.0, 1.0), wp.clamp(value[2], 0.0, 1.0))
    if srgb:
        value = wp.vec3(_srgb_to_linear(value[0]), _srgb_to_linear(value[1]), _srgb_to_linear(value[2]))
    if normal_map:
        xy = 2.0 * wp.vec2(value[0], value[1]) - wp.vec2(1.0)
        length_squared = wp.dot(xy, xy)
        xy /= wp.sqrt(wp.max(1.0, length_squared))
        value = 0.5 * wp.vec3(xy[0], xy[1], wp.sqrt(wp.max(0.0, 1.0 - length_squared))) + wp.vec3(0.5)
    image[row, col] = value


def reference_ids() -> tuple[str, ...]:
    """List the ambientCG material IDs in the order of the reference snapshot."""
    return tuple(REFERENCES)


def reference_maps(material: str) -> tuple[str, ...]:
    """List the fitted maps for an ambientCG material ID."""
    return tuple(REFERENCES[material])


def _smoothstep(low: float, high: float, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
    encoded = decompress(b85decode(REFERENCES[material][map_type]))
    channels = {"color": 3, "normal": 2}.get(map_type, 1)
    header_size = 1 + 3 * channels
    scale, *levels = unpack_from(f"<{header_size}f", encoded)
    mean, detail, grain = [levels[i : i + channels] for i in range(0, len(levels), channels)]
    modes = np.frombuffer(encoded, dtype=np.int8, offset=4 * header_size).reshape(-1, 2 + 2 * channels)
    frequencies = modes[:, :2].astype(np.float32)
    cosine = modes[:, 2 : 2 + channels].astype(np.float32) * scale
    sine = modes[:, 2 + channels :].astype(np.float32) * scale
    # Cull or fade modes that the requested image cannot resolve. Do this once
    # per bake instead of repeating the filtering for every texel.
    cycles_per_texel = np.max(np.abs(frequencies) / np.asarray((width, height), dtype=np.float32), axis=1)
    weights = 1.0 - _smoothstep(0.35, 0.5, cycles_per_texel)
    keep = weights > 0.0
    frequencies = frequencies[keep]
    cosine = cosine[keep] * weights[keep, None]
    sine = sine[keep] * weights[keep, None]
    noise_seed = 0
    if seed is not None:
        rng = np.random.default_rng(seed)
        origin = rng.random(2, dtype=np.float32)
        jitter = rng.uniform(0.5, 6.0, 2).astype(np.float32)
        phase = (2.0 * np.pi) * (frequencies @ origin)
        # Shift every map together and perturb finer waves using the same
        # phase for a given frequency, independent of each map's mode order.
        radius = np.hypot(frequencies[:, 0], frequencies[:, 1])
        phase += 0.5 * _smoothstep(3.0, 18.0, radius) * np.sin(frequencies @ jitter)
        c = np.cos(phase[:, None])
        s = np.sin(phase[:, None])
        cosine, sine = cosine * c + sine * s, sine * c - cosine * s
        noise_seed = int(rng.integers(1, 1 << 30))
    frequencies *= 2.0 * np.pi
    if channels == 1:
        mean *= 3
        detail *= 3
        grain *= 3
        cosine = np.repeat(cosine, 3, axis=1)
        sine = np.repeat(sine, 3, axis=1)
    elif channels == 2:
        mean.append(0.0)
        detail.append(0.0)
        grain.append(0.0)
        cosine = np.pad(cosine, ((0, 0), (0, 1)))
        sine = np.pad(sine, ((0, 0), (0, 1)))
    resolution = min(height, width)
    detail_gain = _smoothstep(24.0, 60.0, resolution)
    grain_gain = _smoothstep(64.0, 160.0, resolution)
    image = wp.empty(shape, dtype=wp.vec3, device=device)
    order = np.argsort(frequencies[:, 0], kind="stable")
    frequencies = frequencies[order]
    _, starts = np.unique(frequencies[:, 0], return_index=True)
    groups = len(starts)
    row_cosine = wp.empty((height, groups), dtype=wp.vec3, device=image.device)
    row_sine = wp.empty_like(row_cosine)
    columns = wp.empty((groups, width), dtype=wp.vec2, device=image.device)
    if groups:
        # Collapse waves with the same horizontal frequency. The final pass
        # reuses phases across rows and columns without per-pixel trigonometry.
        offsets = np.append(starts, len(frequencies)).astype(np.int32)
        wp.launch(
            _separate_waves,
            dim=(groups, max(height, width)),
            inputs=[
                wp.array(frequencies, dtype=wp.vec2, device=image.device),
                wp.array(cosine[order], dtype=wp.vec3, device=image.device),
                wp.array(sine[order], dtype=wp.vec3, device=image.device),
                wp.array(offsets, dtype=int, device=image.device),
            ],
            outputs=[row_cosine, row_sine, columns],
            device=image.device,
        )
    wp.launch(
        _bake_reference,
        dim=shape,
        inputs=[
            row_cosine,
            row_sine,
            columns,
            wp.vec3(*mean),
            wp.vec3(*(v * detail_gain for v in detail)),
            wp.vec3(*(v * grain_gain for v in grain)),
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
