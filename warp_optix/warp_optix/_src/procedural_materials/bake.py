# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tileable material textures generated with standard Warp launches."""

from functools import lru_cache
from pathlib import Path
from struct import unpack_from
from zlib import decompress

import numpy as np
import warp as wp

from .bark_surfaces import bark_surface
from .carbon_fiber import carbon_fiber
from .chipboard import chipboard
from .configurable_metals import METAL_MAPS, MetalSurface, bake_metal, pack_metal
from .construction_surfaces import construction_surface
from .functional_surfaces import functional_surface
from .industrial_surfaces import industrial_surface
from .knotted_softwood import knotted_softwood
from .lawn_leather_stone import fractured_marble, lake_ice, lawn_surface, leather_panel, onyx_layers
from .material_blend import blend_materials
from .mineral_surfaces import perforated_sheet, riveted_metal, tactile_paving, terrazzo
from .natural_and_foil import natural_or_foil
from .organic_surfaces import organic_surface
from .plastic_surfaces import PLASTIC_PRESETS, PlasticSurface, plastic_configuration, plastic_surface
from .reclaimed_planks import reclaimed_planks
from .relief import _normal_for_gltf, _normal_from_relief
from .roof_and_mosaic import roof_or_mosaic
from .sponge_and_paper import sponge_or_paper
from .textile_surfaces import textile_surface
from .texture_coefficients import (
    ANALYTIC_MATERIALS,
    BUILTIN_SURFACES,
    DIRECT_REFERENCES,
    SURFACE_PRESETS,
    decode,
    read_index,
    read_record,
)
from .wall_concrete import wall_concrete
from .weathered_surfaces import weathered_surface

__all__ = ["bake", "bake_gltf", "bake_layered_gltf", "brick", "marble", "reference_ids", "reference_maps", "wood"]

_package = Path(__file__).resolve().parents[2]
_catalog = _package / "_src" / "material_catalog"
if not (_catalog / "models" / "index.json").is_file():
    _catalog = _package.parents[1] / "material_catalog"
_MODELS = _catalog / "models"
_DETAIL = _catalog / "detail"
INDEX = read_index(_MODELS) if (_MODELS / "index.json").is_file() else {}
_SURFACE_CHANNELS = ("color", "normal", "roughness", "displacement", "occlusion", "metalness", "opacity")
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
def _surface_noise(seed: int, uv: wp.vec2, nx: int, ny: int):
    return wp.pnoise(wp.rand_init(seed), wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)


@wp.func
def _wood_surface(uv: wp.vec2, seed: int, style: int, resolution: float):
    # A longitudinal cut through growth rings. Elliptical knots bend the
    # surrounding fibers, and expose end grain near their centers.
    warp_y = 0.014 * _surface_noise(seed, uv, 3, 5)
    if style == 2:
        warp_y *= 3.0
    p = wp.vec2(uv[0], uv[1] + warp_y)
    cells = wp.vec2(2.0 * p[0], 4.0 * p[1])
    ix = int(wp.floor(cells[0]))
    iy = int(wp.floor(cells[1]))
    state = wp.rand_init(seed + 17, (ix & 1) + 2 * (iy & 3))
    center = wp.vec2(wp.randf(state, 0.26, 0.74), wp.randf(state, 0.30, 0.70))
    delta = cells - wp.vec2(float(ix), float(iy)) - center
    radius = wp.length(wp.vec2(delta[0] * 4.8, delta[1] * 3.7))
    influence = 1.0 - wp.smoothstep(0.12, 1.0, radius)
    if style == 0:
        influence *= 0.23
    p += wp.vec2(0.0, 0.09 * delta[1] * influence)
    annual = 42.0 * p[1] + 0.72 * _surface_noise(seed + 2, p, 3, 17)
    if style == 2:
        # Flatsawn cuts cross a curved ring surface and expose nested arches.
        u = wp.sin(wp.PI * (p[0] + 0.15))
        v = wp.sin(2.0 * wp.PI * (p[1] + 0.08))
        annual = 23.0 * wp.sqrt(0.014 + v * v + 0.44 * u * u) + 2.4 * _surface_noise(seed + 3, p, 3, 4)
    band = 0.5 + 0.5 * wp.cos(2.0 * wp.PI * annual)
    latewood = band * band * band * band
    fiber = _surface_noise(seed + 5, p, 21, 360) * wp.smoothstep(180.0, 720.0, resolution)
    pores = _surface_noise(seed + 6, p, 97, 970) * wp.smoothstep(485.0, 1940.0, resolution)
    knot = influence * (1.0 - wp.smoothstep(0.11, 0.31, radius))
    knot_ring = wp.sin(78.0 * radius + 2.1 * fiber) * knot
    tone = 0.029 - 0.075 * latewood + 0.048 * fiber + 0.036 * pores - 0.24 * knot + 0.057 * knot_ring
    # Split earlywood into thin uneven fiber bundles; elongated open pores
    # break the smooth bands where the cut exposes vessel openings.
    strand = 0.5 + 0.5 * wp.cos(2.0 * wp.PI * (annual * 7.0 + 0.84 * _surface_noise(seed + 8, p, 9, 91)))
    cut = wp.smoothstep(0.26, 0.57, _surface_noise(seed + 9, p, 141, 811))
    micro = (0.042 * strand ** 5.0 + 0.072 * cut) * wp.smoothstep(300.0, 1300.0, resolution)
    tone -= micro
    tone += 0.038 * _surface_noise(seed + 10, p, 3, 29)
    relief = 0.53 - 0.30 * micro - 0.032 * latewood + 0.023 * fiber + 0.015 * pores - 0.10 * knot + 0.02 * knot_ring
    base = wp.vec3(0.65, 0.52, 0.37)
    if style == 1:
        base = wp.vec3(0.54, 0.42, 0.31)
    elif style == 2:
        base = wp.vec3(0.56, 0.45, 0.34)
    return base + wp.vec3(tone, 0.90 * tone, 0.74 * tone), relief, (0.765 if style == 2 else 0.80) - 0.64 * tone, 1.0


@wp.func
def _paver_surface(uv: wp.vec2, seed: int, resolution: float):
    # Distort a staggered grid before evaluating the signed stone boundary.
    # Its chipped rim stays continuous across all five maps.
    bend = _surface_noise(seed, uv, 5, 5)
    chip = _surface_noise(seed + 7, uv, 84, 62)
    fine = _surface_noise(seed + 8, uv, 360, 210) * wp.smoothstep(180.0, 720.0, resolution)
    grit = _surface_noise(seed + 9, uv, 950, 650) * wp.smoothstep(475.0, 1900.0, resolution)
    y = 15.0 * uv[1] + 0.57 * bend + 0.16 * _surface_noise(seed + 2, uv, 26, 13)
    row = int(wp.floor(y))
    r = wp.rand_init(seed + 19, (row % 15 + 15) % 15)
    x = 20.0 * uv[0] + wp.randf(r) + 0.64 * _surface_noise(seed + 1, uv, 8, 7)
    x += 0.29 * _surface_noise(seed + 3, uv, 17, 6)
    col = int(wp.floor(x))
    stone = wp.rand_init(seed + 31, ((col % 20 + 20) % 20) + 20 * ((row % 15 + 15) % 15))
    tint = wp.randf(stone, -0.062, 0.062)
    radius = wp.randf(stone, 0.09, 0.21)
    q = wp.vec2(wp.abs(x - wp.floor(x) - 0.5) - 0.47 + radius, wp.abs(y - wp.floor(y) - 0.5) - 0.46 + radius)
    inside = radius - wp.length(wp.vec2(wp.max(q[0], 0.0), wp.max(q[1], 0.0))) - wp.min(wp.max(q[0], q[1]), 0.0)
    inside += 0.081 * chip + 0.031 * fine
    bevel = wp.smoothstep(-0.035, 0.108, inside)
    face = wp.smoothstep(-0.015, 0.034, inside)
    mineral = _surface_noise(seed + 4, uv, 51, 42)
    patina = _surface_noise(seed + 5, uv, 12, 11)
    moss = wp.smoothstep(-0.12, 0.42, patina + 0.30 * fine)
    soil = wp.lerp(wp.vec3(0.225, 0.187, 0.131), wp.vec3(0.315, 0.327, 0.135), moss)
    rock = wp.vec3(0.548, 0.506, 0.391) + wp.vec3(tint + 0.118 * mineral + 0.06 * patina)
    color = wp.lerp(soil, rock, face) + wp.vec3(0.11 * fine + 0.098 * grit)
    relief = 0.45 + bevel * (0.25 + tint) + 0.065 * mineral + 0.031 * fine + 0.021 * grit
    return color, relief, 0.74 - 0.13 * face + 0.09 * fine, 0.72 + 0.26 * bevel


@wp.func
def _concrete_chip(uv: wp.vec2, seed: int, frequency: int, aa: float):
    grid = uv * float(frequency)
    cell = wp.vec2(wp.floor(grid[0]), wp.floor(grid[1]))
    ix = (int(cell[0]) % frequency + frequency) % frequency
    iy = (int(cell[1]) % frequency + frequency) % frequency
    rng = wp.rand_init(seed, ix + frequency * iy)
    radius = wp.randf(rng, 0.08, 0.38)
    jitter = (0.49 - radius) * wp.vec2(wp.randf(rng, -1.0, 1.0), wp.randf(rng, -1.0, 1.0))
    q = grid - cell - wp.vec2(0.5) - jitter
    angle = wp.randf(rng, 0.0, wp.PI)
    u = wp.cos(angle) * q[0] + wp.sin(angle) * q[1]
    v = -wp.sin(angle) * q[0] + wp.cos(angle) * q[1]
    dist = wp.max(wp.abs(u), wp.abs(v * 1.48 + 0.33 * u))
    dist = wp.max(dist, 0.83 * wp.abs(u - v))
    mask = 1.0 - wp.smoothstep(radius - aa, radius + aa, dist)
    pick = wp.randf(rng)
    pigment = wp.vec3(0.23, 0.215, 0.205)
    if pick > 0.47:
        pigment = wp.vec3(0.58, 0.38, 0.23)
    if pick > 0.72:
        pigment = wp.vec3(0.87, 0.78, 0.65)
    return mask, pigment + wp.vec3(wp.randf(rng, -0.06, 0.06))


@wp.func
def _concrete_surface(uv: wp.vec2, seed: int, resolution: float):
    fleck = _surface_noise(seed, uv, 640, 640) * wp.smoothstep(320.0, 1280.0, resolution)
    grade = _surface_noise(seed + 18, uv, 86, 99)
    binder = 0.085 * _surface_noise(seed + 11, uv, 43, 43) + 0.13 * fleck
    color = wp.vec3(0.355, 0.333, 0.315) + wp.vec3(binder)
    relief = 0.48 + 0.10 * fleck
    p = uv + 0.0016 * wp.vec2(_surface_noise(seed + 2, uv, 73, 71), _surface_noise(seed + 3, uv, 71, 73))
    p += 0.0007 * wp.vec2(fleck, _surface_noise(seed + 26, uv, 280, 280))
    fracture = wp.abs(_surface_noise(seed + 16, uv, 126, 114))
    coverage = float(0.0)
    for scale in range(3):
        freq = 184 >> scale
        mask, mineral = _concrete_chip(p, seed + scale * 13, freq, 0.68 * float(freq) / resolution)
        color = wp.lerp(color, mineral * (0.75 + 0.29 * fracture) + wp.vec3(0.13 * fleck + 0.05 * grade), mask)
        relief = wp.lerp(relief, 0.56 + 0.05 * fleck, mask)
        coverage = wp.max(coverage, mask)
    return color, relief, 0.71 - 0.11 * coverage + 0.10 * fleck, 0.91 + 0.08 * coverage


@wp.func
def _parquet_surface(uv: wp.vec2, seed: int, resolution: float):
    grid = uv * 10.0
    cell = wp.vec2(wp.floor(grid[0]), wp.floor(grid[1]))
    local = grid - cell
    ix, iy = (int(cell[0]) % 10 + 10) % 10, (int(cell[1]) % 10 + 10) % 10
    if (ix + iy) % 2 == 1:
        local = wp.vec2(local[1], local[0])
    strip = int(wp.floor(local[1] * 7.0))
    rng = wp.rand_init(seed, ix + 10 * iy + 100 * strip)
    p = wp.vec2(wp.randf(rng) + local[0] * 0.31, wp.randf(rng) + local[1] * 0.25)
    color, relief, rough, ao = _wood_surface(p, seed, 0, resolution)
    shade = wp.randf(rng, -0.083, 0.083)
    across = local[1] * 7.0 - float(strip)
    edge = wp.min(wp.min(across, 1.0 - across), 7.0 * wp.min(local[0], 1.0 - local[0]))
    bevel = wp.smoothstep(0.0, wp.max(0.034, 35.0 / resolution), edge)
    return (color + wp.vec3(shade + 0.113, shade + 0.035, shade + 0.023)) * (0.76 + 0.24 * bevel), relief - 0.09 * (1.0 - bevel), 0.32 + 0.36 * (rough - 0.80) - 0.13 * shade, ao


@wp.func
def _brick_wall_surface(uv: wp.vec2, seed: int, resolution: float):
    grade = _surface_noise(seed + 3, uv, 50, 57)
    sand = _surface_noise(seed + 5, uv, 622, 650) * wp.smoothstep(310.0, 1240.0, resolution)
    y = uv[1] * 24.0 + 0.037 * grade
    row = int(wp.floor(y))
    x = uv[0] * 8.0 + 0.5 * float(row & 1) + 0.027 * grade
    col = int(wp.floor(x))
    rng = wp.rand_init(seed, (col % 8 + 8) % 8 + 8 * ((row % 24 + 24) % 24))
    fx, fy = x - wp.floor(x), y - wp.floor(y)
    joint = wp.min(3.0 * wp.min(fx, 1.0 - fx), wp.min(fy, 1.0 - fy)) + 0.025 * sand
    face = wp.smoothstep(0.052, 0.052 + wp.max(0.018, 16.0 / resolution), joint)
    bevel = wp.smoothstep(0.04, 0.12, joint)
    pigment = wp.randf(rng, -0.085, 0.085) + 0.091 * grade
    clay = wp.vec3(0.58, 0.305, 0.212) + wp.vec3(pigment, 0.76 * pigment, 0.55 * pigment)
    scorch = _surface_noise(seed + 18, uv, 22, 47) + 0.40 * grade
    inclusions = wp.smoothstep(0.25, 0.60, grade + 0.72 * sand)
    clay += wp.vec3(0.12, 0.083, 0.052) * scorch - wp.vec3(0.13) * inclusions
    color = wp.lerp(wp.vec3(0.68, 0.63, 0.55), clay, face) + wp.vec3(0.087 * sand)
    return color, 0.45 + 0.22 * bevel + 0.04 * grade + 0.027 * sand, 0.77 - 0.17 * face + 0.06 * sand, 0.79 + 0.2 * bevel



@wp.func
def _mineral_finish(uv: wp.vec2, seed: int, resolution: float):
    cloud = _surface_noise(seed, uv, 5, 7) + 0.45 * _surface_noise(seed + 3, uv, 17, 19)
    stipple = _surface_noise(seed + 4, uv, 211, 201) * wp.smoothstep(110.0, 420.0, resolution)
    micro = _surface_noise(seed + 5, uv, 670, 700) * wp.smoothstep(350.0, 1400.0, resolution)
    pore = wp.smoothstep(0.27, 0.55, stipple + 0.33 * micro)
    return cloud, stipple, micro, pore


@wp.func
def _finish_surface(uv: wp.vec2, seed: int, resolution: float, palette: wp.vec3, contrast: float,
                    base_roughness: float, formwork: float):
    cloud, stipple, micro, pore = _mineral_finish(uv, seed, resolution)
    cast = _surface_noise(seed + 23, uv, 90, 8) + 0.45 * _surface_noise(seed + 9, uv, 210, 28)
    if formwork < 0.0:
        grain_uv = uv + 0.0007 * wp.vec2(stipple, micro)
        coverage, stone = _concrete_chip(grain_uv, seed + 29, 154, 105.0 / resolution)
        exposed = coverage * (0.65 + 0.35 * micro)
        color = palette + wp.vec3(0.087 * stipple + 0.24 * micro)
        color += -formwork * exposed * wp.vec3(0.10 + 0.38 * stone[0])
        return color, 0.48 + 0.11 * exposed + 0.075 * micro, 0.86 - 0.1 * exposed, 0.91 + 0.08 * exposed
    tone = contrast * (0.58 * cloud + 0.36 * stipple + 0.31 * micro - 0.62 * pore + formwork * cast)
    height = 0.55 + 0.04 * cloud + 0.025 * stipple + 0.030 * micro - 0.12 * pore + 0.07 * formwork * cast
    return palette + wp.vec3(tone), height, base_roughness + 0.087 * pore - 0.047 * cloud, 0.96 - 0.16 * pore


@wp.func
def _tile_surface(uv: wp.vec2, seed: int, resolution: float, palette: wp.vec3, contrast: float,
                  base_roughness: float, columns: int, rows: int, stagger: float):
    cloud, stipple, micro, pore = _mineral_finish(uv, seed, resolution)
    grid_y = uv[1] * float(rows)
    row = int(wp.floor(grid_y))
    grid_x = uv[0] * float(columns) + stagger * float(row & 1)
    col = int(wp.floor(grid_x))
    cell = (col % columns + columns) % columns + columns * ((row % rows + rows) % rows)
    state = wp.rand_init(seed + 81, cell)
    clay_brick = rows >= 16 and base_roughness >= 0.60
    grout_width = wp.randf(state, 0.009, 0.013)
    if clay_brick:
        grout_width *= 4.0
    x, y = grid_x - wp.floor(grid_x), grid_y - wp.floor(grid_y)
    d = wp.min(wp.min(x, 1.0 - x), wp.min(y, 1.0 - y))
    chip = 0.08 * contrast * wp.smoothstep(0.30, 0.61, stipple + 0.5 * cloud)
    aa = float(wp.max(columns, rows)) / resolution
    face = wp.smoothstep(grout_width - aa, grout_width + aa, d - chip)
    bevel = wp.smoothstep(0.35 * grout_width, grout_width * 1.8 + aa, d - chip)
    tone = contrast * (wp.randf(state, -0.57, 0.57) + 0.44 * cloud + 0.45 * stipple + 0.24 * micro - 0.76 * pore)
    tile_color = palette + wp.vec3(tone)
    grout_color = wp.lerp(wp.vec3(0.26, 0.24, 0.22), palette, 0.38)
    if clay_brick:
        grout_color = wp.vec3(0.65, 0.61, 0.52)
    grout_color += wp.vec3(0.08 * stipple)
    relief = wp.lerp(0.43, 0.66 + 0.04 * wp.sin(wp.PI * x) * wp.sin(wp.PI * y), bevel)
    relief += contrast * (0.16 * stipple + 0.065 * micro - 0.35 * pore) * face
    return wp.lerp(grout_color, tile_color, face), relief, wp.lerp(0.85, base_roughness + 0.11 * pore, face), 0.76 + 0.24 * bevel



@wp.func
def _analytic_surface(uv: wp.vec2, seed: int, style: int, resolution: float):
    if style == 4:
        return _concrete_surface(uv, seed, resolution)
    if style == 5:
        return _parquet_surface(uv, seed, resolution)
    if style == 6:
        return _brick_wall_surface(uv, seed, resolution)
    if style == 3:
        return _paver_surface(uv, seed, resolution)
    return _wood_surface(uv, seed, style, resolution)


@wp.func
def _surface_maps(
    uv: wp.vec2, resolution: float, style: int, seed: int, palette: wp.vec3,
    contrast: float, base_roughness: float, pattern: float, columns: int, rows: int,
):
    # Some generators warp UVs beyond the unit square. Start from a canonical
    # tile phase so offsets by whole tiles reproduce exactly the same surface.
    uv = uv - wp.vec2(wp.floor(uv[0]), wp.floor(uv[1]))
    if style >= 0:
        if columns > 0:
            color, height, roughness, ao = _tile_surface(
                uv, seed, resolution, palette, contrast, base_roughness, columns, rows, pattern
            )
        elif style >= 7:
            color, height, roughness, ao = _finish_surface(
                uv, seed, resolution, palette, contrast, base_roughness, pattern
            )
        else:
            color, height, roughness, ao = _analytic_surface(uv, seed, style, resolution)
        return color, height, roughness, ao, 0.0, 1.0
    if style == -1:
        color, height, roughness = terrazzo(uv, seed, resolution)
        return color, height, roughness, 1.0, 0.0, 1.0
    if style == -2:
        color, height, roughness, alpha = perforated_sheet(uv, seed, resolution)
        return color, height, roughness, 1.0, 1.0, alpha
    if style == -3:
        color, height, roughness, ao = tactile_paving(uv, seed, resolution)
        return color, height, roughness, ao, 0.0, 1.0
    if style == -4:
        color, height, roughness = riveted_metal(uv, seed, resolution)
        return color, height, roughness, 1.0, 1.0, 1.0
    if style >= -7:
        color, height, roughness, ao, alpha = textile_surface(uv, seed, resolution, -style - 5)
        return color, height, roughness, ao, 0.0, alpha
    if style >= -9:
        color, height, roughness, ao = organic_surface(uv, seed, resolution, -style - 8)
        return color, height, roughness, ao, 0.0, 1.0
    if style >= -11:
        color, height, roughness, ao = bark_surface(uv, seed, resolution, -style - 10)
    elif style >= -14:
        color, height, roughness, ao = weathered_surface(uv, seed, resolution, -style - 12)
    elif style >= -17:
        color, height, roughness, ao = wall_concrete(uv, seed, resolution, -style - 15)
    elif style >= -20:
        color, height, roughness, ao = roof_or_mosaic(uv, seed, resolution, -style - 18)
    elif style == -21:
        color, height, roughness, ao = knotted_softwood(uv, seed, resolution)
    elif style == -22:
        return carbon_fiber(uv, seed, resolution)
    elif style == -23:
        return reclaimed_planks(uv, seed, resolution)
    elif style >= -26:
        return industrial_surface(uv, seed, resolution, -style - 24)
    elif style >= -29:
        return functional_surface(uv, seed, resolution, -style - 27)
    elif style >= -31:
        return natural_or_foil(uv, seed, resolution, -style - 30)
    elif style == -32:
        return lawn_surface(uv, seed, resolution)
    elif style == -33:
        return leather_panel(uv, seed, resolution)
    elif style == -34:
        return onyx_layers(uv, seed, resolution)
    elif style == -35:
        return fractured_marble(uv, seed, resolution)
    elif style == -36:
        return lake_ice(uv, seed, resolution)
    elif style == -37:
        return chipboard(uv, seed, resolution)
    elif style >= -39:
        return sponge_or_paper(uv, seed, resolution, -style - 38)
    elif style >= -47:
        return plastic_surface(uv, seed, resolution, -style - 40, palette, base_roughness)
    else:
        return construction_surface(uv, seed, resolution, -style - 48)
    return color, height, roughness, ao, 0.0, 1.0


@wp.func
def _linear_color(color: wp.vec3) -> wp.vec3:
    return wp.vec3(
        _srgb_to_linear(wp.clamp(color[0], 0.0, 1.0)),
        _srgb_to_linear(wp.clamp(color[1], 0.0, 1.0)),
        _srgb_to_linear(wp.clamp(color[2], 0.0, 1.0)),
    )


@wp.kernel(enable_backward=False)
def _bake_surface(
    style: int, seed: int, palette: wp.vec3, contrast: float, base_roughness: float, pattern: float,
    columns: int, rows: int, channel: int, image: wp.array2d(dtype=wp.vec3), relief: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    h, w = wp.max(image.shape[0], relief.shape[0]), wp.max(image.shape[1], relief.shape[1])
    color, height, roughness, ao, metal, alpha = _surface_maps(
        _texel_uv(row, col, h, w), float(wp.min(h, w)), style, seed,
        palette, contrast, base_roughness, pattern, columns, rows,
    )
    if channel == 1:
        relief[row, col] = wp.clamp(height, 0.0, 1.0)
        return
    result = wp.vec3(height)
    if channel == 0:
        result = _linear_color(color)
    elif channel == 2:
        result = wp.vec3(roughness)
    elif channel == 4:
        result = wp.vec3(ao)
    elif channel == 5:
        result = wp.vec3(metal)
    elif channel == 6:
        result = wp.vec3(alpha)
    image[row, col] = wp.vec3(wp.clamp(result[0], 0.0, 1.0), wp.clamp(result[1], 0.0, 1.0), wp.clamp(result[2], 0.0, 1.0))



@wp.kernel(enable_backward=False)
def _height_normals(relief: wp.array2d(dtype=float), strength: float, image: wp.array2d(dtype=wp.vec3)):
    row, col = wp.tid()
    image[row, col] = _normal_from_relief(relief, row, col, strength)


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
    """List the ambientCG and built-in material IDs."""
    return tuple(dict.fromkeys((*INDEX, *BUILTIN_SURFACES, *DIRECT_REFERENCES, *PLASTIC_PRESETS)))


def reference_maps(material: str | MetalSurface | PlasticSurface) -> tuple[str, ...]:
    """List the maps available for a material."""
    if isinstance(material, MetalSurface):
        return METAL_MAPS
    if isinstance(material, PlasticSurface) or material in PLASTIC_PRESETS:
        return _SURFACE_CHANNELS[:4]
    if material in DIRECT_REFERENCES:
        return (*_SURFACE_CHANNELS[:4], *DIRECT_REFERENCES[material][1])
    if material in BUILTIN_SURFACES:
        return _SURFACE_CHANNELS[:5]
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


def _surface_configuration(material, seed):
    state = 173 if seed is None else int(np.random.default_rng(seed).integers(1, 1 << 30))
    plastic = material if isinstance(material, PlasticSurface) else PLASTIC_PRESETS.get(material)
    if plastic is not None:
        style, palette, roughness, strength = plastic_configuration(plastic)
        return [style, state, wp.vec3(*palette), 0.0, roughness, 0.0, 0, 0], strength
    preset = SURFACE_PRESETS.get(material, ((0.0, 0.0, 0.0), 0.0, 0.0, 0.0))
    palette, contrast, roughness, pattern, *grid = preset
    columns, rows = grid or (0, 0)
    if material in DIRECT_REFERENCES:
        style, _, strength = DIRECT_REFERENCES[material]
    else:
        style = ANALYTIC_MATERIALS.index(material)
        if style < 7:
            strength = (0.0009, 0.0008, 0.0007, 0.004, 0.0032, 0.00028, 0.003)[style]
        else:
            strength = 0.0013 if columns else 0.0009
    return [style, state, wp.vec3(*palette), contrast, roughness, pattern, columns, rows], strength


def _uses_surface_kernel(material):
    return (
        isinstance(material, PlasticSurface)
        or material in PLASTIC_PRESETS
        or material in ANALYTIC_MATERIALS
        or material in DIRECT_REFERENCES
    )


def bake(
    material: str | MetalSurface | PlasticSurface,
    shape: tuple[int, int],
    *,
    map_type: str = "color",
    device: wp.DeviceLike = None,
    seed: int | None = None,
) -> wp.array:
    """Bake a material texture into a Warp array of shape (height, width).

    Color is linear RGB. Normals are OpenGL tangent-space normals encoded
    in [0, 1]. Scalar maps repeat the normalized value in all three output
    channels. Displacement is unitless. Pass the same non-negative seed for
    every map of a variant. None gives the closest fit to the reference.
    """
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("Texture dimensions must be positive.")
    if isinstance(material, MetalSurface):
        return bake_metal(material, shape, map_type, device, seed)
    if _uses_surface_kernel(material):
        if map_type not in reference_maps(material):
            raise KeyError(map_type)
        channel = _SURFACE_CHANNELS.index(map_type)
        image = wp.empty(shape, dtype=wp.vec3, device=device)
        params, strength = _surface_configuration(material, seed)
        if channel == 1:
            relief = wp.empty(shape, dtype=float, device=image.device)
            unused = wp.empty((0, 0), dtype=wp.vec3, device=image.device)
            wp.launch(_bake_surface, dim=shape, inputs=[*params, channel], outputs=[unused, relief], device=image.device)
            wp.launch(_height_normals, dim=shape, inputs=[relief, strength], outputs=[image], device=image.device)
        else:
            unused = wp.empty((0, 0), dtype=float, device=image.device)
            wp.launch(_bake_surface, dim=shape, inputs=[*params, channel], outputs=[image, unused], device=image.device)
        return image
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
    packed[1, row, col] = _rgba8(_normal_for_gltf(normal[row, col]), 1.0)
    packed[2, row, col] = _rgba8(wp.vec3(red, rough[row, col][0], blue), 1.0)


@wp.kernel(enable_backward=False)
def _bake_packed_surface(
    style: int, seed: int, palette: wp.vec3, contrast: float, base_roughness: float, pattern: float,
    columns: int, rows: int, has_occlusion: bool, color_gain: float,
    relief: wp.array2d(dtype=float), packed: wp.array3d(dtype=wp.vec4ub),
):
    row, col = wp.tid()
    h, w = relief.shape[0], relief.shape[1]
    color, height, roughness, ao, metal, alpha = _surface_maps(
        _texel_uv(row, col, h, w), float(wp.min(h, w)), style, seed,
        palette, contrast, base_roughness, pattern, columns, rows,
    )
    if not has_occlusion:
        ao = 1.0
    packed[0, row, col] = _rgba8(color_gain * _linear_color(color), wp.clamp(alpha, 0.0, 1.0))
    packed[2, row, col] = _rgba8(wp.vec3(wp.clamp(ao, 0.0, 1.0), wp.clamp(roughness, 0.0, 1.0), metal), 1.0)
    relief[row, col] = wp.clamp(height, 0.0, 1.0)


@wp.kernel(enable_backward=False)
def _pack_relief_normals(
    relief: wp.array2d(dtype=float), strength: float, packed: wp.array3d(dtype=wp.vec4ub),
):
    row, col = wp.tid()
    normal = _normal_from_relief(relief, row, col, strength)
    packed[1, row, col] = _rgba8(_normal_for_gltf(normal), 1.0)


def bake_gltf(
    material: str | MetalSurface | PlasticSurface,
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
    A seed generates a consistent variant across all maps. Packed normals
    use the mesh UV frame, where positive green points along increasing V.
    """
    # Dark foam occupies only a handful of linear RGBA8 levels. Store its
    # albedo at a higher scale and compensate with the glTF color factor.
    color_gain = 8.0 if material == "AcousticFoam001" else 1.0
    packed = _bake_gltf_arrays(material, shape, device, seed, color_gain=color_gain)
    maps = reference_maps(material)
    kwargs = _gltf_properties(first_texture_index, "occlusion" in maps, "opacity" in maps)
    if color_gain != 1.0:
        kwargs["base_color"] = (1.0 / color_gain,) * 3 + (1.0,)
    return list(packed.numpy()), kwargs


def _bake_gltf_arrays(material, shape, device, seed, *, color_gain=1.0):
    if isinstance(material, MetalSurface):
        if min(shape) < 1:
            raise ValueError("Texture dimensions must be positive.")
        return pack_metal(material, shape, device, seed)
    maps = reference_maps(material)

    def generate(map_type: str) -> wp.array:
        return bake(material, shape, map_type=map_type, device=device, seed=seed)

    has_occlusion = "occlusion" in maps
    has_metal = "metalness" in maps
    has_opacity = "opacity" in maps
    if _uses_surface_kernel(material):
        if min(shape) < 1:
            raise ValueError("Texture dimensions must be positive.")
        packed = wp.empty((3, *shape), dtype=wp.vec4ub, device=device)
        relief = wp.empty(shape, dtype=float, device=packed.device)
        params, strength = _surface_configuration(material, seed)
        wp.launch(
            _bake_packed_surface, dim=shape, inputs=[*params, has_occlusion, color_gain],
            outputs=[relief, packed], device=packed.device,
        )
        wp.launch(_pack_relief_normals, dim=shape, inputs=[relief, strength], outputs=[packed], device=packed.device)
    else:
        base = generate("color")
        normal = generate("normal")
        rough = generate("roughness")
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
    return packed


def _gltf_properties(first_texture_index, has_occlusion, has_opacity):
    kwargs: dict[str, dict[str, int] | str] = {
        "base_color_texture": {"index": first_texture_index},
        "normal_texture": {"index": first_texture_index + 1},
        "metallic_roughness_texture": {"index": first_texture_index + 2},
    }
    if has_occlusion:
        kwargs["occlusion_texture"] = {"index": first_texture_index + 2}
    if has_opacity:
        kwargs["alpha_mode"] = "MASK"
    return kwargs

def bake_layered_gltf(
    base: str | MetalSurface | PlasticSurface,
    overlay: str | MetalSurface | PlasticSurface,
    shape: tuple[int, int],
    *,
    mask: np.ndarray | wp.array | None = None,
    frequency: int = 5,
    threshold: float = 0.5,
    softness: float = 0.20,
    height_blend: float = 0.0,
    first_texture_index: int = 0,
    seed: int | None = None,
    device: wp.DeviceLike = None,
) -> tuple[list[np.ndarray], dict]:
    """Bake two materials into one glTF texture set.

    A custom float mask of shape (height, width) blends from base at zero to
    overlay at one. With no mask, periodic noise generates the blend weights.
    Frequency, threshold, and softness control that mask. A positive
    height_blend biases the transition toward the higher local displacement.
    """
    if min(shape) < 1:
        raise ValueError("Texture dimensions must be positive.")
    if mask is None and (frequency < 1 or not 0 <= threshold <= 1 or not 0 < softness <= 1):
        raise ValueError("Use a positive frequency, threshold in [0, 1], and softness in (0, 1].")
    if not np.isfinite(height_blend) or height_blend < 0:
        raise ValueError("height_blend must be finite and non-negative.")
    if mask is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError("The blend mask must match the requested texture shape.")
    if isinstance(mask, wp.array) and mask.dtype != wp.float32:
        raise TypeError("Warp blend masks must have dtype wp.float32.")

    base_maps, overlay_maps = reference_maps(base), reference_maps(overlay)
    if height_blend and ("displacement" not in base_maps or "displacement" not in overlay_maps):
        raise ValueError("Height blending requires displacement in both materials.")

    base_data = _bake_gltf_arrays(base, shape, device, seed)
    device = base_data.device
    overlay_data = _bake_gltf_arrays(overlay, shape, device, seed)
    if mask is None:
        weights = wp.empty((0, 0), dtype=float, device=device)
    elif isinstance(mask, wp.array):
        weights = mask.to(device)
    else:
        weights = wp.array(mask, dtype=float, device=device)
    if height_blend:
        base_height = bake(base, shape, map_type="displacement", device=device, seed=seed)
        overlay_height = bake(overlay, shape, map_type="displacement", device=device, seed=seed)
    else:
        base_height = overlay_height = wp.empty((0, 0), dtype=wp.vec3, device=device)
    mask_seed = 173 if seed is None else int(np.random.default_rng(seed).integers(1, 1 << 30))
    output = wp.empty_like(base_data)
    wp.launch(
        blend_materials,
        dim=shape,
        inputs=[
            base_data, overlay_data, weights, base_height, overlay_height,
            mask_seed, frequency, threshold, softness, height_blend,
        ],
        outputs=[output],
        device=device,
    )
    available = set(base_maps) | set(overlay_maps)
    kwargs = _gltf_properties(first_texture_index, "occlusion" in available, "opacity" in available)
    return list(output.numpy()), kwargs
