# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configurable, tileable metallic surfaces."""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .relief import _normal_from_relief

# Normal-incidence sRGB reflectance for clean conductors from Filament's material
# reference. Stainless steel and chromium use representative neutral values.
# https://github.com/google/filament/blob/main/docs/Filament.md.html
_METAL_SRGB = {
    "aluminum": (0.91, 0.92, 0.92),
    "brass": (0.98, 0.90, 0.59),
    "chromium": (0.82, 0.83, 0.85),
    "copper": (0.97, 0.74, 0.62),
    "gold": (1.0, 0.85, 0.57),
    "iron": (0.77, 0.78, 0.78),
    "platinum": (0.83, 0.81, 0.78),
    "silver": (0.97, 0.96, 0.91),
    "stainless_steel": (0.79, 0.80, 0.81),
    "titanium": (0.76, 0.73, 0.69),
}
# Perceptual roughness, brushing amplitude, and default scratch amount.
_FINISHES = {
    "polished": (0.075, 0.0, 0.0),
    "brushed": (0.29, 0.75, 0.16),
    "scratched": (0.36, 0.22, 0.80),
}
METAL_MAPS = ("color", "normal", "roughness", "displacement", "metalness")


@dataclass(frozen=True, slots=True)
class MetalSurface:
    """Configure a conductor for the procedural texture bakers.

    Direction is in degrees, measured clockwise from U toward increasing V.
    Scratch density is the number of placement cells across one tile. Amount
    is in [0, 1]. A custom reflectance overrides the named metal in linear RGB.
    """

    metal: str = "aluminum"
    finish: str = "brushed"
    roughness: float | None = None
    scratches: float | None = None
    scratch_density: int = 10
    direction: float = 0.0
    reflectance: tuple[float, float, float] | None = None

    def __post_init__(self):
        if self.reflectance is None and self.metal not in _METAL_SRGB:
            raise ValueError(f"Unknown metal {self.metal!r}. Choose from {', '.join(metal_names())} or supply reflectance.")
        if self.reflectance is not None:
            if len(self.reflectance) != 3 or not all(0.0 <= x <= 1.0 for x in self.reflectance):
                raise ValueError("Metal reflectance must contain three linear RGB values in [0, 1].")
        if self.finish not in _FINISHES:
            raise ValueError(f"Unknown finish {self.finish!r}. Choose from {', '.join(_FINISHES)}.")
        if self.roughness is not None and not 0.0 <= self.roughness <= 1.0:
            raise ValueError("Metal roughness must be in [0, 1].")
        if self.scratches is not None and not 0.0 <= self.scratches <= 1.0:
            raise ValueError("Scratch amount must be in [0, 1].")
        if not isinstance(self.scratch_density, int) or not 1 <= self.scratch_density <= 128:
            raise ValueError("Scratch density must be an integer from 1 to 128.")
        if not math.isfinite(self.direction):
            raise ValueError("Scratch direction must be finite.")


def metal_names() -> tuple[str, ...]:
    """List the built-in conductor reflectance presets."""
    return tuple(_METAL_SRGB)


def _parameters(surface, seed):
    default_roughness, brushing, default_scratches = _FINISHES[surface.finish]
    if surface.reflectance is not None:
        reflectance = surface.reflectance
    else:
        reflectance = tuple(
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in _METAL_SRGB[surface.metal]
        )
    angle = math.radians(surface.direction % 360.0)
    # Integer frequencies preserve a seamless tile at arbitrary brush angles.
    perpendicular = (-math.sin(angle), math.cos(angle))
    wave = wp.vec2(*(round(108 * x) for x in perpendicular))
    state = 173 if seed is None else int(np.random.default_rng(seed).integers(1, 1 << 30))
    return [
        wp.vec3(*reflectance), default_roughness if surface.roughness is None else surface.roughness,
        brushing, default_scratches if surface.scratches is None else surface.scratches,
        surface.scratch_density, angle, wave, state,
    ]


@wp.func
def _scratch_field(
    uv: wp.vec2, seed: int, count: int, angle: float, amount: float, resolution: float, coarse: bool,
):
    grid = uv * float(count) - wp.vec2(0.5)
    x, y = int(wp.floor(grid[0])), int(wp.floor(grid[1]))
    cut = float(0.0)
    lip = float(0.0)
    wear = float(0.0)
    aa = 0.38 * float(count) / resolution
    for row in range(2):
        for col in range(2):
            cx, cy = x + col, y + row
            index = (cx % count + count) % count + count * ((cy % count + count) % count)
            state = wp.rand_init(seed, index)
            pick = wp.randf(state)
            density = amount if not coarse else 0.84 * amount
            if pick < density:
                center = wp.vec2(float(cx) + wp.randf(state, -0.11, 0.11), float(cy) + wp.randf(state, -0.11, 0.11))
                delta = grid - center
                if wp.dot(delta, delta) < 0.73:
                    # A skewed distribution gives many short scratches and
                    # comparatively few long, deep cuts.
                    length = 0.085 + 0.59 * wp.pow(wp.randf(state), 2.35)
                    width = 0.009 + 0.061 * wp.pow(wp.randf(state), 2.5)
                    depth = 0.19 + 0.81 * wp.pow(wp.randf(state), 1.85)
                    direction = angle + wp.randf(state, -1.55, 1.55)
                    orient = wp.vec2(wp.cos(direction), wp.sin(direction))
                    along = wp.dot(delta, orient) / length
                    taper = 1.0 - wp.smoothstep(wp.randf(state, 0.31, 0.73), 1.0, wp.abs(along))
                    curvature = wp.randf(state, -0.083, 0.083)
                    offset = wp.dot(delta, wp.vec2(-orient[1], orient[0])) - curvature * along * along
                    width *= 0.42 + 0.58 * taper
                    # A glancing scrape may leave two or three shorter
                    # parallel cuts within the same scuffed patch.
                    if not coarse and wp.randf(state) > 0.68:
                        spacing = 2.8 * width
                        cluster = 1.0 - wp.smoothstep(spacing, 2.2 * spacing, wp.abs(offset))
                        offset -= spacing * wp.floor(offset / spacing + 0.5)
                        taper *= cluster
                    distance = wp.abs(offset)
                    trough = 1.0 - wp.smoothstep(wp.max(0.0, 0.15 * width - aa), width + aa, distance)
                    ridge = 1.0 - wp.smoothstep(0.0, 0.58 * width + aa, wp.abs(distance - 0.98 * width))
                    coverage = taper * width / (width + aa)
                    cut = wp.max(cut, trough * depth * coverage)
                    lip = wp.max(lip, ridge * (0.25 + 0.75 * depth) * coverage)
                    wear = wp.max(wear, (0.69 * trough + 0.31 * ridge) * coverage * (0.34 + 0.66 * depth))
    return wp.vec3(cut, lip, wear)


@wp.func
def _metal_maps(
    uv: wp.vec2, resolution: float, reflectance: wp.vec3, roughness: float, brushing: float,
    scratches: float, density: int, direction: float, wave: wp.vec2, seed: int,
):
    brush = float(0.0)
    if brushing > 0.0:
        phase = wp.dot(wave, uv)
        phase += 0.76 * wp.pnoise(wp.rand_init(seed), 19.0 * uv, 19, 19)
        grooves = 0.5 + 0.5 * wp.cos(2.0 * wp.pi * phase)
        filament = 0.5 + 0.5 * wp.cos(4.0 * wp.pi * phase + 0.28 * wp.pnoise(wp.rand_init(seed + 2), 31.0 * uv, 31, 31))
        brush = (0.74 * grooves * grooves + 0.26 * filament * filament)
        brush *= wp.smoothstep(160.0, 680.0, resolution) * brushing
    marks = wp.vec3(0.0)
    if scratches > 0.0:
        coarse = _scratch_field(uv, seed + 42, wp.max(1, 2 * density // 3), direction, scratches, resolution, True)
        fine = _scratch_field(uv, seed + 79, 2 * density, direction, scratches, resolution, False)
        marks = (coarse + wp.cw_mul(fine, wp.vec3(0.42, 0.56, 0.71))) * wp.sqrt(scratches)
    polishing = wp.pnoise(wp.rand_init(seed + 5), 27.0 * uv, 27, 27) * brushing
    # The same grooves and raised lips also drive tangent-space normals.
    # Roughness and reflected light carry most of the appearance of a scratch.
    color = reflectance * wp.clamp(1.0 - 0.105 * marks[0] + 0.019 * marks[1] - 0.032 * brush, 0.0, 1.0)
    height = 0.55 - 0.16 * marks[0] + 0.028 * marks[1] - 0.019 * brush + 0.008 * polishing
    rough = wp.clamp(roughness + 0.36 * marks[2] + 0.10 * brush + 0.022 * polishing, 0.0, 1.0)
    return color, height, rough


@wp.kernel(enable_backward=False)
def _generate_metal(
    reflectance: wp.vec3, roughness: float, brushing: float, scratches: float, density: int,
    direction: float, wave: wp.vec2, seed: int, channel: int,
    image: wp.array2d(dtype=wp.vec3), relief: wp.array2d(dtype=float), packed: wp.array3d(dtype=wp.vec4ub),
):
    row, col = wp.tid()
    h = wp.max(image.shape[0], relief.shape[0])
    w = wp.max(image.shape[1], relief.shape[1])
    uv = wp.vec2((float(col) + 0.5) / float(w), (float(row) + 0.5) / float(h))
    color, height, rough = _metal_maps(
        uv, float(wp.min(h, w)), reflectance, roughness, brushing, scratches, density, direction, wave, seed
    )
    if relief.shape[0] > 0:
        relief[row, col] = height
    if packed.shape[0] > 0:
        packed[0, row, col] = wp.vec4ub(
            wp.uint8(255.0 * color[0] + 0.5), wp.uint8(255.0 * color[1] + 0.5),
            wp.uint8(255.0 * color[2] + 0.5), wp.uint8(255),
        )
        packed[2, row, col] = wp.vec4ub(wp.uint8(255), wp.uint8(255.0 * rough + 0.5), wp.uint8(255), wp.uint8(255))
    elif image.shape[0] > 0:
        result = color
        if channel == 2:
            result = wp.vec3(rough)
        elif channel == 3:
            result = wp.vec3(height)
        image[row, col] = result


@wp.kernel(enable_backward=False)
def _metal_normals(relief: wp.array2d(dtype=float), image: wp.array2d(dtype=wp.vec3), packed: wp.array3d(dtype=wp.vec4ub)):
    row, col = wp.tid()
    n = _normal_from_relief(relief, row, col, 0.0035)
    if packed.shape[0] > 0:
        packed[1, row, col] = wp.vec4ub(
            wp.uint8(255.0 * n[0] + 0.5), wp.uint8(255.0 * n[1] + 0.5),
            wp.uint8(255.0 * n[2] + 0.5), wp.uint8(255),
        )
    else:
        image[row, col] = n


def bake_metal(surface, shape, map_type, device, seed):
    channel = METAL_MAPS.index(map_type) if map_type in METAL_MAPS else -1
    if channel < 0:
        raise KeyError(map_type)
    if channel == 4:
        return wp.full(shape, wp.vec3(1.0), dtype=wp.vec3, device=device)
    image = wp.empty(shape, dtype=wp.vec3, device=device)
    device = image.device
    target = wp.empty((0, 0), dtype=wp.vec3, device=device) if channel == 1 else image
    relief = wp.empty(shape if channel == 1 else (0, 0), dtype=float, device=device)
    unused = wp.empty((0, 0, 0), dtype=wp.vec4ub, device=device)
    wp.launch(_generate_metal, dim=shape, inputs=[*_parameters(surface, seed), channel],
              outputs=[target, relief, unused], device=device)
    if channel == 1:
        wp.launch(_metal_normals, dim=shape, inputs=[relief], outputs=[image, unused], device=device)
    return image


def pack_metal(surface, shape, device, seed):
    packed = wp.empty((3, *shape), dtype=wp.vec4ub, device=device)
    device = packed.device
    relief = wp.empty(shape, dtype=float, device=device)
    unused = wp.empty((0, 0), dtype=wp.vec3, device=device)
    wp.launch(_generate_metal, dim=shape, inputs=[*_parameters(surface, seed), 0],
              outputs=[unused, relief, packed], device=device)
    wp.launch(_metal_normals, dim=shape, inputs=[relief], outputs=[unused, packed], device=device)
    return packed
