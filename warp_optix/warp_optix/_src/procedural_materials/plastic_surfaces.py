# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opaque molded and sheet polymers, with tileable surface finishes."""

from dataclasses import dataclass

import warp as wp

from .configurable_metals import _scratch_field

# Perceptual roughness, normal strength, and default sRGB mold color.
_FINISHES = {
    "gloss": (0.165, 0.00014, (0.77, 0.085, 0.06)),
    "satin": (0.40, 0.0016, (0.93, 0.918, 0.914)),
    "pebbled": (0.53, 0.0025, (0.23, 0.235, 0.24)),
    "grained": (0.61, 0.0030, (0.245, 0.22, 0.195)),
    "scuffed": (0.45, 0.00165, (0.169, 0.156, 0.150)),
    "recycled": (0.52, 0.00095, (0.67, 0.65, 0.57)),
    "vinyl": (0.55, 0.0013, (0.56, 0.61, 0.59)),
    "rubber": (0.78, 0.0017, (0.205, 0.22, 0.23)),
}


@dataclass(frozen=True, slots=True)
class PlasticSurface:
    """Configure an opaque polymer finish with a custom sRGB color and roughness.

    Gloss and satin represent smooth molded plastic. Pebbled and grained
    represent mold textures. Scuffed, recycled, vinyl, and rubber model
    different types of wear, inclusions, and surface microstructure.
    """

    finish: str = "gloss"
    color: tuple[float, float, float] | None = None
    roughness: float | None = None

    def __post_init__(self):
        if self.finish not in _FINISHES:
            raise ValueError(f"Unknown plastic finish {self.finish!r}. Choose from {', '.join(_FINISHES)}.")
        if self.color is not None and (len(self.color) != 3 or not all(0.0 <= c <= 1.0 for c in self.color)):
            raise ValueError("Plastic color must contain three sRGB values in [0, 1].")
        if self.roughness is not None and not 0.0 <= self.roughness <= 1.0:
            raise ValueError("Plastic roughness must be in [0, 1].")


PLASTIC_PRESETS = {
    "PlasticGlossABS": PlasticSurface("gloss"),
    "Plastic013A": PlasticSurface("satin"),
    "PlasticPebbledTrim": PlasticSurface("pebbled"),
    "PlasticGrainedDash": PlasticSurface("grained"),
    "Plastic012B": PlasticSurface("scuffed"),
    "PlasticRecycled": PlasticSurface("recycled"),
    "PlasticCommercialVinyl": PlasticSurface("vinyl"),
    "PlasticMatteRubber": PlasticSurface("rubber"),
    "Plastic017A": PlasticSurface("gloss", (0.314, 0.733, 0.290), 0.235),
}


def plastic_configuration(surface):
    roughness, strength, color = _FINISHES[surface.finish]
    style = -40 - tuple(_FINISHES).index(surface.finish)
    return (
        style,
        surface.color if surface.color is not None else color,
        (surface.roughness if surface.roughness is not None else roughness),
        strength,
    )


@wp.func
def _plastic_noise(seed: int, uv: wp.vec2, nx: int, ny: int, resolution: float):
    frequency = float(wp.max(nx, ny))
    band = wp.smoothstep(1.8 * frequency, 4.5 * frequency, resolution)
    return band * wp.pnoise(wp.rand_init(seed), wp.cw_mul(uv, wp.vec2(float(nx), float(ny))), nx, ny)


@wp.func
def _plastic_cells(uv: wp.vec2, count: int, seed: int):
    position = uv * float(count)
    x, y = int(wp.floor(position[0])), int(wp.floor(position[1]))
    nearest = float(9.0)
    second = float(9.0)
    tag = float(0.0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            cx, cy = x + i, y + j
            key = (cx % count + count) % count + count * ((cy % count + count) % count)
            state = wp.rand_init(seed, key)
            center = wp.vec2(float(cx) + wp.randf(state, 0.05, 0.95), float(cy) + wp.randf(state, 0.05, 0.95))
            delta = position - center
            distance = wp.dot(delta, delta)
            if distance < nearest:
                second = nearest
                nearest = distance
                tag = wp.randf(state)
            elif distance < second:
                second = distance
    return wp.vec3(wp.sqrt(nearest), wp.sqrt(second) - wp.sqrt(nearest), tag)


@wp.func
def plastic_surface(uv: wp.vec2, seed: int, resolution: float, style: int, color: wp.vec3, roughness: float):
    flow = _plastic_noise(seed + 1, uv, 9, 4, resolution)
    fine = _plastic_noise(seed + 2, uv, 174, 181, resolution)
    height = 0.50 + 0.10 * flow + 0.034 * fine
    tint = 1.0 + 0.016 * flow
    rough = roughness + 0.025 * flow + 0.012 * fine
    if style == 1:
        # Mold polish leaves broad satin streaks with subtle fine tooling.
        tool = _plastic_noise(seed + 3, uv, 88, 11, resolution)
        height += 0.047 * tool + 0.032 * fine
        rough += 0.045 * tool
    elif style == 2 or style == 3:
        count = 69 if style == 2 else 36
        warp = wp.vec2(
            _plastic_noise(seed + 3, uv, 15, 17, resolution), _plastic_noise(seed + 4, uv, 19, 15, resolution)
        )
        cell = _plastic_cells(uv + (0.76 / float(count)) * warp, count, seed + 5)
        grain = 1.0 - wp.smoothstep(0.13, 0.85, cell[0])
        if style == 3:
            # Molded leather grain has irregular valleys between plateaus.
            grain = wp.smoothstep(0.023, 0.33, cell[1]) * (0.71 + 0.29 * grain)
        grain *= wp.smoothstep(2.0 * float(count), 5.0 * float(count), resolution)
        height += (0.15 if style == 2 else 0.24) * grain + 0.042 * fine
        tint += 0.071 * (grain - 0.3) + 0.033 * fine
        rough += 0.085 * (0.42 - grain) + 0.036 * fine
    elif style == 4:
        mark = _scratch_field(uv, seed + 4, 11, 0.7, 0.70, resolution, True)
        scuffs = _scratch_field(uv, seed + 6, 25, -0.7, 0.53, resolution, False)
        wear = mark[2] + 0.6 * scuffs[2]
        height -= 0.18 * (mark[0] + 0.45 * scuffs[0])
        tint += 0.47 * wear
        rough += 0.35 * wear
    elif style == 5 or style == 6:
        # Pigment chips are embedded in one continuous molded or rolled sheet.
        count = 72 if style == 5 else 105
        chip = _plastic_cells(uv, count, seed + 7)
        width = 0.16 + 0.28 * chip[2]
        edge = 0.55 * float(count) / resolution
        flake = 1.0 - wp.smoothstep(wp.max(0.0, width - edge), width + edge, chip[0] + 0.10 * fine)
        flake *= wp.smoothstep(1.8 * float(count), 4.5 * float(count), resolution)
        if style == 5:
            pigment = wp.vec3(0.34, 0.41, 0.54)
            if chip[2] > 0.60:
                pigment = wp.vec3(0.78, 0.49, 0.31)
            mix = flake * wp.smoothstep(0.37, 0.68, chip[2])
            color = wp.cw_mul(color, wp.lerp(wp.vec3(1.0), pigment, mix))
        else:
            tint += 0.36 * (chip[2] - 0.53) * flake
        height += 0.042 * flake + 0.037 * fine
        rough += 0.11 * flake
    elif style == 7:
        tooth = _plastic_noise(seed + 3, uv, 103, 96, resolution)
        cavity = wp.smoothstep(0.27, 0.62, tooth)
        height += 0.105 * tooth - 0.17 * cavity + 0.06 * fine
        tint += 0.12 * tooth - 0.16 * cavity
        rough += 0.09 * tooth
    return color * tint, height, rough, 1.0, 0.0, 1.0
