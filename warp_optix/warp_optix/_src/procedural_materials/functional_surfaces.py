# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-link wire, slotted steel grating, and inlaid low-pile carpet."""

import warp as wp

from .sampling import periodic_noise as _noise
from .sampling import wrap_index, wrap_uv


@wp.func
def _centered_cell(value: float) -> float:
    return value - wp.floor(value + 0.5)


@wp.func
def _chain_link(uv: wp.vec2, seed: int, resolution: float):
    # Each wire bends around its mate. The height offset alternates over and
    # under the crossing, while coverage includes the outside of both wires.
    vertical = 16.0 * uv[1] + 0.025
    vertical -= 0.070 * wp.sin(4.0 * wp.pi * vertical)
    x = 16.0 * uv[0] + vertical + 0.205
    y = 16.0 * uv[0] - vertical + 0.205
    ox = _noise(uv, seed + 11, 29, 27)
    oy = _noise(uv, seed + 13, 31, 27)
    dx = wp.abs(_centered_cell(x + 0.029 * wp.sin(2.0 * wp.pi * y) + 0.017 * ox))
    dy = wp.abs(_centered_cell(y - 0.036 * wp.sin(2.0 * wp.pi * x) + 0.017 * oy))
    radius = 0.0200 + 0.0028 * ox
    aa = wp.clamp(11.3 / resolution, 0.001, 0.0145)
    wire_x = 1.0 - wp.smoothstep(radius - aa, radius + aa, dx)
    wire_y = 1.0 - wp.smoothstep(radius - aa, radius + aa, dy)
    coverage = wp.max(wire_x, wire_y)
    roll_x = wp.sqrt(wp.max(0.0, 1.0 - (dx / (radius + aa)) ** 2.0))
    roll_y = wp.sqrt(wp.max(0.0, 1.0 - (dy / (radius + aa)) ** 2.0))
    twist = wp.cos(wp.pi * wp.floor(x + 0.5)) * wp.cos(wp.pi * wp.floor(y + 0.5))
    height_x = wire_x * (0.28 + 0.24 * roll_x + 0.040 * twist * wp.exp(-60.0 * dy * dy))
    height_y = wire_y * (0.28 + 0.24 * roll_y - 0.040 * twist * wp.exp(-60.0 * dx * dx))
    height = wp.max(height_x, height_y)
    abrasion = _noise(uv, seed + 7, 51, 47) + 0.42 * _noise(uv, seed + 3, 167, 172)
    chipped = wp.smoothstep(0.19, 0.43, abrasion + 0.065 * wp.max(roll_x, roll_y))
    grit = _noise(uv, seed + 5, 601, 576) * wp.smoothstep(310.0, 1230.0, resolution)
    coating = wp.vec3(0.142, 0.252, 0.146) + wp.vec3(0.16 * ox + 0.055 * grit)
    steel = wp.vec3(0.609, 0.585, 0.569) + wp.vec3(0.11 * oy + 0.074 * grit)
    corrosion = 0.47 * wp.smoothstep(0.14, 0.47, _noise(uv, seed + 37, 111, 117))
    steel = wp.lerp(steel, wp.vec3(0.47, 0.242, 0.131), corrosion)
    roughness = 0.64 - 0.22 * chipped + 0.09 * grit
    return wp.lerp(coating, steel, chipped), height, roughness, 1.0, 0.78 * chipped, coverage


@wp.func
def _slotted_walkway(uv: wp.vec2, seed: int, resolution: float):
    # Alternating rows of oblong stamped openings lie over steel braces
    # forming diamonds. The braces fill the cuts below the face of the plate.
    row = int(wp.floor(32.0 * uv[1]))
    slot = wp.vec2(_centered_cell(8.0 * uv[0] - 0.5 * float(row & 1)), 32.0 * uv[1] - float(row) - 0.5)
    slot = wp.cw_mul(slot, wp.vec2(0.125, 0.03125))
    rounded = wp.vec2(wp.abs(slot[0]) - 0.045, wp.abs(slot[1]) - 0.0096)
    radius = 0.0051
    rounded += wp.vec2(radius)
    outside = wp.vec2(wp.max(rounded[0], 0.0), wp.max(rounded[1], 0.0))
    hole = wp.length(outside) + wp.min(wp.max(rounded[0], rounded[1]), 0.0) - radius
    aa = 0.61 / resolution
    face = wp.smoothstep(-aa, aa, hole)
    brace_dist = wp.min(
        wp.abs(_centered_cell(2.0 * (uv[0] + uv[1]) + 0.02)),
        wp.abs(_centered_cell(2.0 * (uv[0] - uv[1]) - 0.02)),
    )
    brace = 1.0 - wp.smoothstep(0.055 - 2.0 * aa, 0.055 + 2.0 * aa, brace_dist)
    coverage = wp.max(face, brace)
    edge = wp.smoothstep(-0.0005, 0.0035 + aa, hole)

    grime = _noise(uv, seed + 8, 17, 15) + 0.59 * _noise(uv, seed + 13, 64, 74)
    drag = _noise(uv, seed + 7, 88, 310) * wp.smoothstep(225.0, 920.0, resolution)
    pepper = _noise(uv, seed + 29, 461, 489) * wp.smoothstep(250.0, 1100.0, resolution)
    scour = wp.smoothstep(0.44, 0.69, grime + 0.65 * drag)
    cut = 1.0 - wp.smoothstep(0.0009, 0.0032, wp.abs(hole))
    worn = wp.clamp(0.57 * cut * wp.smoothstep(0.045, 0.45, grime) + 0.56 * scour, 0.0, 1.0)
    paint = wp.vec3(0.197, 0.188, 0.222) + wp.vec3(0.037 * grime + 0.031 * drag + 0.052 * pepper)
    bare = wp.vec3(0.300, 0.309, 0.322)
    color = wp.lerp(paint * (0.83 + 0.17 * face), bare, worn)
    height = wp.lerp(0.24 * brace, 0.76 + 0.225 * edge + 0.008 * drag + 0.014 * pepper, face)
    roughness = 0.42 - 0.19 * worn + 0.085 * grime + 0.11 * (1.0 - edge)
    return color, height, roughness, 1.0, wp.clamp((worn - 0.18) * 1.45, 0.0, 1.0), coverage


@wp.func
def _carpet_tufts(uv: wp.vec2, seed: int):
    # Tufts lean in the same direction, but their tips cross the nominal
    # weave cells. Take the raised envelope across neighboring tufts.
    density = 270
    warp = wp.vec2(_noise(uv, seed + 15, 44, 53), _noise(uv, seed + 18, 57, 49))
    knot = float(density) * (uv + 0.0037 * warp)
    base = wp.vec2(wp.floor(knot[0]), wp.floor(knot[1]))
    mound = float(0.0)
    for row in range(-1, 2):
        for col in range(-1, 2):
            cell = base + wp.vec2(float(col), float(row))
            cx = wrap_index(int(cell[0]), density)
            cy = wrap_index(int(cell[1]), density)
            rng = wp.rand_init(seed + 41, cx + density * cy)
            center = wp.vec2(wp.randf(rng, 0.13, 0.87), wp.randf(rng, 0.15, 0.85))
            delta = knot - cell - center
            lean = wp.randf(rng, 0.17, 0.77)
            tip = wp.vec2(0.78 * (delta[0] + lean * delta[1]), 1.65 * (delta[1] - lean * delta[0]))
            footprint = wp.max(0.0, 1.0 - 2.2 * wp.dot(tip, tip))
            mound = wp.max(mound, footprint * footprint * wp.randf(rng, 0.48, 1.0))
    return mound


@wp.func
def _cut_pile_carpet(uv: wp.vec2, seed: int, resolution: float):
    # Both yarn colors pass through the same tuft geometry at the seams.
    mound = _carpet_tufts(uv, seed)
    grain = _noise(uv, seed + 4, 213, 231)
    fibers = _noise(uv, seed + 7, 718, 831) * wp.smoothstep(520.0, 1830.0, resolution)
    micro = (mound - 0.20) * wp.smoothstep(210.0, 1170.0, resolution)
    nap = _noise(uv, seed + 21, 21, 29)

    square = wp.vec2(_centered_cell(8.0 * uv[0] + 0.025), _centered_cell(8.0 * uv[1] - 0.012))
    border = wp.max(wp.abs(square[0]), wp.abs(square[1])) + 0.094 * grain + 0.035 * nap
    aa = 0.017 + 3.5 / resolution
    inlay = 1.0 - wp.smoothstep(0.236 - aa, 0.236 + aa, border)
    wheat = wp.vec3(0.739, 0.687, 0.545)
    oat = wp.vec3(0.853, 0.813, 0.722)
    tone = 0.052 * nap + 0.082 * grain + 0.18 * micro + 0.104 * fibers
    color = wp.lerp(wheat, oat, inlay) + wp.vec3(tone)
    height = 0.48 + 0.19 * micro + 0.075 * grain + 0.061 * fibers + 0.029 * inlay
    occlusion = 0.61 + 0.43 * micro + 0.18 * grain
    return color, height, 0.875 + 0.047 * grain - 0.046 * micro, occlusion, 0.0, 1.0


@wp.func
def functional_surface(uv: wp.vec2, seed: int, resolution: float, style: int):
    uv = wrap_uv(uv)
    if style == 0:
        return _chain_link(uv, seed, resolution)
    if style == 1:
        return _slotted_walkway(uv, seed, resolution)
    return _cut_pile_carpet(uv, seed, resolution)
