# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode localized, periodic basis coefficients used by procedural textures."""

import json
from functools import lru_cache
from struct import unpack_from
from zlib import decompress

import numpy as np

BUILTIN_SURFACES = {
    "CharcoalBrick": ((0.26, 0.25, 0.24), 0.22, 0.75, 0.5, 9, 22),
    "WhiteSubwayTile": ((0.92, 0.91, 0.87), 0.036, 0.27, 0.5, 8, 16),
    "SageCeramicTile": ((0.48, 0.56, 0.47), 0.16, 0.36, 0.0, 10, 10),
    "BlueCeramicTile": ((0.26, 0.42, 0.52), 0.16, 0.36, 0.0, 10, 10),
    "SandstoneFloorTile": ((0.72, 0.60, 0.46), 0.24, 0.75, 0.0, 8, 8),
    "LimewashWall": ((0.81, 0.78, 0.70), 0.23, 0.84, 0.0),
    "GraphiteConcrete": ((0.29, 0.28, 0.27), 0.24, 0.79, 0.21),
    "BoardFormedConcrete": ((0.58, 0.55, 0.50), 0.20, 0.81, 0.77),
}

SURFACE_PRESETS = {
    **BUILTIN_SURFACES,
    "Asphalt031": ((0.41, 0.4, 0.4), 0.26, 0.86, -0.95),
    "Asphalt033": ((0.31, 0.3, 0.27), 0.14, 0.86, -0.38),
    "Concrete031": ((0.43, 0.42, 0.4), 0.29, 0.81, 0),
    "Concrete034": ((0.69, 0.69, 0.69), 0.16, 0.75, 0),
    "Concrete042A": ((0.38, 0.37, 0.36), 0.25, 0.82, 0.13),
    "Concrete046": ((0.69, 0.69, 0.58), 0.21, 0.76, 0.35),
    "Concrete047A": ((0.45, 0.43, 0.36), 0.25, 0.8, 0.18),
    "Concrete048": ((0.55, 0.52, 0.46), 0.19, 0.8, 0.62),
    "PaintedPlaster017": ((0.72, 0.72, 0.7), 0.08, 0.69, 0),
    "Plaster001": ((0.8, 0.77, 0.75), 0.15, 0.86, 0),
    "Plaster007": ((0.69, 0.68, 0.6), 0.26, 0.86, 0),
    "Tiles107": ((0.93, 0.93, 0.93), 0.025, 0.28, 0, 8, 8),
    "Tiles138": ((0.18, 0.17, 0.16), 0.09, 0.41, 0, 12, 12),
    "Tiles139": ((0.62, 0.54, 0.44), 0.19, 0.68, 0, 10, 12),
    "Tiles140": ((0.29, 0.26, 0.23), 0.23, 0.56, 0, 10, 12),
    "Tiles141": ((0.73, 0.69, 0.62), 0.19, 0.71, 0, 10, 12),
    "Tiles142": ((0.76, 0.64, 0.51), 0.19, 0.7, 0, 8, 12),
    "Tiles143": ((0.74, 0.62, 0.46), 0.21, 0.68, 0, 8, 10),
    "Tiles144": ((0.54, 0.38, 0.23), 0.23, 0.68, 0, 8, 10),
    "Bricks104": ((0.55, 0.24, 0.16), 0.18, 0.66, 0.5, 9, 16),
    "Bricks105": ((0.66, 0.53, 0.35), 0.21, 0.69, 0.5, 10, 16),
}
ANALYTIC_MATERIALS = (
    "Wood097", "Wood098", "Wood099", "PavingStones153", "Concrete049", "WoodFloor071", "Bricks101", *SURFACE_PRESETS
)

# Each entry identifies a surface generator, the reference maps it provides,
# and the strength used to convert height gradients to tangent normals.
DIRECT_REFERENCES = {
    "Terrazzo004": (-1, (), 0.00065),
    "SheetMetal002": (-2, ("metalness", "opacity"), 0.0014),
    "TactilePaving004": (-3, ("occlusion",), 0.0028),
    "MetalPlates001": (-4, ("metalness",), 0.0013),
    "AcousticFoam001": (-5, ("occlusion", "metalness"), 0.0036),
    "Wicker010A": (-6, ("occlusion", "opacity"), 0.00964),
    "Fabric013": (-7, ("opacity",), 0.011),
    "Fabric084": (-8, ("occlusion",), 0.0008),
    "Ground111": (-9, ("occlusion",), 0.0012),
    "Bark014": (-10, ("occlusion",), 0.0047),
    "Bark016": (-11, ("occlusion",), 0.0047),
    "Cardboard001": (-12, (), 0.0011),
    "Ground031": (-13, ("occlusion",), 0.0026),
    "Ground094C": (-14, ("occlusion",), 0.006),
    "Concrete008": (-15, ("occlusion",), 0.00045),
    "Concrete015": (-16, (), 0.0009),
    "Concrete036": (-17, (), 0.00065),
    "RoofingTiles008": (-18, ("occlusion",), 0.024),
    "RoofingTiles014A": (-19, ("occlusion",), 0.016),
    "Tiles071": (-20, ("occlusion",), 0.00418),
    "Wood096": (-21, (), 0.0035),
    "Fabric004": (-22, ("metalness",), 0.006),
    "Planks037A": (-23, ("occlusion", "metalness"), 0.0030),
    "Rope003": (-24, ("metalness",), 0.01139),
    "Rust005": (-25, ("metalness",), 0.00472),
    "Concrete041D": (-26, ("occlusion", "metalness"), 0.00601),
    "Fence001": (-27, ("metalness", "opacity"), 0.002),
    "MetalWalkway010": (-28, ("metalness", "opacity"), 0.005),
    "Carpet009": (-29, ("occlusion",), 0.0035),
    "Foil001": (-30, ("occlusion", "metalness"), 0.0095),
    "Moss001": (-31, ("occlusion",), 0.0037),
    "Grass005": (-32, ("occlusion",), 0.0012),
    "Leather035A": (-33, ("occlusion",), 0.0016),
    "Onyx015": (-34, (), 0.00035),
    "Marble012": (-35, (), 0.00055),
    "Ice003": (-36, (), 0.0012),
    "Chipboard002": (-37, ("occlusion",), 0.00078),
    "Sponge001": (-38, ("occlusion",), 0.01043),
    "Paper001": (-39, (), 0.002367),
}

LEVELS = 3


def _restore_axis(values, axis):
    slices = np.moveaxis(values, axis, 0)
    low, high = np.split(slices, 2, axis=0)
    even = low - ((np.roll(high, 1, axis=0) + high + 2) >> 2)
    odd = high + ((even + np.roll(even, -1, axis=0)) >> 1)
    slices[::2] = even
    slices[1::2] = odd


def quantize_bands(values, strength, *, restore=False):
    """Quantize localized high-frequency coefficients, leaving the coarse field intact."""
    size = values.shape[0]
    for level in range(LEVELS):
        half = size // 2
        step = max(1, strength - level)
        for band in (values[:half, half:size], values[half:size, :size]):
            if restore:
                band *= step
            else:
                band[:] = np.rint(band / step)
        size = half


def decode(encoded, kind):
    packed = decompress(encoded)
    size, channels, strength, *detail = unpack_from("<HBB3f", packed)
    values = np.frombuffer(packed, dtype="<i2", offset=16).copy().reshape(size, size, channels)
    for channel in range(channels):
        quantize_bands(values[..., channel], strength + (2 if kind == "color" and channel else 0), restore=True)
    for level in reversed(range(LEVELS)):
        limit = size >> level
        block = values[:limit, :limit]
        _restore_axis(block, 0)
        _restore_axis(block, 1)

    if kind == "color":
        y, co, cg = np.moveaxis(values, 2, 0)
        t = y - (cg >> 1)
        b = t - (co >> 1)
        values = np.stack((b + co, t + cg, b), axis=-1)
    elif channels == 2:
        values = np.pad(values, ((0, 0), (0, 0), (0, 1)))
    else:
        values = np.repeat(values, 3, axis=2)
    return (np.clip(values, 0, 255).astype(np.float32) / 255.0), detail


def read_index(directory):
    return json.loads((directory / "index.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _archive_payload(path):
    with np.load(path, allow_pickle=False) as archive:
        return archive["payload"]


def read_record(directory, spans):
    return b"".join(
        _archive_payload(directory / archive)[offset : offset + length].tobytes() for archive, offset, length in spans
    )
