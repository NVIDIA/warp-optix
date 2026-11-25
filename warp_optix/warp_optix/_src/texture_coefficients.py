# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode localized, periodic basis coefficients used by procedural textures."""

import json
from functools import lru_cache
from struct import unpack_from
from zlib import decompress

import numpy as np

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
