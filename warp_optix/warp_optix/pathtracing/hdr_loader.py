# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pure Python Radiance RGBE HDR loader.
# Aligned with the HDR loader behavior used in the reference sample.
# Preserves RGB float conversion and row order (first scanline -> row height-1).
# No OpenCV or imageio dependency for HDR.

"""Load Radiance RGBE (.hdr) files with reference-compatible behavior."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MINELEN = 8
MAXELEN = 0x7FFF


def _convert_component(expo: int, val: int) -> float:
    """RGBE to float: val/256 * 2^expo."""
    v = val / 256.0
    d = 2.0**expo
    return v * d


class _Reader:
    """Byte reader over buffer with binary-reader semantics."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def read_byte(self) -> int:
        if self.eof():
            return 0
        b = self.data[self.pos]
        self.pos += 1
        return b

    def seek_back(self, n: int = 1) -> None:
        self.pos = max(0, self.pos - n)


def _old_decrunch(reader: _Reader, width: int, scanline: bytearray, offset: int = 0) -> bool:
    """Legacy non-RLE format."""
    rshift = 0
    j = 0

    while j < width:
        if reader.eof():
            return False
        r = reader.read_byte()
        g = reader.read_byte()
        b = reader.read_byte()
        e = reader.read_byte()
        # Do not fail when EOF after reading last pixel.

        idx = offset + j * 4
        scanline[idx + 0] = r
        scanline[idx + 1] = g
        scanline[idx + 2] = b
        scanline[idx + 3] = e

        if r == 1 and g == 1 and b == 1:
            rpt = e << rshift
            prev = j
            for _ in range(rpt - 1):
                j += 1
                if j >= width:
                    break
                pidx = offset + prev * 4
                jidx = offset + j * 4
                scanline[jidx + 0] = scanline[pidx + 0]
                scanline[jidx + 1] = scanline[pidx + 1]
                scanline[jidx + 2] = scanline[pidx + 2]
                scanline[jidx + 3] = scanline[pidx + 3]
            rshift += 8
        else:
            rshift = 0
        j += 1

    return True


def _decrunch(reader: _Reader, width: int, scanline: bytearray) -> bool:
    """RLE or legacy format."""
    if width < MINELEN or width > MAXELEN:
        return _old_decrunch(reader, width, scanline)

    i = reader.read_byte()
    if i != 2:
        reader.seek_back(1)
        return _old_decrunch(reader, width, scanline)

    g = reader.read_byte()
    b = reader.read_byte()
    i = reader.read_byte()

    scanline[0 * 4 + 0] = 2
    scanline[0 * 4 + 1] = g
    scanline[0 * 4 + 2] = b
    scanline[0 * 4 + 3] = i

    if g != 2 or (b & 128) != 0:
        # Fallback: first pixel is (2, g, b, i), decode rest with old format
        scanline[0 * 4 + 0] = 2
        scanline[0 * 4 + 3] = i
        return _old_decrunch(reader, width - 1, scanline, offset=4) if width > 1 else True

    # RLE: read each component (R, G, B, E) as separate stream
    for c in range(4):
        j = 0
        while j < width:
            if reader.eof():
                return False
            code = reader.read_byte()
            if code > 128:
                code &= 127
                if code == 0 or j + code > width or reader.eof():
                    return False
                val = reader.read_byte()
                for _ in range(code):
                    scanline[j * 4 + c] = val
                    j += 1
            else:
                if code == 0 or j + code > width:
                    return False
                for _ in range(code):
                    if reader.eof():
                        return False
                    scanline[j * 4 + c] = reader.read_byte()
                    j += 1

    return True


# Upper bounds applied before allocating decode buffers, so a crafted header
# cannot request an arbitrarily large allocation. They still admit 16K x 16K
# environment maps.
MAX_HDR_DIMENSION = 32768
MAX_HDR_PIXELS = 1 << 28


def load_hdr(path: str | Path) -> tuple[np.ndarray, int, int]:
    """
    Load Radiance RGBE HDR file with reference-compatible row ordering.

    Row order: first scanline in file -> row h-1 (bottom),
    last scanline -> row 0 (top). So row 0 = bottom of image.

    Returns:
        (data, width, height) where data is (height, width, 3) float32 RGB.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDR file not found: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    # Header check: accept #?RADIANCE (10 chars) or #?RGBE (6 chars)
    if len(raw) < 10:
        raise ValueError("Invalid HDR: file too short")
    header = raw[:10].decode("ascii", errors="replace")
    if header[:10] != "#?RADIANCE" and header[:6] != "#?RGBE":
        raise ValueError("Invalid HDR: missing #?RADIANCE or #?RGBE header")

    # Find resolution line: -Y height +X width (Radiance format)
    m = re.search(rb"-Y[ \t]+(\d+)[ \t]+\+X[ \t]+(\d+)[ \t]*\r?\n", raw)
    if not m:
        raise ValueError("Invalid HDR: cannot parse resolution line (-Y H +X W)")
    height = int(m.group(1))
    width = int(m.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid HDR: dimensions must be positive")
    if width > MAX_HDR_DIMENSION or height > MAX_HDR_DIMENSION:
        raise ValueError(
            f"Invalid HDR: {width}x{height} exceeds the {MAX_HDR_DIMENSION} pixel "
            "dimension limit"
        )
    if width * height > MAX_HDR_PIXELS:
        raise ValueError(
            f"Invalid HDR: {width}x{height} exceeds the {MAX_HDR_PIXELS} pixel limit"
        )

    # Pixel data starts after the resolution line
    data = raw[m.end() :]
    reader = _Reader(data)

    rgbe = np.zeros((height, width, 4), dtype=np.uint8)
    scanline = bytearray(width * 4)

    # First scanline in the file maps to row h-1.
    for y in range(height - 1, -1, -1):
        scanline[:] = b"\x00" * (width * 4)
        ok = _decrunch(reader, width, scanline)
        if not ok:
            raise ValueError("Invalid HDR: truncated or malformed pixel data")
        rgbe[y, :, :] = np.frombuffer(scanline, dtype=np.uint8, count=width * 4).reshape(width, 4)

    expo = rgbe[:, :, 3].astype(np.int16) - 128
    scale = np.exp2(expo.astype(np.float32)) * (1.0 / 256.0)
    cols = rgbe[:, :, :3].astype(np.float32) * scale[:, :, None]
    return cols, width, height
