# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit localized periodic basis coefficients from extracted ambientCG maps."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from heapq import heappop, heappush
from pathlib import Path
from struct import pack
from zlib import compress

import numpy as np
from PIL import Image
from warp_optix._src.texture_coefficients import LEVELS, decode, quantize_bands

MAPS = ("color", "normal", "roughness", "displacement", "occlusion", "metalness", "opacity")
REQUIRED = MAPS[:4]
SHARD_PAYLOAD = 80_000
STARTER_IDS = {
    "Bricks113A",
    "Concrete049",
    "Fabric004",
    "Leather037",
    "Marble012",
    "Metal064",
    "PavingStones153",
    "RoofingTiles013A",
    "Rubber001",
    "Tiles107",
    "Wood099",
    "WoodFloor071",
}


def _lift_axis(values, axis):
    samples = np.moveaxis(values, axis, 0)
    even, odd = samples[::2], samples[1::2]
    high = odd - ((even + np.roll(even, -1, axis=0)) >> 1)
    low = even + ((np.roll(high, 1, axis=0) + high + 2) >> 2)
    samples[:] = np.concatenate((low, high), axis=0)


def _resize(image, size):
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    if image.mode == "F":
        return Image.fromarray(np.rint(np.clip(np.asarray(resized) / 257.0, 0, 255)).astype(np.uint8))
    return resized


def _encode(grid, kind, residual):
    size = grid.width
    detail = np.atleast_1d(np.sqrt(np.mean(residual**2, axis=(0, 1))) * (1.45 / 255.0))
    values = np.asarray(grid, dtype=np.int16)
    if kind == "color":
        # Reversible luminance/chroma transform. Quantize chroma more
        # coarsely to retain contrast and edges with fewer coefficients.
        r, g, b = np.moveaxis(values, 2, 0)
        co = r - b
        t = b + (co >> 1)
        cg = g - t
        values = np.stack((t + (cg >> 1), co, cg), axis=-1)
    elif kind == "normal":
        values = np.ascontiguousarray(values[..., :2])
        detail[2] = 0.0
    else:
        values = values[..., None]
        detail = np.repeat(detail, 3)
    if kind not in ("color", "normal", "displacement"):
        detail[:] = 0.0
    for level in range(LEVELS):
        limit = size >> level
        block = values[:limit, :limit]
        _lift_axis(block, 1)
        _lift_axis(block, 0)
    finest = values[size // 2 :, :, 0]
    max_strength = (6 if kind in ("color", "normal", "displacement") else 4) * max(1, size // 256)
    # Low-contrast maps show quantization sooner than noisy surfaces.
    strength = min(max_strength, 1 + round(float(np.sqrt(np.mean(finest.astype(np.float32) ** 2)))))
    for channel in range(values.shape[-1]):
        quantize_bands(values[..., channel], strength + (2 if kind == "color" and channel else 0))
    payload = pack("<HBB3f", size, values.shape[-1], strength, *detail) + values.astype("<i2").tobytes()
    return compress(payload, level=6)


def fit_map(path, kind):
    candidates = []
    with Image.open(path) as image:
        if kind in ("color", "normal"):
            image = image.convert("RGB")
        elif image.mode.startswith("I"):
            image = image.convert("F")
        else:
            image = image.convert("L")
        resolution = 768
        source = np.asarray(_resize(image, resolution), dtype=np.float32)
        sizes = (64, 128, 256, 512)
        if kind in ("color", "normal") and path.stem.startswith(("Fabric004", "Tiles107", "WoodFloor071")):
            sizes = (*sizes, 1024)
        importance = {"color": 3.0, "normal": 2.2, "metalness": 1.2, "opacity": 1.4}.get(kind, 0.7)
        if path.stem.startswith(("Road", "Tiles", "Bricks", "PavingStones", "WoodFloor")) and kind in (
            "color",
            "normal",
        ):
            # Thin paint and grout edges are visible even at low contrast.
            importance *= 2
        for size in sizes:
            grid = _resize(image, size)
            residual = source - np.asarray(
                grid.resize((resolution, resolution), Image.Resampling.BICUBIC), dtype=np.float32
            )
            encoded = _encode(grid, kind, residual)
            restored, _ = decode(encoded, kind)
            pixels = np.rint(restored * 255.0).astype(np.uint8)
            if kind not in ("color", "normal"):
                pixels = pixels[..., 0]
            reconstructed = Image.fromarray(pixels).resize((resolution, resolution), Image.Resampling.BICUBIC)
            residual = source - np.asarray(reconstructed, dtype=np.float32)
            signal = residual[..., :2] if kind == "normal" else residual
            error = np.mean(signal**2)
            edge_error = sum(np.mean((signal - np.roll(signal, 1, axis=axis)) ** 2) for axis in (0, 1))
            candidates.append((float(importance * (error + 2.0 * edge_error)), encoded))
    return candidates


def allocate(candidates, budget):
    """Choose per-map resolutions by how much each upgrade improves the fit per byte."""
    choices = dict.fromkeys(candidates, 0)
    used = sum(len(options[0][1]) for options in candidates.values())
    if used > budget:
        raise ValueError(f"The lowest resolution models need {used} bytes, above the {budget} byte budget.")
    upgrades = []

    def offer(key, current):
        options = candidates[key]
        if current + 1 < len(options):
            old_error, old = options[current]
            error, new = options[current + 1]
            gain = old_error - error
            cost = len(new) - len(old)
            if gain > 0:
                heappush(upgrades, (-gain / max(cost, 1), cost, key, current + 1))

    for key in choices:
        offer(key, 0)
    while upgrades:
        _, cost, key, index = heappop(upgrades)
        if used + cost <= budget:
            used += cost
            choices[key] = index
            offer(key, index)
    return {key: candidates[key][index][1] for key, index in choices.items()}


def write_coefficients(assets, output):
    """Write small NumPy archives, including a separate starter group for wheels."""
    output.mkdir(parents=True, exist_ok=True)
    index = {asset: {} for asset in assets}

    def flush(prefix, shard, payload):
        np.savez_compressed(output / f"{prefix}-{shard:03}.npz", payload=np.frombuffer(payload, dtype=np.uint8))

    for is_starter, prefix in ((True, "starter"), (False, "full")):
        shard = 0
        payload = bytearray()

        for asset, maps in assets.items():
            if (asset in STARTER_IDS) != is_starter:
                continue
            for kind, encoded in maps.items():
                spans = []
                offset = 0
                while offset < len(encoded):
                    if SHARD_PAYLOAD - len(payload) < 1_024:
                        flush(prefix, shard, payload)
                        shard += 1
                        payload = bytearray()
                    count = min(len(encoded) - offset, SHARD_PAYLOAD - len(payload))
                    spans.append((f"{prefix}-{shard:03}.npz", len(payload), count))
                    payload.extend(encoded[offset : offset + count])
                    offset += count
                index[asset][kind] = spans
        if payload:
            flush(prefix, shard, payload)
    starter = {asset: maps for asset, maps in index.items() if asset in STARTER_IDS}
    for name, entries in (("index.json", index), ("starter-index.json", starter)):
        (output / name).write_text(json.dumps(entries, separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", type=Path, help="Directory containing extracted <ID>_<map> files")
    parser.add_argument("ids", type=Path, help="Text file with one material ID per line")
    parser.add_argument("output", type=Path, help="Output NumPy archive directory")
    parser.add_argument("--budget-mib", type=float, default=9.6, help="Budget for compressed coefficient bytes")
    parser.add_argument("--high-resolution-maps", type=Path, help="Prefer maps from this directory when available")
    args = parser.parse_args()
    ids = args.ids.read_text().splitlines()
    args.output.mkdir(parents=True, exist_ok=True)

    def fit_material(asset):
        result = {}
        for kind in MAPS:
            pattern = f"{asset}_{kind}.*"
            paths = sorted(args.high_resolution_maps.glob(pattern)) if args.high_resolution_maps else []
            paths = paths or sorted(args.maps.glob(pattern))
            if paths:
                result[kind] = fit_map(paths[0], kind)
            elif kind in REQUIRED:
                raise FileNotFoundError(f"No {kind} map for {asset}")
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        options = dict(zip(ids, pool.map(fit_material, ids), strict=True))
    fitted = allocate(
        {(asset, kind): models for asset, maps in options.items() for kind, models in maps.items()},
        int(args.budget_mib * 1024**2),
    )
    assets = {asset: {kind: fitted[asset, kind] for kind in maps} for asset, maps in options.items()}
    write_coefficients(assets, args.output)


if __name__ == "__main__":
    main()
