# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit small, seamless high-frequency exemplars from aligned ambientCG maps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from struct import pack
from zlib import compress

import numpy as np
from fit import _resize, write_coefficients
from PIL import Image
from warp_optix._src.texture_coefficients import ANALYTIC_MATERIALS, DIRECT_REFERENCES, decode, read_index, read_record

DETAIL_MAPS = ("color", "normal", "roughness", "displacement")
PATCH = 16
OVERLAP = 4
ENVELOPE = 8
STRUCTURED = ("WoodFloor", "Planks", "Bricks", "PavingStones", "Tiles", "Fabric", "Road", "Marble", "Metal")
LAYOUTS = {"Fabric004": 2}


def _open_map(path, kind):
    with Image.open(path) as image:
        if kind in ("color", "normal"):
            return image.convert("RGB")
        if image.mode.startswith("I"):
            return image.convert("F")
        return image.convert("L")


def _residual(image, kind, resolution):
    fine = _resize(image, resolution)
    blurred = _resize(image, resolution // 2).resize(fine.size, Image.Resampling.BICUBIC)
    values = np.asarray(fine, dtype=np.float32) - np.asarray(blurred, dtype=np.float32)
    if kind == "normal":
        return values[..., :2]
    return values if values.ndim == 3 else values[..., None]


def _choose_origin(residual):
    size = residual.shape[0]
    span = PATCH + OVERLAP
    candidates = [
        (y, x) for y in np.linspace(0, size - span, 8, dtype=int) for x in np.linspace(0, size - span, 8, dtype=int)
    ]
    contrast = [np.std(residual[y : y + span, x : x + span]) for y, x in candidates]
    # Favor a representative patch over outliers such as a single black grout line.
    return candidates[int(np.argsort(contrast)[len(candidates) * 3 // 5])]


def _tile(residual, origin):
    y, x = origin
    span = PATCH + OVERLAP
    crop = residual[y : y + span, x : x + span].copy()
    t = np.linspace(0.0, 1.0, OVERLAP, dtype=np.float32)
    t *= t * (3 - 2 * t)
    crop[:, :OVERLAP] = crop[:, PATCH:] * (1 - t[None, :, None]) + crop[:, :OVERLAP] * t[None, :, None]
    crop[:OVERLAP] = crop[PATCH:] * (1 - t[:, None, None]) + crop[:OVERLAP] * t[:, None, None]
    tile = crop[:PATCH, :PATCH].copy()
    tile -= tile.mean(axis=(0, 1), keepdims=True)
    tile *= np.minimum(3.0, np.std(residual, axis=(0, 1)) / np.maximum(np.std(tile, axis=(0, 1)), 1e-4))
    return tile / 255.0


def _envelope(values):
    cell = values.shape[0] // ENVELOPE
    power = np.mean(values.reshape(ENVELOPE, cell, ENVELOPE, cell, -1) ** 2, axis=(1, 3, 4))
    relative = np.sqrt(power / max(float(power.mean()), 1e-6))
    return np.rint(np.clip(relative, 0.0, 3.0) * 85.0).astype(np.uint8)


def _edge_correction(image, model, kind):
    base, _ = decode(model, kind)
    size = min(1024, max(image.size))
    if size < 512:
        return None
    size = 1024 if size >= 1024 else 512
    source = np.asarray(_resize(image, size), dtype=np.float32)
    predicted = Image.fromarray(np.rint(base * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)
    residual = source - np.asarray(predicted, dtype=np.float32)
    if kind == "color":
        residual = (residual @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32))[..., None]
    else:
        residual = residual[..., :2]
    # Drop weak changes and spend the correction budget on the position and
    # contrast of the visible edges. Quantize symmetrically around zero.
    budget = 20_800 if kind == "color" else 9_600
    for step in (2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 18.0, 24.0):
        quantized = np.rint(residual / step).clip(-127, 127).astype(np.int8)
        quantized[np.abs(residual) < 1.35 * step] = 0
        if not quantized.any():
            return None
        header = pack("<HBf", size, residual.shape[-1], step / 255.0)
        encoded = compress(header + quantized.tobytes(), level=6)
        if len(encoded) <= budget:
            return encoded
    return None


def _fit_layout(image, kind, layout):
    # Carbon fiber repeats five times per axis. Averaging corresponding
    # texels fits one high-resolution period and rejects JPEG artifacts.
    pixels = np.asarray(_resize(image, 960), dtype=np.float32)
    values = pixels.reshape(5, 192, 5, 192, -1).mean(axis=(0, 2))[None, ...]
    if kind == "normal":
        values = values[..., :2]
    data = np.rint(values).astype(np.uint8)
    header = pack("<BHHHB", layout, *data.shape)
    return compress(header + data.tobytes(), level=7)


def fit_material(asset, sources, high_sources, model_dir, model_index):
    if asset in ANALYTIC_MATERIALS or asset in DIRECT_REFERENCES:
        return {}
    images = {}
    for kind in DETAIL_MAPS:
        pattern = f"{asset}_{kind}.*"
        paths = sorted(high_sources.glob(pattern)) if high_sources else []
        paths = paths or sorted(sources.glob(pattern))
        if paths:
            images[kind] = _open_map(paths[0], kind)
    top = 2048 if min(images["color"].size) >= 2048 else 1024
    levels = (256, 512, 1024, 2048) if top == 2048 else (256, 512, 1024)
    bands = {kind: [] for kind in images}
    envelopes = {kind: [] for kind in images}
    for resolution in levels:
        color = _residual(images["color"], "color", resolution)
        origin = _choose_origin(color)
        for kind, image in images.items():
            values = color if kind == "color" else _residual(image, kind, resolution)
            bands[kind].append(_tile(values, origin))
            envelopes[kind].append(_envelope(values))
    data = {}
    for kind, tiles in bands.items():
        values = np.stack(tiles)
        # One fixed-point scale per map preserves small residuals as well as
        # high-contrast fibers without introducing a color bias.
        scale = max(float(np.max(np.abs(values))) / 63, 1e-6)
        quantized = np.rint(values / scale).astype(np.int8)
        payload = (
            pack("<BBBBf", len(levels), values.shape[-1], PATCH, ENVELOPE, scale)
            + quantized.tobytes()
            + np.stack(envelopes[kind]).tobytes()
        )
        data[kind] = compress(payload, level=6)
    if asset in LAYOUTS:
        for kind, image in images.items():
            data[f"{kind}_layout"] = _fit_layout(image, kind, LAYOUTS[asset])
    elif asset.startswith(STRUCTURED):
        for kind in ("color", "normal"):
            model = read_record(model_dir, model_index[asset][kind])
            correction = _edge_correction(images[kind], model, kind)
            if correction is not None:
                data[f"{kind}_edges"] = correction
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", type=Path)
    parser.add_argument("ids", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", required=True, type=Path, help="Fitted base coefficient package")
    parser.add_argument("--high-resolution-maps", type=Path)
    parser.add_argument("--existing-detail", type=Path, help="Refit only explicit layouts; reuse other detail records.")
    args = parser.parse_args()
    ids = args.ids.read_text().splitlines()
    model_index = read_index(args.models)
    if args.existing_detail:
        index = read_index(args.existing_detail)
        assets = {
            asset: (
                {} if asset in ANALYTIC_MATERIALS or asset in DIRECT_REFERENCES else
                {kind: read_record(args.existing_detail, shards) for kind, shards in index[asset].items()}
            )
            for asset in ids
        }
        for asset in LAYOUTS.keys() & assets.keys():
            assets[asset] = fit_material(asset, args.maps, args.high_resolution_maps, args.models, model_index)
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            maps = pool.map(
                lambda asset: fit_material(asset, args.maps, args.high_resolution_maps, args.models, model_index), ids
            )
            assets = dict(zip(ids, maps, strict=True))
    write_coefficients(assets, args.output)


if __name__ == "__main__":
    main()
