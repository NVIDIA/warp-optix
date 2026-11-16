# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compress and store a small neural texture using the optional warp-nn backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from neural_texture import (
    compress_texture,
    decode_image,
    load_asset,
    save_asset,
)


def make_material_texture(width: int, height: int) -> np.ndarray:
    """Create a periodic RGB material texture with several spatial frequencies."""
    x = np.arange(width, dtype=np.float32)[None, :] / width
    y = np.arange(height, dtype=np.float32)[:, None] / height
    checker = ((np.floor(x * 8.0) + np.floor(y * 8.0)) % 2.0).astype(np.float32)
    rings = 0.5 + 0.5 * np.cos(18.0 * np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2))
    texture = np.empty((height, width, 3), dtype=np.float32)
    texture[..., 0] = 0.1 + 0.75 * checker
    texture[..., 1] = 0.15 + 0.75 * rings
    texture[..., 2] = 0.2 + 0.7 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (3.0 * x + 2.0 * y)))
    return np.clip(texture, 0.0, 1.0)


def load_texture(path: Path) -> np.ndarray:
    """Load an HWC texture from a safe NumPy array file."""
    image = np.load(path, allow_pickle=False)
    if image.ndim == 2:
        image = image[..., None]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional HWC .npy texture; otherwise generate a procedural RGB texture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("material.wnt"),
        help="Output neural-texture asset.",
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        help="Optionally store the decoded HWC reference image as .npy.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--refinement-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument(
        "--latent-scale",
        type=int,
        help="Advanced: fix the latent scale instead of automatic selection.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--srgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    image = load_texture(args.input) if args.input is not None else make_material_texture(args.width, args.height)
    result = compress_texture(
        image,
        steps=args.steps,
        refinement_steps=args.refinement_steps,
        learning_rate=args.learning_rate,
        latent_scale=args.latent_scale,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        channel_name="albedo",
        color_space="srgb" if args.srgb else "linear",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_asset(args.output, result.asset)
    stored = load_asset(args.output)

    if args.reconstruction is not None:
        args.reconstruction.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.reconstruction, decode_image(stored, batch_size=args.batch_size))

    dense_bytes = image.nbytes
    file_bytes = args.output.stat().st_size
    print(f"stored {args.output} ({file_bytes:,} bytes)")
    print(f"portable payload: {stored.storage_bytes:,} bytes")
    print(f"dense float32 input: {dense_bytes:,} bytes")
    print(f"file compression ratio: {dense_bytes / file_bytes:.2f}x")
    estimated = result.evaluation_pixels < stored.width * stored.height
    label = "Estimated PSNR" if estimated else "Full-image PSNR"
    print(f"{label}: {result.psnr:.2f} dB")
    if result.selection_reason:
        print(result.selection_reason)
    if result.losses:
        print(f"training loss: {result.losses[0]:.6g} -> {result.losses[-1]:.6g}")


if __name__ == "__main__":
    main()
