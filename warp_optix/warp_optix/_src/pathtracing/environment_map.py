# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Modified Python/Warp adaptation of environment-map alias-table construction:
# https://github.com/nvpro-samples/nvpro_core/blob/bc19d6ac3ef62938d0ea0e099735878457ce1b6e/nvvkhl/hdr_env.cpp
# Original notices: licenses/vk_denoise_dlssrr-NOTICE.txt and
# licenses/nvpro_core-NOTICE.txt.

#
# Environment map loading, processing and importance sampling for OptiX path tracing.
# Aligned with the environment-map sampling flow used in the reference sample.

"""
Environment map handling with importance sampling using the alias method.

This module provides:
- HDR image loading (via imageio or pure Python fallback)
- Importance-based alias table construction for efficient environment sampling
- GPU buffer creation for OptiX shader consumption

The alias table enables O(1) importance sampling of environment maps,
crucial for efficient Monte Carlo path tracing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

# Try to import imageio for HDR loading, fall back to pure Python
try:
    import imageio.v3 as iio

    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


@wp.kernel(module="unique", enable_backward=False)
def _pair_alias_table(
    partition: wp.array[wp.uint32], small_count: int, q: wp.array[wp.float32], alias: wp.array[wp.uint32]
):
    size = partition.shape[0]
    s = int(0)
    large = small_count
    while s < large and large < size:
        small_index = partition[s]
        high_index = partition[large]
        # Associate the lower-energy texel with its higher-energy alias and move
        # to the next high-energy texel once this one falls below average.
        alias[small_index] = high_index
        q[high_index] = q[high_index] - (1.0 - q[small_index])
        if q[high_index] < 1.0:
            large += 1
        s += 1


def _build_alias_table(importance: np.ndarray, total_importance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build alias table for O(1) importance sampling.

    This implements the alias method as described in:
    https://arxiv.org/pdf/1901.05423.pdf, section 2.6

    Implements the alias-table construction used by the reference sample.

    Args:
        importance: Array of importance values (area * max(R,G,B)) for each pixel
        total_importance: Sum of all importance values

    Returns:
        Tuple of (alias_indices, q_values) arrays
    """
    size = len(importance)

    # Handle edge case of zero total importance
    if total_importance < 1e-10:
        total_importance = float(size)
        importance = np.ones(size, dtype=np.float64)

    # For each texel, compute the ratio q between the emitted radiance and the average
    # q = importance[i] / average_importance = importance[i] * size / sum
    q = (np.asarray(importance, dtype=np.float64) * (float(size) / total_importance)).astype(np.float32)
    alias = np.arange(size, dtype=np.uint32)

    # Texels with q < 1 (below average) are stored from the beginning, texels
    # with q >= 1 (above average) from the end, in reverse order.
    below = q < 1.0
    small = np.flatnonzero(below).astype(np.uint32)
    partition = np.concatenate([small, np.flatnonzero(~below)[::-1].astype(np.uint32)])

    q_device = wp.array(q, dtype=wp.float32, device="cpu")
    alias_device = wp.array(alias, dtype=wp.uint32, device="cpu")
    wp.launch(
        _pair_alias_table,
        dim=1,
        inputs=[wp.array(partition, dtype=wp.uint32, device="cpu"), len(small), q_device, alias_device],
        device="cpu",
    )
    return alias_device.numpy(), q_device.numpy()


class EnvironmentMap:
    """
    HDR environment map with importance sampling acceleration structure.

    This class loads HDR images and builds the alias table for efficient
    importance sampling in OptiX shaders.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        pixels: RGBA float32 pixel data (height, width, 4) with PDF in alpha
        integral: Total radiance integral
        average: Average CIE luminance
        rotation: Environment rotation in radians
    """

    def __init__(self):
        """Create an empty environment map."""
        self.width = 0
        self.height = 0
        self.pixels: np.ndarray | None = None
        self.integral = 0.0
        self.average = 0.0
        self.rotation = 0.0

        # GPU buffers
        self._env_map_buffer: wp.array | None = None
        self._env_accel_buffer: wp.array | None = None

    @property
    def env_map_address(self) -> int:
        """Get device address of environment map texture data."""
        return self._env_map_buffer.ptr if self._env_map_buffer is not None else 0

    @property
    def env_accel_address(self) -> int:
        """Get device address of environment acceleration structure."""
        return self._env_accel_buffer.ptr if self._env_accel_buffer is not None else 0

    @property
    def accel_count(self) -> int:
        """Get number of entries in acceleration structure."""
        return self.width * self.height

    def load_from_file(self, path: str | Path, scaling: float = 1.0) -> bool:
        """
        Load an HDR environment map from file.

        Supports .hdr (Radiance RGBE) format via imageio or pure Python fallback.

        Args:
            path: Path to HDR file
            scaling: Intensity multiplier

        Returns:
            True if loading succeeded
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Environment map file not found: %s", path)
            return False

        # Try imageio first
        arr = None
        loader = "imageio"

        if _HAS_IMAGEIO:
            try:
                arr = np.asarray(iio.imread(str(path)))
                # Validate it's actually HDR data (not LDR)
                if arr is not None:
                    if np.issubdtype(arr.dtype, np.integer):
                        arr = None  # Integer data indicates LDR
                    else:
                        test = arr[..., :3] if arr.ndim >= 3 else arr
                        vmax = float(np.nanmax(test.astype(np.float32)))
                        if vmax <= 1.0:
                            arr = None  # Values clamped to [0,1] indicates LDR
            except Exception:
                arr = None

        # Fall back to pure Python HDR loader
        if arr is None:
            from .hdr_loader import load_hdr  # noqa: PLC0415

            loader = "hdr_loader"
            try:
                arr, _, _ = load_hdr(str(path))
                # hdr_loader returns bottom-up; flip to top-down
                arr = np.flipud(arr)
            except Exception as e:
                logger.warning("Failed to load HDR '%s': %s", path, e)
                return False

        return self.load_from_array(arr, scaling=scaling, source_path=str(path), loader=loader)

    def load_from_array(
        self,
        rgb_data: np.ndarray,
        scaling: float = 1.0,
        source_path: str = "<array>",
        loader: str = "array",
    ) -> bool:
        """
        Load environment map from RGB float array.

        Args:
            rgb_data: RGB or RGBA float array of shape (H, W, 3) or (H, W, 4)
            scaling: Intensity multiplier
            source_path: Source path for logging
            loader: Loader name for logging

        Returns:
            True if loading succeeded
        """
        arr = np.asarray(rgb_data, dtype=np.float32)

        # Handle various input shapes
        if arr.ndim == 2:
            # Grayscale - expand to RGB
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3:
            logger.warning("Invalid environment array shape: %s", arr.shape)
            return False

        # Ensure RGB or RGBA
        if arr.shape[-1] == 3:
            alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
            arr = np.concatenate([arr, alpha], axis=-1)
        elif arr.shape[-1] > 4:
            arr = arr[..., :4]

        # Apply scaling
        if scaling != 1.0:
            arr[..., :3] *= scaling

        self.height, self.width = arr.shape[0], arr.shape[1]

        # Build importance sampling acceleration structure
        self._build_acceleration(arr)

        # Log info
        vmin = float(np.nanmin(arr[..., :3]))
        vmax = float(np.nanmax(arr[..., :3]))
        logger.info(
            "Loaded env map %s (%dx%d) via %s, range=[%.2f, %.2f], integral=%.2f, avg=%.4f",
            source_path,
            self.width,
            self.height,
            loader,
            vmin,
            vmax,
            self.integral,
            self.average,
        )

        if vmax <= 1.0:
            logger.warning("Environment map max value <= 1.0; image may be LDR.")

        return True

    def load_from_color(self, color: tuple[float, float, float], width: int = 4, height: int = 2) -> bool:
        """
        Create a uniform color environment map.

        Args:
            color: RGB color values
            width: Image width
            height: Image height

        Returns:
            True always
        """
        arr = np.zeros((height, width, 4), dtype=np.float32)
        arr[..., 0] = color[0]
        arr[..., 1] = color[1]
        arr[..., 2] = color[2]
        arr[..., 3] = 1.0
        return self.load_from_array(arr, source_path=f"<color {color}>", loader="procedural")

    def _build_acceleration(self, pixels_rgba: np.ndarray):
        """
        Build the importance sampling acceleration structure.

        This computes:
        1. Solid angle weighted importance for each pixel
        2. Alias table for O(1) sampling
        3. PDF values stored in alpha channel of texture

        Follows the environment-acceleration construction used by the reference sample.
        """
        h, w = pixels_rgba.shape[0], pixels_rgba.shape[1]
        size = w * h

        # Each texel's importance is its solid angle times max(R,G,B): the energy it emits.
        # The average CIE luminance drives tonemapping.
        rgb = pixels_rgba[..., :3].astype(np.float64)
        cos_theta = np.cos(np.arange(h + 1) * (np.pi / h))
        area = (cos_theta[:-1] - cos_theta[1:]) * (2.0 * np.pi / w)
        max_component = rgb.max(axis=-1)
        importance_data = (area[:, None] * np.maximum(max_component, 0.0)).reshape(-1)
        self.average = float((rgb @ np.array([0.2126, 0.7152, 0.0722])).sum()) / size

        self.integral = float(np.sum(importance_data))
        if self.integral == 0.0:
            self.integral = 1.0

        alias, q = _build_alias_table(importance_data, self.integral)

        # PDF = max(R,G,B) / integral, stored in the alpha channel.
        pixels_rgba[..., 3] = max_component * (1.0 / self.integral)

        self.pixels = np.ascontiguousarray(pixels_rgba, dtype=np.float32)

        # Create EnvAccel structure array (C++ parity layout).
        # EnvAccel layout: { uint alias; float q; float pdf; float pad } - 16 bytes total.
        accel_dtype = np.dtype(
            [
                ("alias", np.uint32),
                ("q", np.float32),
                ("pdf", np.float32),
                ("pad", np.float32),
            ],
            align=False,  # No padding to keep the expected packed layout.
        )
        env_accel = np.zeros(size, dtype=accel_dtype)
        env_accel["alias"] = alias
        env_accel["q"] = q
        env_accel["pdf"] = self.pixels[..., 3].reshape(-1).astype(np.float32)
        env_accel["pad"] = 0.0

        # Upload to GPU
        self._upload_to_gpu(env_accel)

    def _upload_to_gpu(self, env_accel: np.ndarray):
        """Upload environment map and acceleration structure to GPU."""
        # Upload RGBA pixel data as float32
        flat_pixels = self.pixels.reshape(-1).astype(np.float32)
        self._env_map_buffer = wp.array(flat_pixels, dtype=wp.float32, device="cuda")

        # Upload acceleration structure
        accel_bytes = env_accel.view(np.uint8).reshape(-1)
        self._env_accel_buffer = wp.array(accel_bytes, dtype=wp.uint8, device="cuda")

    def set_rotation(self, radians: float):
        """Set environment rotation in radians."""
        self.rotation = radians

    def dispose(self):
        """Release GPU resources."""
        self._env_map_buffer = None
        self._env_accel_buffer = None
