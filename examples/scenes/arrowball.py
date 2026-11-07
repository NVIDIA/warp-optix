# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Randomized outward arrows on a sphere, colored in rainbow height bands."""

from __future__ import annotations

import numpy as np

from .style import rainbow_height_slots


def generate_arrow_ball(
    arrow_count: int = 1800,
    ball_radius: float = 0.8,
    arrow_length: float = 0.42,
    swirl: float = 0.24,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate randomized outward arrows on a Fibonacci sphere."""
    if arrow_count < 1:
        raise ValueError("arrow_count must be positive")
    if ball_radius <= 0.0 or arrow_length <= 0.0:
        raise ValueError("ball_radius and arrow_length must be positive")
    if swirl < 0.0:
        raise ValueError("swirl must be non-negative")

    rng = np.random.default_rng(seed)
    arrow_id = np.arange(arrow_count, dtype=np.float64)
    y = 1.0 - 2.0 * (arrow_id + 0.5) / arrow_count
    azimuth = arrow_id * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.cos(azimuth),
            y,
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.sin(azimuth),
        )
    )
    radial += rng.normal(scale=0.008, size=radial.shape)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    reference = np.zeros_like(radial)
    reference[:, 1] = 1.0
    reference[np.abs(radial[:, 1]) > 0.9] = (1.0, 0.0, 0.0)
    tangent = np.cross(reference, radial)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(radial, tangent)
    phase = rng.uniform(0.0, 2.0 * np.pi, (arrow_count, 1))
    tangential = np.cos(phase) * tangent + np.sin(phase) * bitangent
    direction = radial + float(swirl) * tangential
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    starts = float(ball_radius) * radial
    lengths = float(arrow_length) * rng.uniform(0.78, 1.22, (arrow_count, 1))
    ends = starts + lengths * direction
    slots = rainbow_height_slots(starts)
    return starts.astype(np.float32), ends.astype(np.float32), slots
