# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural hair ball made of tapered helical cubic Bezier strands."""

from __future__ import annotations

import numpy as np

from .style import rainbow_height_slots


def rainbow_segment_slots(points: np.ndarray, segments: int) -> np.ndarray:
    """Assign one palette slot per segment using gently rippled height bands."""
    roots = points[:: 3 * segments + 1]
    strand_slots = rainbow_height_slots(roots)
    return np.repeat(strand_slots, segments)


def generate_hair_ball(
    hair_count: int = 4000,
    segments: int = 5,
    ball_radius: float = 0.8,
    hair_length: float = 0.48,
    curl_radius: float = 0.065,
    curl_turns: float = 1.65,
    root_radius: float = 0.012,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack disjoint tapered helical hairs as C1-continuous cubic Beziers."""
    if hair_count < 1:
        raise ValueError("hair_count must be positive")
    if segments < 2:
        raise ValueError("segments must be at least 2")
    if min(ball_radius, hair_length, root_radius) <= 0.0 or curl_radius < 0.0:
        raise ValueError("ball_radius, hair_length, and root_radius must be positive")

    rng = np.random.default_rng(seed)
    root_id = np.arange(hair_count, dtype=np.float64)
    y = 1.0 - 2.0 * (root_id + 0.5) / hair_count
    azimuth = root_id * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.cos(azimuth),
            y,
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.sin(azimuth),
        )
    )
    radial += rng.normal(scale=0.012, size=radial.shape)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    reference = np.zeros_like(radial)
    reference[:, 1] = 1.0
    reference[np.abs(radial[:, 1]) > 0.9] = (1.0, 0.0, 0.0)
    tangent = np.cross(reference, radial)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(radial, tangent)

    phase = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    length = hair_length * rng.uniform(0.78, 1.18, (hair_count, 1))
    turns = curl_turns * rng.uniform(0.72, 1.28, (hair_count, 1))
    curl = curl_radius * rng.uniform(0.65, 1.35, (hair_count, 1))
    lean_angle = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    lean = np.cos(lean_angle)[..., None] * tangent[:, None, :] + np.sin(lean_angle)[..., None] * bitangent[:, None, :]

    u = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)[None, :]
    angle = phase + 2.0 * np.pi * turns * u
    envelope = np.sin(0.5 * np.pi * u) ** 1.25
    helix = np.cos(angle)[..., None] * tangent[:, None, :] + np.sin(angle)[..., None] * bitangent[:, None, :]
    center = (ball_radius + length * u)[..., None] * radial[:, None, :]
    knots = center + (curl * envelope)[..., None] * helix + (0.13 * length * u * u)[..., None] * lean

    width_scale = rng.uniform(0.8, 1.2, (hair_count, 1))
    radius_knots = root_radius * width_scale * (0.08 + 0.92 * (1.0 - u) ** 0.75)
    step = 1.0 / segments
    derivatives = np.empty_like(knots)
    derivatives[:, 0] = (knots[:, 1] - knots[:, 0]) / step
    derivatives[:, -1] = (knots[:, -1] - knots[:, -2]) / step
    derivatives[:, 1:-1] = (knots[:, 2:] - knots[:, :-2]) / (2.0 * step)
    radius_derivatives = np.empty_like(radius_knots)
    radius_derivatives[:, 0] = (radius_knots[:, 1] - radius_knots[:, 0]) / step
    radius_derivatives[:, -1] = (radius_knots[:, -1] - radius_knots[:, -2]) / step
    radius_derivatives[:, 1:-1] = (radius_knots[:, 2:] - radius_knots[:, :-2]) / (2.0 * step)

    points_per_hair = 3 * segments + 1
    points = np.empty((hair_count, points_per_hair, 3), dtype=np.float64)
    radii = np.empty((hair_count, points_per_hair), dtype=np.float64)
    points[:, 0] = knots[:, 0]
    radii[:, 0] = radius_knots[:, 0]
    points[:, 1::3] = knots[:, :-1] + derivatives[:, :-1] * (step / 3.0)
    points[:, 2::3] = knots[:, 1:] - derivatives[:, 1:] * (step / 3.0)
    points[:, 3::3] = knots[:, 1:]
    radii[:, 1::3] = radius_knots[:, :-1] + radius_derivatives[:, :-1] * (step / 3.0)
    radii[:, 2::3] = radius_knots[:, 1:] - radius_derivatives[:, 1:] * (step / 3.0)
    radii[:, 3::3] = radius_knots[:, 1:]
    radii = np.maximum(radii, root_radius * 0.01)
    segment_indices = (
        np.arange(hair_count, dtype=np.uint32)[:, None] * points_per_hair
        + 3 * np.arange(segments, dtype=np.uint32)[None, :]
    ).reshape(-1)
    return (
        points.astype(np.float32).reshape(-1, 3),
        radii.astype(np.float32).reshape(-1),
        segment_indices,
    )
