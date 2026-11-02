# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact native-curve representation for dynamic arrow batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(slots=True)
class ArrowBatch:
    """A fixed-capacity batch of two-segment native-curve arrows.

    Each arrow occupies four control points: two for its constant-radius shaft
    and two for its tapered tip.  Unused slots have zero radius, which keeps the
    OptiX primitive count fixed while making those slots non-intersecting.
    """

    geometry_id: int
    instance_id: int
    capacity: int
    small_radius: float
    large_radius: float
    tip_length_ratio: float
    active_count: int = 0


def arrow_segment_indices(capacity: int) -> np.ndarray:
    """Return the two disjoint round-linear segment starts per arrow."""
    arrow_ids = np.arange(int(capacity), dtype=np.uint32)
    return np.column_stack((4 * arrow_ids, 4 * arrow_ids + 2)).reshape(-1)


def fill_arrow_curve_buffers(
    starts: np.ndarray,
    ends: np.ndarray,
    vertices: np.ndarray,
    radii: np.ndarray,
    small_radius: float,
    large_radius: float,
    tip_length_ratio: float,
) -> np.ndarray:
    """Fill fixed-capacity curve buffers and return valid-arrow flags."""
    starts = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
    ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have matching shape (N, 3)")
    if len(starts) > len(vertices) // 4:
        raise ValueError("arrow count exceeds batch capacity")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("arrow endpoints must be finite")

    vertices.fill(0.0)
    radii.fill(0.0)
    delta = ends - starts
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1.0e-8
    if not np.any(valid):
        return valid

    directions = np.zeros_like(delta)
    directions[valid] = delta[valid] / lengths[valid, None]
    necks = ends - directions * (lengths * float(tip_length_ratio))[:, None]
    packed = vertices[: 4 * len(starts)].reshape(-1, 4, 3)
    packed[:, 0] = starts
    packed[:, 1] = necks
    packed[:, 2] = necks
    packed[:, 3] = ends
    packed_radii = radii[: 4 * len(starts)].reshape(-1, 4)
    packed_radii[:, 0:2] = float(small_radius)
    packed_radii[:, 2] = float(large_radius)
    packed_radii[:, 3] = 0.0
    packed[~valid] = 0.0
    packed_radii[~valid] = 0.0
    return valid


@wp.func
def _write_arrow_curve(
    arrow_id: int,
    active: bool,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    small_radius: float,
    large_radius: float,
    tip_length_ratio: float,
    vertices: wp.array(dtype=wp.float32),
    radii: wp.array(dtype=wp.float32),
):
    point_base = 4 * arrow_id
    float_base = 3 * point_base
    start = wp.vec3(0.0)
    neck = wp.vec3(0.0)
    end = wp.vec3(0.0)
    shaft_radius = 0.0
    tip_radius = 0.0
    if active:
        start = starts[arrow_id]
        end = ends[arrow_id]
        delta = end - start
        length = wp.length(delta)
        if length > 1.0e-8:
            neck = end - delta * tip_length_ratio
            shaft_radius = small_radius
            tip_radius = large_radius
        else:
            start = wp.vec3(0.0)
            end = wp.vec3(0.0)

    vertices[float_base + 0] = start[0]
    vertices[float_base + 1] = start[1]
    vertices[float_base + 2] = start[2]
    vertices[float_base + 3] = neck[0]
    vertices[float_base + 4] = neck[1]
    vertices[float_base + 5] = neck[2]
    vertices[float_base + 6] = neck[0]
    vertices[float_base + 7] = neck[1]
    vertices[float_base + 8] = neck[2]
    vertices[float_base + 9] = end[0]
    vertices[float_base + 10] = end[1]
    vertices[float_base + 11] = end[2]
    radii[point_base + 0] = shaft_radius
    radii[point_base + 1] = shaft_radius
    radii[point_base + 2] = tip_radius
    radii[point_base + 3] = 0.0


@wp.kernel
def update_arrow_curves_host_count(
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    active_count: int,
    small_radius: float,
    large_radius: float,
    tip_length_ratio: float,
    vertices: wp.array(dtype=wp.float32),
    radii: wp.array(dtype=wp.float32),
):
    arrow_id = wp.tid()
    _write_arrow_curve(
        arrow_id,
        arrow_id < active_count,
        starts,
        ends,
        small_radius,
        large_radius,
        tip_length_ratio,
        vertices,
        radii,
    )


@wp.kernel
def update_arrow_curves_device_count(
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    active_count: wp.array(dtype=wp.int32),
    capacity: int,
    small_radius: float,
    large_radius: float,
    tip_length_ratio: float,
    vertices: wp.array(dtype=wp.float32),
    radii: wp.array(dtype=wp.float32),
):
    arrow_id = wp.tid()
    count = wp.clamp(active_count[0], 0, capacity)
    _write_arrow_curve(
        arrow_id,
        arrow_id < count,
        starts,
        ends,
        small_radius,
        large_radius,
        tip_length_ratio,
        vertices,
        radii,
    )


@wp.kernel
def expand_arrow_material_ids_host_count(
    material_ids: wp.array(dtype=wp.int32),
    active_count: int,
    output: wp.array(dtype=wp.uint32),
    output_offset: int,
):
    arrow_id = wp.tid()
    if arrow_id < active_count:
        material_id = wp.uint32(wp.max(material_ids[arrow_id], 0))
        output[output_offset + 2 * arrow_id] = material_id
        output[output_offset + 2 * arrow_id + 1] = material_id


@wp.kernel
def expand_arrow_material_ids_device_count(
    material_ids: wp.array(dtype=wp.int32),
    active_count: wp.array(dtype=wp.int32),
    capacity: int,
    output: wp.array(dtype=wp.uint32),
    output_offset: int,
):
    arrow_id = wp.tid()
    count = wp.clamp(active_count[0], 0, capacity)
    if arrow_id < count:
        material_id = wp.uint32(wp.max(material_ids[arrow_id], 0))
        output[output_offset + 2 * arrow_id] = material_id
        output[output_offset + 2 * arrow_id + 1] = material_id
