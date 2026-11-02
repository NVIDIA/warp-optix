# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples_warp"))

from example_warp_optix_pathtraced_hairball import (  # noqa: E402
    generate_hair_ball,
    rainbow_segment_slots,
)


def test_hair_ball_geometry_is_disjoint_tapered_and_deterministic():
    kwargs = dict(
        hair_count=7,
        segments=5,
        ball_radius=0.8,
        hair_length=0.45,
        curl_radius=0.06,
        curl_turns=1.5,
        root_radius=0.012,
        seed=123,
    )
    points, radii, starts = generate_hair_ball(**kwargs)
    repeated = generate_hair_ball(**kwargs)

    assert points.shape == (112, 3)
    assert radii.shape == (112,)
    assert starts.shape == (35,)
    np.testing.assert_array_equal(points, repeated[0])
    np.testing.assert_array_equal(radii, repeated[1])
    np.testing.assert_array_equal(starts, repeated[2])
    np.testing.assert_allclose(np.linalg.norm(points[::16], axis=1), 0.8, atol=1.0e-6)
    assert np.all(radii > 0.0)
    assert np.all(radii[::16] > radii[15::16])
    assert not np.any(np.isin(np.arange(15, 112, 16, dtype=np.uint32), starts))

    curves = points.reshape(7, 16, 3)
    for join in (3, 6, 9, 12):
        left_tangent = curves[:, join] - curves[:, join - 1]
        right_tangent = curves[:, join + 1] - curves[:, join]
        np.testing.assert_allclose(left_tangent, right_tangent, atol=2.0e-7)

    np.testing.assert_array_equal(
        starts,
        (np.arange(7, dtype=np.uint32)[:, None] * 16 + (0, 3, 6, 9, 12)).reshape(-1),
    )


def test_rainbow_materials_are_uniform_per_strand_and_flow_bottom_to_top():
    hair_count = 2000
    segments = 5
    points, _radii, _starts = generate_hair_ball(
        hair_count, segments, 0.8, 0.45, 0.06, 1.5, 0.012, 123
    )
    slots = rainbow_segment_slots(points, segments)
    strand_slots = slots.reshape(hair_count, segments)
    root_heights = points[:: 3 * segments + 1, 1]

    assert slots.dtype == np.uint32
    assert slots.shape == (hair_count * segments,)
    assert np.all(strand_slots == strand_slots[:, :1])
    np.testing.assert_array_equal(np.unique(slots), np.arange(12, dtype=np.uint32))
    assert np.corrcoef(root_heights, strand_slots[:, 0])[0, 1] > 0.98


@pytest.mark.parametrize("hair_count,segments", [(0, 5), (1, 1)])
def test_hair_ball_rejects_invalid_topology(hair_count, segments):
    with pytest.raises(ValueError):
        generate_hair_ball(hair_count, segments, 0.8, 0.45, 0.06, 1.5, 0.012, 1)
