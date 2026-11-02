# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples_warp"))

from example_warp_optix_pathtraced_arrowball import generate_arrow_ball  # noqa: E402


def test_arrow_ball_is_outward_randomized_and_deterministic():
    kwargs = dict(
        arrow_count=2000,
        ball_radius=0.8,
        arrow_length=0.42,
        swirl=0.24,
        seed=123,
    )
    starts, ends, slots = generate_arrow_ball(**kwargs)
    repeated = generate_arrow_ball(**kwargs)

    assert starts.shape == ends.shape == (2000, 3)
    assert slots.shape == (2000,)
    assert slots.dtype == np.uint32
    np.testing.assert_array_equal(starts, repeated[0])
    np.testing.assert_array_equal(ends, repeated[1])
    np.testing.assert_array_equal(slots, repeated[2])
    np.testing.assert_allclose(np.linalg.norm(starts, axis=1), 0.8, atol=1.0e-6)
    lengths = np.linalg.norm(ends - starts, axis=1)
    assert np.all(lengths >= 0.42 * 0.78 - 1.0e-6)
    assert np.all(lengths <= 0.42 * 1.22 + 1.0e-6)
    assert np.all(np.sum(starts * (ends - starts), axis=1) > 0.0)
    np.testing.assert_array_equal(np.unique(slots), np.arange(12, dtype=np.uint32))
    assert np.corrcoef(starts[:, 1], slots)[0, 1] > 0.98


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arrow_count": 0}, "arrow_count"),
        ({"ball_radius": 0.0}, "positive"),
        ({"arrow_length": 0.0}, "positive"),
        ({"swirl": -0.1}, "non-negative"),
    ],
)
def test_arrow_ball_rejects_invalid_parameters(kwargs, message):
    parameters = dict(
        arrow_count=8, ball_radius=0.8, arrow_length=0.42, swirl=0.24, seed=1
    )
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        generate_arrow_ball(**parameters)
