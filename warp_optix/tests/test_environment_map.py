# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from warp_optix.pathtracing.environment_map import _build_alias_table


@pytest.mark.parametrize("size", [7, 4096])
def test_alias_table_reproduces_importance_distribution(size):
    importance = np.random.default_rng(size).random(size) ** 4
    importance[::5] = 0.0
    alias, q = _build_alias_table(importance, float(importance.sum()))

    # Texel i keeps probability q[i] / size and receives (1 - q[j]) / size from every j aliased to it.
    others = alias != np.arange(size)
    selected = q.astype(np.float64)
    np.add.at(selected, alias[others], 1.0 - q[others].astype(np.float64))
    np.testing.assert_allclose(selected / size, importance / importance.sum(), atol=1e-5)
