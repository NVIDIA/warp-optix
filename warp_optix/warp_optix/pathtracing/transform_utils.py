# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def quaternion_to_matrix3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def build_transform_matrix(
    position: Iterable[float],
    rotation_xyzw: Iterable[float],
    scale: float | Iterable[float] = 1.0,
) -> np.ndarray:
    """Build a row-major 4x4 transform matrix."""
    px, py, pz = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in rotation_xyzw]
    if isinstance(scale, Iterable) and not isinstance(scale, (str, bytes)):
        sx, sy, sz = [float(v) for v in scale]
    else:
        s = float(scale)
        sx = sy = sz = s

    matrix = np.eye(4, dtype=np.float32)
    rot = quaternion_to_matrix3(qx, qy, qz, qw)
    rot[:, 0] *= sx
    rot[:, 1] *= sy
    rot[:, 2] *= sz
    matrix[:3, :3] = rot
    matrix[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return matrix


def mat4_to_optix_transform12(matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to OptiX 3x4 row-major transform array."""
    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return np.array(
        [
            matrix[0, 0],
            matrix[0, 1],
            matrix[0, 2],
            matrix[0, 3],
            matrix[1, 0],
            matrix[1, 1],
            matrix[1, 2],
            matrix[1, 3],
            matrix[2, 0],
            matrix[2, 1],
            matrix[2, 2],
            matrix[2, 3],
        ],
        dtype=np.float32,
    )
