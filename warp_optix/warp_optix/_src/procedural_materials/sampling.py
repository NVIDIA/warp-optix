# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.func
def texel_uv(row: int, col: int, height: int, width: int) -> wp.vec2:
    return wp.vec2((float(col) + 0.5) / float(width), (float(row) + 0.5) / float(height))


@wp.func
def periodic_noise(uv: wp.vec2, seed: int, nx: int, ny: int) -> float:
    return wp.pnoise(wp.rand_init(seed), wp.vec2(uv[0] * float(nx), uv[1] * float(ny)), nx, ny)
