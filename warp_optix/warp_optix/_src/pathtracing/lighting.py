# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared physical-light unit conversions for the path tracer."""

# The physical sky maps a reference daylight luminance of 80,000 nits to one
# renderer-linear radiance unit. Physical USD light emission must use the same scale.
RENDERER_RADIANCE_PER_NIT = 1.0 / 80000.0
