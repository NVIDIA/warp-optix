# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared defaults for interactive path-tracing viewers."""

DEFAULT_VIEWER_WIDTH = 1920
DEFAULT_VIEWER_HEIGHT = 1080
DEFAULT_VIEWER_SIZE = (DEFAULT_VIEWER_WIDTH, DEFAULT_VIEWER_HEIGHT)
# NGX application identifier of the warp_optix path tracer.
DLSS_PROJECT_ID = "dddbee68-a452-4fab-9371-f9575480a154"

__all__ = ["DEFAULT_VIEWER_HEIGHT", "DEFAULT_VIEWER_SIZE", "DEFAULT_VIEWER_WIDTH", "DLSS_PROJECT_ID"]
