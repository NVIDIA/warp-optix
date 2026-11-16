# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package version, read from the installed distribution metadata.

``[project].version`` in ``pyproject.toml`` is the single source of truth.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("warp-optix")
except PackageNotFoundError:
    __version__ = "unknown"
