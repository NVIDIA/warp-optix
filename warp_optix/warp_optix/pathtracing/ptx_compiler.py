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

"""OptiX header discovery helpers."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def get_optix_include_dir(optix_module=None) -> str | None:
    """Get the OptiX include directory used for downstream compilation."""

    def _has_optix_device_header(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "optix_device.h"))

    def _parse_version_from_path(path: str) -> tuple[int, ...]:
        # Windows example: ".../OptiX SDK 9.0.0/include"
        m = re.search(r"OptiX SDK (\d+(?:\.\d+)*)", path)
        if not m:
            return (0,)
        return tuple(int(p) for p in m.group(1).split("."))

    # Preferred path: query the include directory from the installed wrapper.
    if optix_module is not None and hasattr(optix_module, "get_optix_include_dir"):
        try:
            optix_dir = optix_module.get_optix_include_dir()
        except Exception:
            optix_dir = None
        if optix_dir and os.path.isdir(optix_dir) and _has_optix_device_header(optix_dir):
            logger.info("OptiX header found: %s", os.path.join(optix_dir, "optix_device.h"))
            return optix_dir
        logger.warning(
            "[Viewer] optix.get_optix_include_dir() did not provide a directory "
            "with optix_device.h; falling back to SDK discovery."
        )

    env_candidates = [
        os.environ.get("OPTIX_SDK_INCLUDE_DIR"),
        os.environ.get("OPTIX_INCLUDE_DIR"),
    ]
    for path in env_candidates:
        if path and os.path.isdir(path) and _has_optix_device_header(path):
            logger.info("Using OptiX include directory from environment: %s", path)
            return path

    discovered: list[str] = []
    windows_sdk_root = Path("C:/ProgramData/NVIDIA Corporation")
    if windows_sdk_root.is_dir():
        for include_dir in windows_sdk_root.glob("OptiX SDK */include"):
            if include_dir.is_dir() and _has_optix_device_header(str(include_dir)):
                discovered.append(str(include_dir))

    for include_dir in (
        Path("/opt/optix/include"),
        Path.home() / "optix" / "include",
    ):
        if include_dir.is_dir() and _has_optix_device_header(str(include_dir)):
            discovered.append(str(include_dir))

    discovered = sorted(set(discovered), key=_parse_version_from_path, reverse=True)
    for path in discovered:
        if os.path.isdir(path) and _has_optix_device_header(path):
            return path

    return None
