# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public namespace contracts."""

import subprocess
import sys

import warp_optix as wo
from warp_optix import native


def test_import_boundary_in_fresh_process(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import warp_optix as wo
import warp as wp
import optix
assert 'warp_nn' not in sys.modules
assert 'pyglet' not in sys.modules
assert 'imgui_bundle' not in sys.modules
assert wp._src.context.runtime is None
assert wo.__version__ == '0.2.0'
from warp_optix import native
assert native.version() == (9, 1, 0)
assert native.DeviceContext is optix.DeviceContext
assert native.Module is optix.Module
assert 'warp_optix._bindings' not in sys.modules
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_public_enums_match_bundled_sdk():
    assert int(wo.PrimitiveType.ROUND_LINEAR) == int(native.PRIMITIVE_TYPE_ROUND_LINEAR)
    flags = wo.BuildFlags.PREFER_FAST_TRACE | wo.BuildFlags.ALLOW_UPDATE
    assert int(flags) == int(native.BUILD_FLAG_PREFER_FAST_TRACE) | int(native.BUILD_FLAG_ALLOW_UPDATE)
