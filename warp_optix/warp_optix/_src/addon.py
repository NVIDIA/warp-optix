# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp addon registration and the build options of OptiX modules."""

from __future__ import annotations

import functools
from pathlib import Path

import warp as wp

from warp_optix import native
from warp_optix._src.builtins import register_addon_builtins

_HEADER_DIR = Path(__file__).resolve().parents[1] / "_native" / "include"

# Warp's generated source defines function-like ``float`` and ``int`` macros
# before addon preambles. CUDA's half headers declare conversion operators with
# those names, so the macros are suspended while the OptiX headers are parsed.
_PREAMBLE = (
    "#undef float\n"
    "#undef int\n"
    '#include "warp_optix_builtins.h"\n'
    "#define float(x) cast_float(x)\n"
    "#define int(x) cast_int(x)\n"
)


@functools.cache
def build_options() -> wp.ModuleBuildOptions:
    """Return the build options that make the OptiX builtins available to a module."""
    # No CUDA Toolkit include directory: warp_optix_builtins.h supplies the only
    # CUDA type the OptiX device headers need.
    return wp.ModuleBuildOptions(
        extra_cuda_include_dirs=[str(_HEADER_DIR), native.get_optix_include_dir()],
        extra_cuda_preamble=_PREAMBLE,
        extra_build_dependencies=[str(_HEADER_DIR / "warp_optix_builtins.h")],
    )


@functools.cache
def register_with_warp() -> None:
    """Register the OptiX builtins with Warp once per process."""
    register_addon_builtins()
