# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp addon registration and the build options of OptiX modules."""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

import warp as wp

from warp_optix import native
from warp_optix.builtins import register_addon_builtins

_HEADER_DIR = Path(__file__).resolve().parent / "_native" / "include"

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


def _cuda_include_dir() -> str | None:
    """Return the CUDA Toolkit include directory required by some OptiX headers."""
    roots = [Path(os.environ[v]) for v in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(v)]
    if nvcc := shutil.which("nvcc"):
        roots.append(Path(nvcc).resolve().parent.parent)
    roots.append(Path("/usr/local/cuda"))
    candidates = [path for root in roots for path in (root / "include", *sorted(root.glob("targets/*/include")))]
    return next((str(path) for path in candidates if (path / "cuda_fp16.h").is_file()), None)


@functools.cache
def build_options() -> wp.ModuleBuildOptions:
    """Return the build options that make the OptiX builtins available to a module."""
    include_dirs = [str(_HEADER_DIR), native.get_optix_include_dir()]
    if cuda_include_dir := _cuda_include_dir():
        include_dirs.append(cuda_include_dir)
    return wp.ModuleBuildOptions(
        extra_cuda_include_dirs=include_dirs,
        extra_cuda_preamble=_PREAMBLE,
        extra_build_dependencies=[str(_HEADER_DIR / "warp_optix_builtins.h")],
    )


@functools.cache
def register_with_warp() -> None:
    """Register the OptiX builtins with Warp once per process."""
    register_addon_builtins()
