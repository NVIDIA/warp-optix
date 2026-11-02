# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time SDK checks also apply to user-supplied OptiX headers."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "header_text, expected_error",
    [
        ("#define OPTIX_VERSION 90100\n", None),
        ("#define OPTIX_VERSION 90000\n", "headers are 9.0.0"),
        ("#define OPTIX_VERSION 90101\n", "headers are 9.1.1"),
        ("#define OPTIX_VERSION 100100\n", "headers are 10.1.0"),
        ("// no SDK version\n", "Cannot determine OptiX SDK version"),
        (None, "OptiX SDK header is missing"),
    ],
)
def test_build_rejects_incompatible_sdk_headers(header_text, expected_error, tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to exercise the build-time SDK check")
    checker = Path(__file__).resolve().parents[2] / "cmake" / "CheckOptixVersion.cmake"
    header = tmp_path / "optix.h"
    if header_text is not None:
        header.write_text(header_text)
    script = tmp_path / "check.cmake"
    script.write_text(
        f'include("{checker.as_posix()}")\n'
        f'warp_optix_check_optix_version("{header.as_posix()}" "9.1.0")\n'
    )
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    if expected_error is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert expected_error in result.stderr
