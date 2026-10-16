# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import warp as wp


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail at startup if CUDA is unavailable instead of skipping GPU tests.",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda") and not wp.is_cuda_available():
        raise pytest.UsageError("--require-cuda needs an NVIDIA GPU and a working CUDA driver")
