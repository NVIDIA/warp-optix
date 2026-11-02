# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced access to PyOptiX plus warp-optix's supplemental bindings.

Native objects use the binding API's explicit destroy/deinit conventions.
Handles borrowed from the owned API must not be destroyed by the caller.
"""

from ._backend import get_backend as _get_backend


def __getattr__(name):
    return getattr(_get_backend(), name)


def __dir__():
    return sorted(name for name in dir(_get_backend()) if not name.startswith("_"))
