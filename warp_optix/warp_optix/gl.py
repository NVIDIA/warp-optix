# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenGL display helpers for interactive programs (requires pyglet).

:class:`GLInteropViewer` shows a CUDA-mapped RGBA pixel buffer each frame without a host copy, and
:class:`GLLineOverlay` draws depth-tested lines over it.
"""

from warp_optix._src.gl.interop import GLInteropViewer as GLInteropViewer
from warp_optix._src.gl.line_overlay import GLLineOverlay as GLLineOverlay

__all__ = ["GLInteropViewer", "GLLineOverlay"]
