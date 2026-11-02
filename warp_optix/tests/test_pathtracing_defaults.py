# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_SIZE,
    DEFAULT_VIEWER_WIDTH,
    PathTracerAPI,
    PathTracingRenderer,
    PathTracingViewerBackend,
)


def test_high_level_viewers_share_the_full_hd_default():
    assert DEFAULT_VIEWER_SIZE == (DEFAULT_VIEWER_WIDTH, DEFAULT_VIEWER_HEIGHT) == (1920, 1080)
    for constructor in (PathTracerAPI.__init__, PathTracingRenderer.__init__, PathTracingViewerBackend.__init__):
        parameters = inspect.signature(constructor).parameters
        assert parameters["width"].default == DEFAULT_VIEWER_WIDTH
        assert parameters["height"].default == DEFAULT_VIEWER_HEIGHT
        assert parameters["backface_culling"].default is True
