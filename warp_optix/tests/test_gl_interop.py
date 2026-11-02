# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import warp as wp
from warp_optix.gl import GLInteropViewer


def test_fps_caption_updates_at_requested_interval():
    captions = []
    viewer = GLInteropViewer.__new__(GLInteropViewer)
    viewer._base_title = "Example"
    viewer._show_fps = True
    viewer._fps_update_interval = 0.5
    viewer._fps_sample_time = 10.0
    viewer._fps_sample_frame = 4
    viewer.frame_index = 14
    viewer.window = SimpleNamespace(set_caption=captions.append)

    viewer._update_fps_caption(10.49)
    assert captions == []

    viewer._update_fps_caption(10.5)
    assert captions == ["Example — 20.0 FPS"]
    assert viewer._fps_sample_time == 10.5
    assert viewer._fps_sample_frame == 14


def test_render_stream_completes_before_gl_consumes_pbo():
    events = []

    class _CudaGL:
        def map(self, *, dtype, shape):
            del dtype, shape
            events.append("map")
            return object()

        def unmap(self):
            events.append("unmap")

    viewer = GLInteropViewer.__new__(GLInteropViewer)
    viewer.device = "cuda:0"
    viewer.render_stream = object()
    viewer.cuda_gl = _CudaGL()
    viewer.width = 8
    viewer.height = 4
    viewer.frame_index = 3
    viewer.start_time = time.perf_counter()
    viewer._show_fps = False
    viewer.max_frames = 0
    viewer._render_callback = lambda mapped, frame, elapsed: events.append(("render", mapped, frame, elapsed))
    viewer.window = SimpleNamespace(
        switch_to=lambda: events.append("switch"),
        flip=lambda: events.append("flip"),
    )
    viewer._on_draw = lambda: events.append("draw")

    with (
        mock.patch.object(wp, "ScopedDevice", side_effect=lambda _device: nullcontext()),
        mock.patch.object(
            wp,
            "ScopedStream",
            side_effect=lambda _stream, **_kwargs: nullcontext(),
        ),
        mock.patch.object(wp, "synchronize_stream", side_effect=lambda _stream: events.append("sync")),
    ):
        viewer._render_frame()

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "map",
        "render",
        "unmap",
        "sync",
        "switch",
        "draw",
        "flip",
    ]


def test_render_callback_failure_unmaps_pbo():
    events = []
    viewer = GLInteropViewer.__new__(GLInteropViewer)
    viewer.device = "cuda:0"
    viewer.render_stream = object()
    viewer.cuda_gl = SimpleNamespace(
        map=lambda **_kwargs: events.append("map"),
        unmap=lambda: events.append("unmap"),
    )
    viewer.width = viewer.height = 1
    viewer.frame_index = 0
    viewer.start_time = time.perf_counter()

    def fail(*_args):
        raise RuntimeError("render failed")

    viewer._render_callback = fail
    with (
        mock.patch.object(wp, "ScopedDevice", side_effect=lambda _device: nullcontext()),
        mock.patch.object(wp, "ScopedStream", side_effect=lambda _stream, **_kwargs: nullcontext()),
        pytest.raises(RuntimeError, match="render failed"),
    ):
        viewer._render_frame()

    assert events == ["map", "unmap"]
