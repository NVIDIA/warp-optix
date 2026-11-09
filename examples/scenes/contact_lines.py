# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Changing depth-tested OpenGL contact lines drawn over a path-traced frame."""

from __future__ import annotations

import numpy as np
import warp as wp
from warp_optix.gl import GLLineOverlay

from .arrowball import generate_arrow_ball
from .style import RAINBOW_SRGB


@wp.kernel
def _animate_contacts(
    base_starts: wp.array(dtype=wp.vec3),
    base_ends: wp.array(dtype=wp.vec3),
    base_colors: wp.array(dtype=wp.vec3),
    phase: wp.float32,
    permutation_offset: wp.int32,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    line = wp.tid()
    count = base_starts.shape[0]
    # An odd multiplier is coprime to power-of-two counts and shuffles the order as the offset changes.
    source = (line * 8191 + permutation_offset) % count
    start = base_starts[source]
    direction = base_ends[source] - start
    pulse = 0.82 + 0.18 * wp.sin(phase + wp.float32(source % 97) * 0.071)
    starts[line] = start
    ends[line] = start + direction * pulse
    colors[line] = base_colors[source]


class ContactLines:
    """Up to `count` lines whose number and order change every frame."""

    def __init__(self, api, count: int, ball_radius: float = 0.8, line_length: float = 0.32):
        if count < 1:
            raise ValueError("count must be positive")
        self.api = api
        self.count = count
        starts, ends, slots = generate_arrow_ball(count, ball_radius, line_length, swirl=0.14, seed=29)
        self.base_starts = wp.array(starts, dtype=wp.vec3, device="cuda")
        self.base_ends = wp.array(ends, dtype=wp.vec3, device="cuda")
        self.base_colors = wp.array(RAINBOW_SRGB[slots], dtype=wp.vec3, device="cuda")
        self.starts = wp.empty_like(self.base_starts)
        self.ends = wp.empty_like(self.base_ends)
        self.colors = wp.empty_like(self.base_colors)
        self.overlay = None

    def attach(self, viewer):
        self.viewer = viewer
        self.overlay = GLLineOverlay(
            viewer.gl,
            self.count,
            device="cuda",
            depth_buffer=self.api.linear_depth_output,
            line_width=1.5,
            antialias_width=1.0,
            stream=viewer.render_stream,
        )
        viewer.set_draw_overlay(self.draw)

    def resize(self):
        self.overlay.set_depth_buffer(self.api.linear_depth_output)

    def update(self, frame: int, elapsed: float):
        active_count = max(1, int(self.count * (0.75 + 0.25 * (0.5 + 0.5 * np.sin(elapsed)))))
        wp.launch(
            _animate_contacts,
            dim=active_count,
            inputs=[self.base_starts, self.base_ends, self.base_colors, elapsed * 2.0, (frame * 8191) % self.count],
            outputs=[self.starts, self.ends, self.colors],
            device="cuda",
        )
        self.overlay.update_device(self.starts, self.ends, self.colors, count=active_count)

    def draw(self):
        camera = self.api.viewer.camera
        self.overlay.draw(
            camera.get_view_matrix(),
            camera.get_projection_matrix(),
            (self.viewer.width, self.viewer.height),
            camera_near=camera.near,
            camera_far=camera.far,
        )

    def close(self):
        if self.overlay is not None:
            self.overlay.close()
