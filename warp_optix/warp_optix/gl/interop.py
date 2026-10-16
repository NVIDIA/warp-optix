# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal OpenGL viewer with CUDA zero-copy PBO interop."""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable

import warp as wp


class OptixGLInteropViewer:
    @staticmethod
    def _create_texture_2d_compat(pyglet_module, gl_module, width: int, height: int):
        try:
            # pyglet 2.x style (legacy rectangle arg still accepted on some versions)
            return pyglet_module.image.Texture.create(width=width, height=height, rectangle=False)
        except TypeError:
            try:
                # pyglet compatibility path used by the working viewer implementation
                return pyglet_module.image.Texture.create(width=width, height=height, target=gl_module.GL_TEXTURE_2D)
            except TypeError:
                # fallback for older pyglet variants
                return pyglet_module.image.Texture.create(width=width, height=height)

    def __init__(
        self,
        width: int,
        height: int,
        device: str,
        title: str = "Warp OptiX Tiny Raytracer",
        fps: int = 0,
        on_resize: Callable[[int, int], None] | None = None,
        on_draw_overlay: Callable[[], None] | None = None,
        vsync: bool = False,
        fallback_to_copy: bool = True,
        render_stream: wp.Stream | None = None,
        show_fps: bool = True,
        fps_update_interval: float = 0.5,
    ):
        import pyglet
        from pyglet import gl

        self.width = width
        self.height = height
        self.device = device
        self.render_stream = render_stream or wp.get_stream(device)
        self.pyglet = pyglet
        self.gl = gl
        self.frame_index = 0
        self.start_time = time.perf_counter()
        self._base_title = str(title)
        self._show_fps = bool(show_fps)
        self._fps_update_interval = max(float(fps_update_interval), 0.05)
        self._fps_sample_time = self.start_time
        self._fps_sample_frame = 0
        self.max_frames = 0
        self.closed = False
        self._update_scheduled = False
        self._update_interval = None if fps <= 0 else 1.0 / float(fps)
        self._render_callback: Callable[[wp.array, int, float], None] | None = None
        self._on_resize_callback = on_resize
        self._on_draw_overlay = on_draw_overlay
        self._dispatching_events = False
        self._fallback_to_copy = bool(fallback_to_copy)

        self.window = pyglet.window.Window(width=width, height=height, caption=title, vsync=bool(vsync), resizable=True)
        self.window.push_handlers(on_draw=self._on_draw, on_close=self._on_close, on_resize=self._on_resize)
        self._recreate_gl_resources()

    def _recreate_gl_resources(self):
        # Recreate texture + PBO + CUDA interop for current resolution.
        gl = self.gl

        self.cuda_gl = None
        if hasattr(self, "pbo") and self.pbo is not None:
            gl.glDeleteBuffers(1, self.pbo)

        self.texture = self._create_texture_2d_compat(self.pyglet, gl, self.width, self.height)
        self.texture.min_filter = gl.GL_NEAREST
        self.texture.mag_filter = gl.GL_NEAREST
        self.sprite = self.pyglet.sprite.Sprite(self.texture, x=0, y=0)

        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            self.width * self.height * 4,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.cuda_gl = wp.RegisteredGLBuffer(
            int(self.pbo.value),
            device=self.device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=self._fallback_to_copy,
        )
        gl.glViewport(0, 0, self.width, self.height)

    def run(
        self,
        render_callback: Callable[[wp.array, int, float], None],
        max_frames: int = 0,
    ):
        self._render_callback = render_callback
        self.max_frames = max_frames
        if self._update_interval is None:
            self.pyglet.clock.schedule(self._update)
        else:
            self.pyglet.clock.schedule_interval(self._update, self._update_interval)
        self._update_scheduled = True
        try:
            self.pyglet.app.run()
        finally:
            self.pyglet.clock.unschedule(self._update)
            self._update_scheduled = False

    def render_once(self, render_callback: Callable[[wp.array, int, float], None] | None = None):
        """Render and present one frame without entering pyglet's event loop."""
        if self.closed:
            return
        if render_callback is not None:
            self._render_callback = render_callback
        self._dispatching_events = True
        try:
            self.window.dispatch_events()
        finally:
            self._dispatching_events = False
        if self.closed:
            return
        self._render_frame()

    def set_draw_overlay(self, callback: Callable[[], None] | None) -> None:
        """Set a callback drawn after the CUDA-produced image is presented."""
        self._on_draw_overlay = callback

    def _update(self, _dt):
        self._render_frame()

    def _render_frame(self):
        if self._render_callback is None:
            return
        with wp.ScopedDevice(self.device):
            with wp.ScopedStream(self.render_stream, sync_enter=False, sync_exit=False):
                mapped = self.cuda_gl.map(dtype=wp.uint32, shape=(self.width * self.height,))
                try:
                    elapsed = time.perf_counter() - self.start_time
                    self._render_callback(mapped, self.frame_index, elapsed)
                finally:
                    self.cuda_gl.unmap()
                # Unmapping is asynchronous. Drain only the render stream so
                # CUDA releases PBO ownership before OpenGL consumes it below.
                wp.synchronize_stream(self.render_stream)

        # Drive presentation explicitly each update so rendering does not depend
        # on backend-specific on_draw invalidation behavior.
        self.window.switch_to()
        self._on_draw()
        self.window.flip()

        self.frame_index += 1
        self._update_fps_caption(time.perf_counter())
        if self.max_frames > 0 and self.frame_index >= self.max_frames:
            self.pyglet.app.exit()

    def _update_fps_caption(self, now: float):
        if not self._show_fps:
            return
        elapsed = float(now) - self._fps_sample_time
        if elapsed < self._fps_update_interval:
            return
        fps = float(self.frame_index - self._fps_sample_frame) / elapsed
        self.window.set_caption(f"{self._base_title} — {fps:.1f} FPS")
        self._fps_sample_time = float(now)
        self._fps_sample_frame = self.frame_index

    def _on_draw(self):
        gl = self.gl
        self.window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glTexSubImage2D(
            self.texture.target,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(self.texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.sprite.draw()
        if self._on_draw_overlay is not None and not self._dispatching_events:
            self._on_draw_overlay()

    def _on_close(self):
        if self.closed:
            return
        self.closed = True
        if self.cuda_gl is not None:
            self.cuda_gl = None
        if self.pbo is not None:
            self.gl.glDeleteBuffers(1, self.pbo)
            self.pbo = None
        self.pyglet.app.exit()

    def close(self):
        """Release presentation resources and close the window."""
        if self.closed:
            return
        self._on_close()
        if self._update_scheduled:
            self.pyglet.clock.unschedule(self._update)
            self._update_scheduled = False
        self.window.close()

    def is_running(self) -> bool:
        """Return whether the presentation window remains open."""
        return not self.closed and not self.window.has_exit

    def _on_resize(self, width: int, height: int):
        if width <= 0 or height <= 0:
            return

        self.width = int(width)
        self.height = int(height)
        self._recreate_gl_resources()

        if self._on_resize_callback is not None:
            self._on_resize_callback(self.width, self.height)
