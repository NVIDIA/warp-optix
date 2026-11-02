# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth-aware, CUDA-updatable OpenGL line overlays for OptiX images."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

import numpy as np
import warp as wp


@wp.struct
class LineVertex:
    position: wp.vec3
    color: wp.vec3


@wp.kernel
def _write_line_vertices(
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=LineVertex),
):
    line = wp.tid()
    vertex = 2 * line
    vertices[vertex].position = starts[line]
    vertices[vertex].color = colors[line]
    vertices[vertex + 1].position = ends[line]
    vertices[vertex + 1].color = colors[line]


@wp.kernel
def _write_uniform_line_vertices(
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    color: wp.vec3,
    vertices: wp.array(dtype=LineVertex),
):
    line = wp.tid()
    vertex = 2 * line
    vertices[vertex].position = starts[line]
    vertices[vertex].color = color
    vertices[vertex + 1].position = ends[line]
    vertices[vertex + 1].color = color


_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertex_color;

void main()
{
    vertex_color = a_color;
    gl_Position = projection * view * vec4(a_position, 1.0);
}
"""


# This follows Newton's wireframe technique: expand GL_LINES into screen-space
# triangles so line width is stable and independent of deprecated GL line width.
_GEOMETRY_SHADER = """
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vertex_color[2];
out vec3 line_color;
noperspective out float line_distance;

uniform float inverse_aspect;
uniform float line_width;
uniform float antialias_width;

void main()
{
    vec4 start_clip = gl_in[0].gl_Position;
    vec4 end_clip = gl_in[1].gl_Position;
    if (start_clip.w <= 0.0 || end_clip.w <= 0.0)
        return;

    vec2 start_ndc = start_clip.xy / start_clip.w;
    vec2 end_ndc = end_clip.xy / end_clip.w;
    vec2 direction_ndc = end_ndc - start_ndc;
    float safe_aspect = max(inverse_aspect, 1.0e-6);
    vec2 direction_screen = vec2(direction_ndc.x / safe_aspect, direction_ndc.y);
    float direction_length = length(direction_screen);
    if (direction_length < 1.0e-8)
        return;

    vec2 right_screen = vec2(direction_screen.y, -direction_screen.x) / direction_length;
    vec2 right_ndc = vec2(right_screen.x * safe_aspect, right_screen.y);
    float outer_width = 0.5 * line_width + antialias_width;
    vec2 offset = outer_width * right_ndc;
    float start_depth = 2.0 * start_clip.z / start_clip.w - 1.0;
    float end_depth = 2.0 * end_clip.z / end_clip.w - 1.0;
    line_color = 0.5 * (vertex_color[0] + vertex_color[1]);

    line_distance = -outer_width;
    gl_Position = vec4(start_ndc - offset, start_depth, 1.0); EmitVertex();
    gl_Position = vec4(end_ndc - offset, end_depth, 1.0); EmitVertex();
    line_distance = outer_width;
    gl_Position = vec4(start_ndc + offset, start_depth, 1.0); EmitVertex();
    gl_Position = vec4(end_ndc + offset, end_depth, 1.0); EmitVertex();
    EndPrimitive();
}
"""


_FRAGMENT_SHADER = """
#version 330 core
in vec3 line_color;
noperspective in float line_distance;
out vec4 fragment_color;

uniform sampler2D scene_depth;
uniform vec2 viewport_size;
uniform bool use_scene_depth;
uniform float camera_near;
uniform float camera_far;
uniform float depth_bias;
uniform float alpha;
uniform float line_width;
uniform float antialias_width;

void main()
{
    vec2 uv = gl_FragCoord.xy / max(viewport_size, vec2(1.0));
    uv.y = 1.0 - uv.y;
    float linear_depth = texture(scene_depth, uv).r;
    if (use_scene_depth && linear_depth > 0.0) {
        float scene_window_depth = camera_far / (camera_far - camera_near)
            - (camera_far * camera_near)
                / ((camera_far - camera_near) * linear_depth);
        if (gl_FragCoord.z > scene_window_depth + depth_bias)
            discard;
    }
    float coverage = 1.0;
    if (antialias_width > 0.0) {
        float half_width = 0.5 * line_width;
        coverage = 1.0 - smoothstep(
            half_width, half_width + antialias_width, abs(line_distance));
    }
    float output_alpha = alpha * coverage;
    if (output_alpha <= 0.0)
        discard;
    fragment_color = vec4(line_color, output_alpha);
}
"""


def _float_pointer(values: np.ndarray):
    return np.ascontiguousarray(values, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _opengl_projection(projection: np.ndarray, camera_near: float, camera_far: float) -> np.ndarray:
    """Convert path-tracer projection to OpenGL's presentation convention."""
    near = float(camera_near)
    far = float(camera_far)
    if near <= 0.0 or far <= near:
        raise ValueError("camera_far must be greater than positive camera_near")
    result = np.array(projection, dtype=np.float32, copy=True).reshape(4, 4)
    # Camera projection is vertically flipped for the OptiX/Vulkan image. The
    # displayed image has already been oriented by the presentation pass.
    result[1, 1] = abs(result[1, 1])
    result[2, 2] = far / (near - far)
    result[2, 3] = -1.0
    result[3, 2] = -(near * far) / (far - near)
    result[3, 3] = 0.0
    return result


class GLLineOverlay:
    """Efficient depth-tested line batch composited over an OptiX image.

    The batch owns a fixed-capacity OpenGL VBO registered with CUDA. Device
    updates write directly into mapped GL memory, so changing line positions,
    ordering, and active count does not rebuild an acceleration structure.
    OptiX positive view-space depth is uploaded to an R32F texture for fragment
    occlusion. The OpenGL context must be current when constructing and drawing.
    """

    def __init__(
        self,
        gl,
        capacity: int,
        *,
        device: str | wp.context.Device = "cuda",
        depth_buffer: wp.array | None = None,
        line_width: float = 1.5,
        antialias_width: float = 1.0,
        alpha: float = 1.0,
        depth_bias: float = 2.0e-6,
        use_depth_test: bool = True,
        fallback_to_copy: bool = True,
        stream: wp.Stream | None = None,
    ):
        if int(capacity) < 1:
            raise ValueError("capacity must be positive")
        if float(line_width) <= 0.0:
            raise ValueError("line_width must be positive")
        if float(antialias_width) < 0.0:
            raise ValueError("antialias_width must be nonnegative")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be between zero and one")

        from pyglet.graphics.shader import Shader, ShaderProgram

        self.gl = gl
        self.capacity = int(capacity)
        self.device = wp.get_device(device)
        self.stream = stream or wp.get_stream(self.device)
        self.line_width = float(line_width)
        self.antialias_width = float(antialias_width)
        self.alpha = float(alpha)
        self.depth_bias = float(depth_bias)
        self.active_count = 0
        self.depth_buffer = None
        self.use_depth_test = bool(use_depth_test)
        self._depth_shape = None
        self._depth_texture = None
        self._depth_pbo = None
        self._depth_cuda_gl = None
        self._destroyed = False
        self._fallback_to_copy = bool(fallback_to_copy)

        self._program = ShaderProgram(
            Shader(_VERTEX_SHADER, "vertex"),
            Shader(_GEOMETRY_SHADER, "geometry"),
            Shader(_FRAGMENT_SHADER, "fragment"),
        )
        self._uniforms = {
            name: gl.glGetUniformLocation(self._program.id, name.encode())
            for name in (
                "view",
                "projection",
                "inverse_aspect",
                "line_width",
                "antialias_width",
                "scene_depth",
                "viewport_size",
                "camera_near",
                "camera_far",
                "depth_bias",
                "alpha",
                "use_scene_depth",
            )
        }

        self._vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._vao)
        gl.glBindVertexArray(self._vao)
        self._vbo = gl.GLuint()
        gl.glGenBuffers(1, self._vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.capacity * 2 * 24,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)
        self._vertex_cuda_gl = wp.RegisteredGLBuffer(
            int(self._vbo.value),
            device=self.device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=self._fallback_to_copy,
        )
        if depth_buffer is not None:
            self.set_depth_buffer(depth_buffer)

    def set_depth_buffer(self, depth_buffer: wp.array) -> None:
        """Select the positive view-space OptiX depth buffer used for occlusion."""
        if depth_buffer is None or depth_buffer.ndim != 2:
            raise ValueError("depth_buffer must be a two-dimensional Warp array")
        if depth_buffer.dtype != wp.float32:
            raise ValueError("depth_buffer must have dtype wp.float32")
        if depth_buffer.device != self.device:
            raise ValueError("depth_buffer must be on the overlay device")
        self.depth_buffer = depth_buffer

    def update_device(
        self,
        starts: wp.array,
        ends: wp.array,
        colors: wp.array | Sequence[float] = (1.0, 0.55, 0.05),
        *,
        count: int | None = None,
    ) -> None:
        """Write CUDA-resident lines directly into the mapped OpenGL VBO."""
        if starts.dtype != wp.vec3 or ends.dtype != wp.vec3:
            raise ValueError("starts and ends must have dtype wp.vec3")
        if starts.device != self.device or ends.device != self.device:
            raise ValueError("starts and ends must be on the overlay device")
        if count is None and len(starts) != len(ends):
            raise ValueError("starts and ends must contain the same number of lines")
        line_count = len(starts) if count is None else int(count)
        if line_count < 0 or line_count > self.capacity:
            raise ValueError("count must be between zero and capacity")
        if line_count > len(starts) or line_count > len(ends):
            raise ValueError("count exceeds the starts or ends array length")
        per_line_colors = hasattr(colors, "dtype")
        if per_line_colors:
            if colors.dtype != wp.vec3 or colors.device != self.device:
                raise ValueError("colors must be a wp.vec3 array on the overlay device")
            if line_count > len(colors):
                raise ValueError("count exceeds the colors array length")
        else:
            color = tuple(float(value) for value in colors)
            if len(color) != 3:
                raise ValueError("uniform color must contain three values")

        self.active_count = line_count
        if line_count == 0:
            return
        with wp.ScopedDevice(self.device):
            with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False):
                vertices = self._vertex_cuda_gl.map(dtype=LineVertex, shape=(2 * self.capacity,))
                try:
                    if per_line_colors:
                        wp.launch(
                            _write_line_vertices,
                            dim=line_count,
                            inputs=[starts, ends, colors],
                            outputs=[vertices],
                            device=self.device,
                        )
                    else:
                        wp.launch(
                            _write_uniform_line_vertices,
                            dim=line_count,
                            inputs=[starts, ends, wp.vec3(*color)],
                            outputs=[vertices],
                            device=self.device,
                        )
                finally:
                    self._vertex_cuda_gl.unmap()
                wp.synchronize_stream(self.stream)

    def update(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        colors: np.ndarray | Sequence[float] = (1.0, 0.55, 0.05),
    ) -> None:
        """Upload host arrays and update the active line batch."""
        starts = np.ascontiguousarray(starts, dtype=np.float32).reshape(-1, 3)
        ends = np.ascontiguousarray(ends, dtype=np.float32).reshape(-1, 3)
        if len(starts) != len(ends):
            raise ValueError("starts and ends must contain the same number of lines")
        device_starts = wp.array(starts, dtype=wp.vec3, device=self.device)
        device_ends = wp.array(ends, dtype=wp.vec3, device=self.device)
        if np.asarray(colors).ndim == 1:
            self.update_device(device_starts, device_ends, colors)
        else:
            colors = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)
            if len(colors) != len(starts):
                raise ValueError("colors must contain one value per line")
            device_colors = wp.array(colors, dtype=wp.vec3, device=self.device)
            self.update_device(device_starts, device_ends, device_colors)

    def _ensure_depth_resources(self) -> None:
        height, width = (int(value) for value in self.depth_buffer.shape)
        if self._depth_shape == (height, width):
            return
        self._destroy_depth_resources()
        gl = self.gl
        self._depth_texture = gl.GLuint()
        gl.glGenTextures(1, self._depth_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_R32F,
            width,
            height,
            0,
            gl.GL_RED,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._depth_pbo = gl.GLuint()
        gl.glGenBuffers(1, self._depth_pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._depth_pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self._depth_cuda_gl = wp.RegisteredGLBuffer(
            int(self._depth_pbo.value),
            device=self.device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=self._fallback_to_copy,
        )
        self._depth_shape = (height, width)

    def _upload_depth(self) -> None:
        self._ensure_depth_resources()
        height, width = self._depth_shape
        with wp.ScopedDevice(self.device):
            with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False):
                mapped = self._depth_cuda_gl.map(dtype=wp.float32, shape=(height * width,))
                try:
                    wp.copy(mapped, self.depth_buffer.flatten())
                finally:
                    self._depth_cuda_gl.unmap()
                wp.synchronize_stream(self.stream)
        gl = self.gl
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._depth_pbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl.GL_RED,
            gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    def draw(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        viewport_size: tuple[int, int],
        *,
        camera_near: float,
        camera_far: float,
    ) -> None:
        """Composite current lines using the OptiX depth buffer for occlusion."""
        if self._destroyed or self.active_count == 0 or self.depth_buffer is None:
            return
        width, height = int(viewport_size[0]), int(viewport_size[1])
        if width < 1 or height < 1:
            return
        self._upload_depth()
        view_matrix = np.ascontiguousarray(view_matrix, dtype=np.float32)
        projection_matrix = _opengl_projection(projection_matrix, camera_near, camera_far)
        gl = self.gl
        gl.glUseProgram(self._program.id)
        gl.glUniform1i(self._uniforms["use_scene_depth"], self.use_depth_test)
        gl.glUniformMatrix4fv(self._uniforms["view"], 1, gl.GL_FALSE, _float_pointer(view_matrix))
        gl.glUniformMatrix4fv(
            self._uniforms["projection"],
            1,
            gl.GL_FALSE,
            _float_pointer(projection_matrix),
        )
        gl.glUniform1f(self._uniforms["inverse_aspect"], height / max(width, 1))
        gl.glUniform1f(self._uniforms["line_width"], self.line_width * 2.0 / height)
        gl.glUniform1f(
            self._uniforms["antialias_width"],
            self.antialias_width * 2.0 / height,
        )
        gl.glUniform2f(self._uniforms["viewport_size"], width, height)
        gl.glUniform1f(self._uniforms["camera_near"], float(camera_near))
        gl.glUniform1f(self._uniforms["camera_far"], float(camera_far))
        gl.glUniform1f(self._uniforms["depth_bias"], self.depth_bias)
        gl.glUniform1f(self._uniforms["alpha"], self.alpha)
        gl.glUniform1i(self._uniforms["scene_depth"], 7)
        gl.glActiveTexture(gl.GL_TEXTURE7)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_texture)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_LINES, 0, 2 * self.active_count)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUseProgram(0)

    def _destroy_depth_resources(self) -> None:
        self._depth_cuda_gl = None
        if self._depth_pbo is not None:
            self.gl.glDeleteBuffers(1, self._depth_pbo)
        if self._depth_texture is not None:
            self.gl.glDeleteTextures(1, self._depth_texture)
        self._depth_pbo = None
        self._depth_texture = None
        self._depth_shape = None

    def destroy(self) -> None:
        """Release CUDA registrations and OpenGL objects while context is current."""
        if self._destroyed:
            return
        self._destroyed = True
        self._vertex_cuda_gl = None
        self._destroy_depth_resources()
        self.gl.glDeleteBuffers(1, self._vbo)
        self.gl.glDeleteVertexArrays(1, self._vao)


__all__ = ["GLLineOverlay", "LineVertex"]
