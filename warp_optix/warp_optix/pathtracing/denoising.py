# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared render guides and a stream-ordered OptiX denoiser for Warp arrays."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import warp as wp

from warp_optix import _ext, native
from warp_optix.context import resolve


@dataclass(frozen=True)
class DenoiserInputs:
    """HDR render buffers, indexed [y, x], shared by the reconstruction backends.

    Normals are world-space float4 (roughness in w); albedo is float4 (RGB
    plus metadata). Motion is float2, current-to-previous in input pixels,
    excluding camera jitter. Optional depth and specular fields describe additional render outputs.
    """

    color: wp.array
    normal: wp.array
    albedo: wp.array
    motion: wp.array
    depth: wp.array | None = None
    specular_albedo: wp.array | None = None
    specular_hit_distance: wp.array | None = None


@wp.kernel(enable_backward=False)
def _prepare_flow(motion: wp.array2d(dtype=wp.vec2), flow: wp.array2d(dtype=wp.vec2)):
    y, x = wp.tid()
    # OptiX expects previous-to-current flow, whereas the renderer uses
    # current-to-previous reprojection offsets.
    flow[y, x] = -motion[y, x]


@wp.kernel(enable_backward=False)
def _crop_output(source: wp.array2d(dtype=wp.vec4), output: wp.array2d(dtype=wp.vec4)):
    y, x = wp.tid()
    output[y, x] = source[y, x]


class OptixDenoiser:
    """Temporal denoising, optionally with 2x upscaling, on one CUDA stream.

    ``device``/``stream`` follow the usual contract: neither given uses the
    current device and its current stream; only ``stream`` uses the stream's
    device; only ``device`` uses that device's current stream; both must match.
    Input arrays are borrowed until work on that stream completes. Output and
    history buffers are owned by this object; call close() to release them early.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        upscale: bool = False,
        device: wp.DeviceLike = None,
        stream: wp.Stream | None = None,
        output_size: tuple[int, int] | None = None,
    ):
        if width < 1 or height < 1:
            raise ValueError("Denoiser input dimensions must be positive")
        context, self.stream = resolve(device, stream)
        self.width, self.height = int(width), int(height)
        scale = 2 if upscale else 1
        self.output_width, self.output_height = self.width * scale, self.height * scale
        target_width, target_height = output_size or (
            self.output_width,
            self.output_height,
        )
        if not (
            self.output_width - scale < target_width <= self.output_width
            and self.output_height - scale < target_height <= self.output_height
        ):
            raise ValueError("output_size must match the denoised size, allowing one cropped pixel for 2x upscaling")
        self._denoiser = None
        self._history_valid = False
        self._bindings = [None, None]
        self._parity = 0
        self._stream_handle = int(self.stream.cuda_stream)
        model_name = "DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X" if upscale else "DENOISER_MODEL_KIND_TEMPORAL_AOV"
        if not hasattr(native, model_name):
            raise RuntimeError("OptiX temporal denoising requires updated PyOptiX denoiser bindings")
        options = native.DenoiserOptions()
        options.guideAlbedo = 1
        options.guideNormal = 1
        device = self.stream.device
        with wp.ScopedStream(self.stream):
            try:
                self._denoiser = context.native.denoiserCreate(getattr(native, model_name), options)
                sizes = self._denoiser.computeMemoryResources(self.width, self.height)
                average_color_size, guide_pixel_size = _ext.denoiser_temporal_sizes(sizes)
                self._state = wp.empty(sizes.stateSizeInBytes, dtype=wp.uint8, device=device)
                self._scratch_size = max(sizes.withoutOverlapScratchSizeInBytes, average_color_size)
                # 2x upscaling from 1080p needs over 2 GiB of scratch, more elements than a uint8 array can hold.
                self._scratch = wp.empty((self._scratch_size + 7) // 8, dtype=wp.uint64, device=device)
                self._average_color = wp.empty(3, dtype=wp.float32, device=device)
                self._params = native.DenoiserParams()
                self._params.hdrAverageColor = self._average_color.ptr
                self._flow = wp.empty((self.height, self.width), dtype=wp.vec2, device=device)
                shape = (self.output_height, self.output_width)
                self._output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._previous_output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._cropped_output = (
                    wp.empty((target_height, target_width), dtype=wp.vec4, device=device)
                    if (target_height, target_width) != shape
                    else None
                )
                self._guide_pixel_size = int(guide_pixel_size)
                guide_size = self.output_width * self.output_height * self._guide_pixel_size
                self._previous_guide = wp.zeros(guide_size, dtype=wp.uint8, device=device)
                self._output_guide = wp.zeros(guide_size, dtype=wp.uint8, device=device)
                self._denoiser.setup(
                    int(self.stream.cuda_stream),
                    self.width,
                    self.height,
                    self._state.ptr,
                    self._state.size,
                    self._scratch.ptr,
                    self._scratch_size,
                )
            except Exception:
                self.close()
                raise

    def _image(self, array, pixel_format, *, pixel_stride=None, width=None, height=None):
        image = native.Image2D()
        image.data = array.ptr
        image.width = self.width if width is None else width
        image.height = self.height if height is None else height
        image.pixelStrideInBytes = array.strides[1] if pixel_stride is None else pixel_stride
        image.rowStrideInBytes = array.strides[0] if pixel_stride is None else image.width * pixel_stride
        image.format = pixel_format
        return image

    def _bind(self, inputs: DenoiserInputs):
        for name, dtype in (
            ("color", wp.vec4),
            ("normal", wp.vec4),
            ("albedo", wp.vec4),
            ("motion", wp.vec2),
        ):
            array = getattr(inputs, name)
            if array.shape != (self.height, self.width) or array.dtype != dtype:
                raise ValueError(f"{name} must be a {dtype.__name__} array of shape {(self.height, self.width)}")
            if array.device != self.stream.device or not array.is_contiguous:
                raise ValueError(f"{name} must be contiguous on {self.stream.device}")
        guides = native.DenoiserGuideLayer()
        guides.albedo = self._image(inputs.albedo, native.PIXEL_FORMAT_FLOAT3)
        guides.normal = self._image(inputs.normal, native.PIXEL_FORMAT_FLOAT3)
        guides.flow = self._image(self._flow, native.PIXEL_FORMAT_FLOAT2)
        _ext.set_internal_guide_layers(
            guides,
            *(
                self._image(
                    buffer,
                    native.PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,
                    pixel_stride=self._guide_pixel_size,
                    width=self.output_width,
                    height=self.output_height,
                )
                for buffer in (self._previous_guide, self._output_guide)
            ),
        )
        layer = native.DenoiserLayer()
        layer.input = self._image(inputs.color, native.PIXEL_FORMAT_FLOAT4)
        layer.output = self._image(
            self._output,
            native.PIXEL_FORMAT_FLOAT4,
            width=self.output_width,
            height=self.output_height,
        )
        layer.previousOutput = self._image(
            self._previous_output,
            native.PIXEL_FORMAT_FLOAT4,
            width=self.output_width,
            height=self.output_height,
        )
        flow = wp.launch(
            _prepare_flow,
            dim=(self.height, self.width),
            inputs=[inputs.motion, self._flow],
            device=self.stream.device,
            record_cmd=True,
        )
        crop = None
        if self._cropped_output is not None:
            crop = wp.launch(
                _crop_output,
                dim=self._cropped_output.shape,
                inputs=[self._output, self._cropped_output],
                device=self.stream.device,
                record_cmd=True,
            )
        key = (inputs.color, inputs.normal, inputs.albedo, inputs.motion)
        return key, guides, layer, flow, crop

    def apply(self, inputs: DenoiserInputs, *, reset=False):
        """Queue denoising and return the owned float4 Warp output array."""
        if self._denoiser is None:
            raise RuntimeError("OptiX denoiser is closed")
        # Recorded launches and native OptiX calls may change the current CUDA context.
        # Preserve the caller's context without adding stream synchronization on every frame.
        with wp.ScopedDevice(self.stream.device):
            return self._apply(inputs, reset=reset)

    def _apply(self, inputs: DenoiserInputs, *, reset):
        # Output and history buffers swap every call, so each parity keeps its own native bindings.
        binding = self._bindings[self._parity]
        if (
            binding is None
            or binding[0][0] is not inputs.color
            or binding[0][1] is not inputs.normal
            or binding[0][2] is not inputs.albedo
            or binding[0][3] is not inputs.motion
        ):
            binding = self._bindings[self._parity] = self._bind(inputs)
        _, guides, layer, flow, crop = binding
        stream = self._stream_handle
        if reset or not self._history_valid:
            with wp.ScopedStream(self.stream):
                self._previous_guide.zero_()
                self._previous_output.zero_()
            self._history_valid = False
        flow.launch(self.stream)
        _ext.set_temporal_mode_use_previous_layers(self._params, int(self._history_valid))
        self._denoiser.computeAverageColor(
            stream,
            layer.input,
            self._average_color.ptr,
            self._scratch.ptr,
            self._scratch_size,
        )
        self._denoiser.invoke(
            stream,
            self._params,
            self._state.ptr,
            self._state.size,
            guides,
            layer,
            1,
            0,
            0,
            self._scratch.ptr,
            self._scratch_size,
        )
        result = self._output
        self._output, self._previous_output = self._previous_output, self._output
        self._output_guide, self._previous_guide = self._previous_guide, self._output_guide
        self._parity ^= 1
        self._history_valid = True
        if crop is not None:
            crop.launch(self.stream)
            result = self._cropped_output
        return result

    def close(self):
        """Finish queued work and destroy the native denoiser."""
        if self._denoiser is not None:
            wp.synchronize_stream(self.stream)
            self._denoiser.destroy()
            self._denoiser = None
            self._history_valid = False
            self._bindings = [None, None]
            for name in (
                "_state",
                "_scratch",
                "_average_color",
                "_flow",
                "_output",
                "_previous_output",
                "_cropped_output",
                "_previous_guide",
                "_output_guide",
            ):
                setattr(self, name, None)

    def __del__(self):
        with suppress(Exception):
            self.close()
