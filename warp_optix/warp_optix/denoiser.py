# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream-ordered OptiX temporal denoiser for Warp arrays."""

from __future__ import annotations

import warp as wp

from warp_optix import native
from warp_optix.context import resolve


@wp.kernel(enable_backward=False)
def _prepare_flow(motion: wp.array2d(dtype=wp.vec2), flow: wp.array2d(dtype=wp.vec2)):
    y, x = wp.tid()
    # OptiX expects previous-to-current flow; motion is current-to-previous.
    flow[y, x] = -motion[y, x]


class Denoiser:
    """OptiX temporal denoising of HDR images, optionally with 2x upscaling.

    Denoises ``height`` x ``width`` inputs into float4 images of shape
    ``(output_height, output_width)``, twice the input size with ``upscale=True``.
    All work runs on one stream, chosen like other constructors: neither
    ``device`` nor ``stream`` uses the current device and its current stream;
    only ``stream`` uses the stream's device; only ``device`` uses that device's
    current stream.

    Release is automatic when the last reference is dropped, or explicit with
    :meth:`close` or ``with``. Device memory is freed in stream order, like
    ``wp.array``; the native denoiser is destroyed once the work submitted
    before the release has finished.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        upscale: bool = False,
        device: wp.DeviceLike = None,
        stream: wp.Stream | None = None,
    ):
        self._denoiser = None
        if width < 1 or height < 1:
            raise ValueError("Denoiser input dimensions must be positive")
        self._context, self.stream = resolve(device, stream)
        self.width, self.height = int(width), int(height)
        scale = 2 if upscale else 1
        self.output_width, self.output_height = self.width * scale, self.height * scale
        self._history_valid = False
        model = native.DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X if upscale else native.DENOISER_MODEL_KIND_TEMPORAL_AOV
        options = native.DenoiserOptions()
        options.guideAlbedo = 1
        options.guideNormal = 1
        device = self.stream.device
        with wp.ScopedStream(self.stream):
            try:
                self._denoiser = self._context.native.denoiserCreate(model, options)
                sizes = self._denoiser.computeMemoryResources(self.width, self.height)
                self._state = wp.empty(sizes.stateSizeInBytes, dtype=wp.uint8, device=device)
                self._scratch_size = max(sizes.withoutOverlapScratchSizeInBytes, sizes.computeAverageColorSizeInBytes)
                # 2x upscaling from 1080p needs over 2 GiB of scratch, more elements than a uint8 array can hold.
                self._scratch = wp.empty((self._scratch_size + 7) // 8, dtype=wp.uint64, device=device)
                self._average_color = wp.empty(3, dtype=wp.float32, device=device)
                self._flow = wp.empty((self.height, self.width), dtype=wp.vec2, device=device)
                shape = (self.output_height, self.output_width)
                self._output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._previous_output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._guide_pixel_size = int(sizes.internalGuideLayerPixelSizeInBytes)
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

    @property
    def device(self) -> wp.Device:
        return self.stream.device

    @property
    def closed(self) -> bool:
        return self._denoiser is None

    def _image(self, array, pixel_format, *, pixel_stride=None, width=None, height=None):
        image = native.Image2D()
        image.data = array.ptr
        image.width = self.width if width is None else width
        image.height = self.height if height is None else height
        image.pixelStrideInBytes = array.strides[1] if pixel_stride is None else pixel_stride
        image.rowStrideInBytes = array.strides[0] if pixel_stride is None else image.width * pixel_stride
        image.format = pixel_format
        return image

    def apply(
        self, color: wp.array, *, albedo: wp.array, normal: wp.array, motion: wp.array, reset: bool = False
    ) -> wp.array:
        """Queue denoising of ``color`` and return the denoised float4 image.

        Inputs are contiguous ``(height, width)`` arrays on the denoiser's device:
        ``color`` is float4 HDR, ``albedo`` and ``normal`` (world space) are
        float3 or float4 (``w`` is ignored), and ``motion`` is float2, the
        offset from each pixel to its position in the previous frame, in input
        pixels. ``reset=True`` discards the temporal history, e.g. after a cut.

        Inputs are read on the denoiser's stream. The returned array is owned by
        the denoiser and stays valid until the next call.
        """
        if self._denoiser is None:
            raise RuntimeError("Denoiser is closed")
        for name, array, dtypes in (
            ("color", color, (wp.vec4,)),
            ("albedo", albedo, (wp.vec3, wp.vec4)),
            ("normal", normal, (wp.vec3, wp.vec4)),
            ("motion", motion, (wp.vec2,)),
        ):
            if array.shape != (self.height, self.width) or array.dtype not in dtypes:
                types = " or ".join(dtype.__name__ for dtype in dtypes)
                raise ValueError(f"{name} must be a {types} array of shape {(self.height, self.width)}")
            if array.device != self.device or not array.is_contiguous:
                raise ValueError(f"{name} must be contiguous on {self.device}")
        stream = int(self.stream.cuda_stream)
        with wp.ScopedStream(self.stream):
            if reset or not self._history_valid:
                self._previous_guide.zero_()
                self._previous_output.zero_()
                self._history_valid = False
            wp.launch(_prepare_flow, dim=(self.height, self.width), inputs=[motion, self._flow])
            guides = native.DenoiserGuideLayer()
            guides.albedo = self._image(albedo, native.PIXEL_FORMAT_FLOAT3)
            guides.normal = self._image(normal, native.PIXEL_FORMAT_FLOAT3)
            guides.flow = self._image(self._flow, native.PIXEL_FORMAT_FLOAT2)
            for field, buffer in (
                ("previousOutputInternalGuideLayer", self._previous_guide),
                ("outputInternalGuideLayer", self._output_guide),
            ):
                setattr(
                    guides,
                    field,
                    self._image(
                        buffer,
                        native.PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,
                        pixel_stride=self._guide_pixel_size,
                        width=self.output_width,
                        height=self.output_height,
                    ),
                )
            layer = native.DenoiserLayer()
            layer.input = self._image(color, native.PIXEL_FORMAT_FLOAT4)
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
            params = native.DenoiserParams()
            params.hdrAverageColor = self._average_color.ptr
            params.temporalModeUsePreviousLayers = int(self._history_valid)
            self._denoiser.computeAverageColor(
                stream,
                layer.input,
                self._average_color.ptr,
                self._scratch.ptr,
                self._scratch_size,
            )
            self._denoiser.invoke(
                stream,
                params,
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
            self._history_valid = True
            return result

    def close(self) -> None:
        """Make the denoiser unusable and release it; the native denoiser outlives work already submitted."""
        if self._denoiser is None:
            return
        denoiser, self._denoiser = self._denoiser, None
        self._history_valid = False
        self._state = self._scratch = self._average_color = self._flow = None
        self._output = self._previous_output = self._previous_guide = self._output_guide = None
        self._context.release(denoiser.destroy, (self.stream,))

    def __enter__(self) -> Denoiser:
        if self._denoiser is None:
            raise RuntimeError("Denoiser is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        if getattr(self, "_denoiser", None) is not None:
            try:
                self.close()
            except Exception:
                pass

    def __repr__(self):
        state = "closed, " if self._denoiser is None else ""
        return f"Denoiser({state}{self.width}x{self.height}, device={self.device.alias!r})"
