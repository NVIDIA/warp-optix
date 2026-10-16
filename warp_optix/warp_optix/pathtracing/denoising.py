# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared render guides and a stream-ordered OptiX denoiser for Warp arrays."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import warp as wp

from warp_optix import native


@dataclass(frozen=True)
class DenoiserInputs:
    """HDR render buffers, indexed [y, x], shared by the reconstruction backends.

    Normals are world-space float4 (roughness in w); albedo is float4 (RGB
    plus metadata). Motion is float2, current-to-previous in input pixels,
    excluding camera jitter. Depth and specular guides are used only by DLSS.
    """

    color: wp.array
    normal: wp.array
    albedo: wp.array
    motion: wp.array
    depth: wp.array | None = None
    specular_albedo: wp.array | None = None
    specular_hit_distance: wp.array | None = None


# One registration table drives DLSS allocation, binding, and frame uploads.
DLSS_INPUTS = (
    ("color", "_dlss_color_in_tex", 4, "RESOURCE_COLOR_IN"),
    ("normal", "_dlss_normal_roughness_tex", 4, "RESOURCE_NORMALROUGHNESS"),
    ("motion", "_dlss_motion_tex", 2, "RESOURCE_MOTIONVECTOR"),
    ("depth", "_dlss_depth_tex", 1, "RESOURCE_LINEARDEPTH"),
    ("albedo", "_dlss_diffuse_tex", 4, "RESOURCE_DIFFUSE_ALBEDO"),
    ("specular_albedo", "_dlss_specular_tex", 4, "RESOURCE_SPECULAR_ALBEDO"),
    (
        "specular_hit_distance",
        "_dlss_spec_hit_dist_tex",
        1,
        "RESOURCE_SPECULAR_HITDISTANCE",
    ),
)


@wp.kernel(enable_backward=False)
def _prepare_flow(motion: wp.array2d(dtype=wp.vec2), flow: wp.array2d(dtype=wp.vec2)):
    y, x = wp.tid()
    # OptiX expects previous-to-current flow, whereas the renderer/DLSS use
    # current-to-previous reprojection offsets.
    flow[y, x] = -motion[y, x]


@wp.kernel(enable_backward=False)
def _crop_output(source: wp.array2d(dtype=wp.vec4), output: wp.array2d(dtype=wp.vec4)):
    y, x = wp.tid()
    output[y, x] = source[y, x]


class OptixDenoiser:
    """Temporal denoising, optionally with 2x upscaling, on one CUDA stream.

    The context must belong to the stream's CUDA device. Input arrays are
    borrowed until work on that stream completes. Output and history buffers
    are owned by this object; call close() before releasing the context.
    """

    def __init__(
        self,
        context,
        width: int,
        height: int,
        *,
        upscale=False,
        stream=None,
        output_size=None,
    ):
        if width < 1 or height < 1:
            raise ValueError("Denoiser input dimensions must be positive")
        self._optix = native
        self.stream = stream if stream is not None else wp.get_stream("cuda")
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
            raise ValueError(
                "output_size must match the denoised size, allowing one cropped pixel for 2x upscaling"
            )
        self._denoiser = None
        self._history_valid = False
        model_name = (
            "DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X"
            if upscale
            else "DENOISER_MODEL_KIND_TEMPORAL_AOV"
        )
        if not hasattr(native, model_name) or not hasattr(
            native.DenoiserParams(), "temporalModeUsePreviousLayers"
        ):
            raise RuntimeError(
                "OptiX temporal denoising requires updated PyOptiX denoiser bindings"
            )
        options = native.DenoiserOptions()
        options.guideAlbedo = 1
        options.guideNormal = 1
        device = self.stream.device
        with wp.ScopedStream(self.stream):
            try:
                self._denoiser = context.denoiserCreate(
                    getattr(native, model_name), options
                )
                sizes = self._denoiser.computeMemoryResources(self.width, self.height)
                self._state = wp.empty(
                    sizes.stateSizeInBytes, dtype=wp.uint8, device=device
                )
                self._scratch = wp.empty(
                    max(
                        sizes.withoutOverlapScratchSizeInBytes,
                        sizes.computeAverageColorSizeInBytes,
                    ),
                    dtype=wp.uint8,
                    device=device,
                )
                self._average_color = wp.empty(3, dtype=wp.float32, device=device)
                self._flow = wp.empty(
                    (self.height, self.width), dtype=wp.vec2, device=device
                )
                shape = (self.output_height, self.output_width)
                self._output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._previous_output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._cropped_output = (
                    wp.empty(
                        (target_height, target_width), dtype=wp.vec4, device=device
                    )
                    if (target_height, target_width) != shape
                    else None
                )
                self._guide_pixel_size = int(sizes.internalGuideLayerPixelSizeInBytes)
                guide_size = (
                    self.output_width * self.output_height * self._guide_pixel_size
                )
                self._previous_guide = wp.zeros(
                    guide_size, dtype=wp.uint8, device=device
                )
                self._output_guide = wp.zeros(guide_size, dtype=wp.uint8, device=device)
                self._denoiser.setup(
                    int(self.stream.cuda_stream),
                    self.width,
                    self.height,
                    self._state.ptr,
                    self._state.size,
                    self._scratch.ptr,
                    self._scratch.size,
                )
            except Exception:
                self.close()
                raise

    def _image(
        self, array, pixel_format, *, pixel_stride=None, width=None, height=None
    ):
        image = self._optix.Image2D()
        image.data = array.ptr
        image.width = self.width if width is None else width
        image.height = self.height if height is None else height
        image.pixelStrideInBytes = (
            array.strides[1] if pixel_stride is None else pixel_stride
        )
        image.rowStrideInBytes = (
            array.strides[0] if pixel_stride is None else image.width * pixel_stride
        )
        image.format = pixel_format
        return image

    def apply(self, inputs: DenoiserInputs, *, reset=False):
        """Queue denoising and return the owned float4 Warp output array."""
        if self._denoiser is None:
            raise RuntimeError("OptiX denoiser is closed")
        for name, dtype in (
            ("color", wp.vec4),
            ("normal", wp.vec4),
            ("albedo", wp.vec4),
            ("motion", wp.vec2),
        ):
            array = getattr(inputs, name)
            if array.shape != (self.height, self.width) or array.dtype != dtype:
                raise ValueError(
                    f"{name} must be a {dtype.__name__} array of shape {(self.height, self.width)}"
                )
            if array.device != self.stream.device or not array.is_contiguous:
                raise ValueError(f"{name} must be contiguous on {self.stream.device}")
        optix = self._optix
        stream = int(self.stream.cuda_stream)
        with wp.ScopedStream(self.stream):
            if reset or not self._history_valid:
                self._previous_guide.zero_()
                self._previous_output.zero_()
                self._history_valid = False
            wp.launch(
                _prepare_flow,
                dim=(self.height, self.width),
                inputs=[inputs.motion, self._flow],
                stream=self.stream,
            )
            guides = optix.DenoiserGuideLayer()
            guides.albedo = self._image(inputs.albedo, optix.PIXEL_FORMAT_FLOAT3)
            guides.normal = self._image(inputs.normal, optix.PIXEL_FORMAT_FLOAT3)
            guides.flow = self._image(self._flow, optix.PIXEL_FORMAT_FLOAT2)
            for field, buffer in (
                ("previousOutputInternalGuideLayer", self._previous_guide),
                ("outputInternalGuideLayer", self._output_guide),
            ):
                setattr(
                    guides,
                    field,
                    self._image(
                        buffer,
                        optix.PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,
                        pixel_stride=self._guide_pixel_size,
                        width=self.output_width,
                        height=self.output_height,
                    ),
                )
            layer = optix.DenoiserLayer()
            layer.input = self._image(inputs.color, optix.PIXEL_FORMAT_FLOAT4)
            layer.output = self._image(
                self._output,
                optix.PIXEL_FORMAT_FLOAT4,
                width=self.output_width,
                height=self.output_height,
            )
            layer.previousOutput = self._image(
                self._previous_output,
                optix.PIXEL_FORMAT_FLOAT4,
                width=self.output_width,
                height=self.output_height,
            )
            params = optix.DenoiserParams()
            params.hdrAverageColor = self._average_color.ptr
            params.temporalModeUsePreviousLayers = int(self._history_valid)
            self._denoiser.computeAverageColor(
                stream,
                layer.input,
                self._average_color.ptr,
                self._scratch.ptr,
                self._scratch.size,
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
                self._scratch.size,
            )
            result = self._output
            self._output, self._previous_output = self._previous_output, self._output
            self._output_guide, self._previous_guide = (
                self._previous_guide,
                self._output_guide,
            )
            self._history_valid = True
            if self._cropped_output is not None:
                wp.launch(
                    _crop_output,
                    dim=self._cropped_output.shape,
                    inputs=[result, self._cropped_output],
                    stream=self.stream,
                )
                result = self._cropped_output
            return result

    def close(self):
        """Finish queued work and destroy the native denoiser."""
        if self._denoiser is not None:
            wp.synchronize_stream(self.stream)
            self._denoiser.destroy()
            self._denoiser = None
            self._history_valid = False
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
