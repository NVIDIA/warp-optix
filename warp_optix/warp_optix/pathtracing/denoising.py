# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared render guides and stream-ordered OptiX and DLSS denoisers for Warp arrays."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_optix import native


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
            raise ValueError("output_size must match the denoised size, allowing one cropped pixel for 2x upscaling")
        self._denoiser = None
        self._history_valid = False
        model_name = "DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X" if upscale else "DENOISER_MODEL_KIND_TEMPORAL_AOV"
        if not hasattr(native, model_name) or not hasattr(native.DenoiserParams(), "temporalModeUsePreviousLayers"):
            raise RuntimeError("OptiX temporal denoising requires updated PyOptiX denoiser bindings")
        options = native.DenoiserOptions()
        options.guideAlbedo = 1
        options.guideNormal = 1
        device = self.stream.device
        with wp.ScopedStream(self.stream):
            try:
                self._denoiser = context.denoiserCreate(getattr(native, model_name), options)
                sizes = self._denoiser.computeMemoryResources(self.width, self.height)
                self._state = wp.empty(sizes.stateSizeInBytes, dtype=wp.uint8, device=device)
                self._scratch = wp.empty(
                    max(
                        sizes.withoutOverlapScratchSizeInBytes,
                        sizes.computeAverageColorSizeInBytes,
                    ),
                    dtype=wp.uint8,
                    device=device,
                )
                self._average_color = wp.empty(3, dtype=wp.float32, device=device)
                self._flow = wp.empty((self.height, self.width), dtype=wp.vec2, device=device)
                shape = (self.output_height, self.output_width)
                self._output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._previous_output = wp.zeros(shape, dtype=wp.vec4, device=device)
                self._cropped_output = (
                    wp.empty((target_height, target_width), dtype=wp.vec4, device=device)
                    if (target_height, target_width) != shape
                    else None
                )
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
                    self._scratch.size,
                )
            except Exception:
                self.close()
                raise

    def _image(self, array, pixel_format, *, pixel_stride=None, width=None, height=None):
        image = self._optix.Image2D()
        image.data = array.ptr
        image.width = self.width if width is None else width
        image.height = self.height if height is None else height
        image.pixelStrideInBytes = array.strides[1] if pixel_stride is None else pixel_stride
        image.rowStrideInBytes = array.strides[0] if pixel_stride is None else image.width * pixel_stride
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
                raise ValueError(f"{name} must be a {dtype.__name__} array of shape {(self.height, self.width)}")
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


# ---------------------------------------------------------------------------
# DLSS Ray Reconstruction
# ---------------------------------------------------------------------------

# DlssDenoiser quality names -> native DlssPerfQuality members. "dlaa" denoises
# at full resolution; the others render below the output size and upscale.
DLSS_QUALITY_MODES = {
    "dlaa": "DLAA",
    "max_quality": "MAX_QUALITY",
    "balanced": "BALANCED",
    "max_perf": "MAX_PERF",
    "ultra_performance": "ULTRA_PERFORMANCE",
}


def dlss_rr_requirements(device=None, *, feature_search_path: str | None = None) -> dict:
    """Ask the driver whether DLSS Ray Reconstruction can run on a CUDA device.

    Returns a dict with ``supported`` and diagnostics. Never raises for missing
    support: builds without DLSS, systems without the NGX driver library, and
    drivers older than R620 (which do not ship the Ray Reconstruction runtime)
    all report ``supported=False`` with a ``reason``.
    """
    if not getattr(native, "DLSS_ENABLED", False):
        return {"supported": False, "reason": "warp-optix was built without DLSS (WARP_OPTIX_ENABLE_DLSS=OFF)"}
    device = wp.get_device(device if device is not None else "cuda")
    if not device.is_cuda:
        return {"supported": False, "reason": "DLSS Ray Reconstruction requires a CUDA device"}
    info = dict(native.dlssRRQueryRequirements(cudaDevice=device.ordinal, featureSearchPath=feature_search_path or ""))
    if not info["supported"]:
        info["reason"] = (
            f"driver reports DLSS Ray Reconstruction unsupported ({info['result_string']}); "
            "the built-in runtime requires an NVIDIA driver R620 or newer"
        )
    return info


def dlss_rr_available(device=None, *, feature_search_path: str | None = None) -> bool:
    """Return whether DLSS Ray Reconstruction can run on a CUDA device."""
    return bool(dlss_rr_requirements(device, feature_search_path=feature_search_path)["supported"])


def _dlss_quality(name: str):
    if name not in DLSS_QUALITY_MODES:
        raise ValueError(f"quality must be one of {sorted(DLSS_QUALITY_MODES)}")
    return getattr(native.DlssPerfQuality, DLSS_QUALITY_MODES[name])


def _column_major(matrix) -> list[float]:
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (4, 4):
        raise ValueError("view and projection matrices must be 4x4")
    return m.T.reshape(-1).tolist()


class DlssDenoiser:
    """DLSS Ray Reconstruction: denoising, optionally with upscaling, on one CUDA stream.

    Uses the NGX runtime that ships with the NVIDIA driver (R620+); warp-optix
    distributes no DLSS binary. ``width``/``height`` are the render (input) size and
    ``output_size`` the reconstructed size; pick the render size for a quality mode
    with :meth:`optimal_render_size`. ``device``/``stream``: neither given uses the
    current CUDA device and its current stream; only ``stream`` uses its device;
    both must match.

    Inputs are copied into owned CUDA textures. DLSS needs every guide of
    :class:`DenoiserInputs`, including ``depth`` (linear view depth) and
    ``specular_albedo``; ``specular_hit_distance`` is optional.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        output_size: tuple[int, int] | None = None,
        quality: str = "dlaa",
        preset: str = "F",
        device=None,
        stream: wp.Stream | None = None,
        feature_search_path: str | None = None,
        enable_logging: bool = False,
    ):
        self._denoiser = None
        self._context = None
        self._textures = {}
        self._output_texture = None
        self._output = None
        if not getattr(native, "DLSS_ENABLED", False):
            raise RuntimeError("warp-optix was built without DLSS; rebuild with WARP_OPTIX_ENABLE_DLSS=ON")
        if width < 1 or height < 1:
            raise ValueError("Denoiser input dimensions must be positive")
        if stream is not None:
            if device is not None and wp.get_device(device) != stream.device:
                raise ValueError("Stream and device must be the same CUDA device")
            self.stream = stream
        else:
            self.stream = wp.get_stream(device if device is not None else "cuda")
        self.device = self.stream.device
        self.width, self.height = int(width), int(height)
        self.output_width, self.output_height = (int(v) for v in (output_size or (width, height)))
        if self.output_width < self.width or self.output_height < self.height:
            raise ValueError("output_size must be at least the render size")

        with wp.ScopedStream(self.stream):
            try:
                self._context = native.DlssRRContext()
                self._context.init(featureSearchPath=feature_search_path or "", enableLogging=enable_logging)
                if not self._context.isDlssRRAvailable():
                    raise RuntimeError(
                        "DLSS Ray Reconstruction is not available; the driver's built-in runtime "
                        "requires an NVIDIA driver R620 or newer"
                    )
                info = native.DlssRRInitInfo()
                info.inputWidth, info.inputHeight = self.width, self.height
                info.outputWidth, info.outputHeight = self.output_width, self.output_height
                info.quality = _dlss_quality(quality)
                info.preset = getattr(native.RayReconstructionHintRenderPreset, preset)
                # Motion vectors exclude jitter and are given at render resolution;
                # radiance is linear HDR and depth is linear view distance.
                info.mvJittered = False
                info.lowResolutionMotionVectors = True
                info.isContentHDR = True
                self._denoiser = self._context.initDlssRR(info, int(self.stream.cuda_stream))

                resource = native.DlssRRResource
                for name, channels, slot in (
                    ("color", 4, resource.RESOURCE_COLOR_IN),
                    ("albedo", 4, resource.RESOURCE_DIFFUSE_ALBEDO),
                    ("specular_albedo", 4, resource.RESOURCE_SPECULAR_ALBEDO),
                    ("normal", 4, resource.RESOURCE_NORMALROUGHNESS),
                    ("motion", 2, resource.RESOURCE_MOTIONVECTOR),
                    ("depth", 1, resource.RESOURCE_LINEARDEPTH),
                    ("specular_hit_distance", 1, resource.RESOURCE_SPECULAR_HITDISTANCE),
                ):
                    texture = wp.Texture2D(
                        width=self.width,
                        height=self.height,
                        num_channels=channels,
                        dtype=wp.float32,
                        device=self.device,
                    )
                    self._textures[name] = texture
                    self._denoiser.setResource(slot, texture.cuda_texture)
                self._output_texture = wp.Texture2D(
                    width=self.output_width,
                    height=self.output_height,
                    num_channels=4,
                    dtype=wp.float32,
                    device=self.device,
                    surface_access=True,
                )
                self._denoiser.setResource(resource.RESOURCE_COLOR_OUT, self._output_texture.cuda_surface)
                self._output = wp.zeros((self.output_height, self.output_width), dtype=wp.vec4, device=self.device)
            except BaseException:
                self.close()
                raise

    @staticmethod
    def optimal_render_size(
        output_size: tuple[int, int], quality: str = "balanced", *, feature_search_path: str | None = None
    ) -> tuple[int, int]:
        """Return the render size DLSS recommends for an output size and quality mode."""
        if quality == "dlaa":
            return int(output_size[0]), int(output_size[1])
        quality_value = _dlss_quality(quality)
        context = native.DlssRRContext()
        context.init(featureSearchPath=feature_search_path or "")
        try:
            sizes = context.querySupportedDlssInputSizes(int(output_size[0]), int(output_size[1]), quality_value)
            return int(sizes.optimalWidth), int(sizes.optimalHeight)
        finally:
            context.deinit()

    def apply(
        self,
        inputs: DenoiserInputs,
        *,
        view_matrix,
        projection_matrix,
        jitter: tuple[float, float] = (0.0, 0.0),
        reset: bool = False,
    ) -> wp.array:
        """Reconstruct one frame; returns the owned ``(out_h, out_w)`` vec4 output.

        ``view_matrix``/``projection_matrix`` are row-major 4x4 world-to-view and
        view-to-clip matrices. ``jitter`` is the sub-pixel offset of this frame in
        render pixels, as DLSS defines it. Set ``reset`` after a camera cut. The
        output is overwritten by the next call.
        """
        if self._denoiser is None:
            raise RuntimeError("DlssDenoiser is closed")
        shape = (self.height, self.width)
        for name in ("color", "normal", "albedo", "motion", "depth", "specular_albedo"):
            value = getattr(inputs, name)
            if value is None:
                raise ValueError(f"DLSS Ray Reconstruction requires inputs.{name}")
            if value.shape != shape:
                raise ValueError(f"inputs.{name} must have shape {shape}")
        with wp.ScopedStream(self.stream):
            for name, texture in self._textures.items():
                value = getattr(inputs, name)
                if value is not None:
                    texture.copy_from(value)
            self._denoiser.denoise(
                self.width,
                self.height,
                float(jitter[0]),
                float(jitter[1]),
                _column_major(view_matrix),
                _column_major(projection_matrix),
                bool(reset),
            )
            self._output_texture.copy_to(self._output)
        return self._output

    def close(self):
        """Finish queued work and release the DLSS feature and its textures."""
        if self._denoiser is not None or self._context is not None:
            wp.synchronize_stream(self.stream)
        if self._denoiser is not None:
            self._denoiser.deinit()
            self._denoiser = None
        if self._context is not None:
            self._context.deinit()
            self._context = None
        self._textures = {}
        self._output_texture = None
        self._output = None

    def __del__(self):
        with suppress(Exception):
            self.close()
