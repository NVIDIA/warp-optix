# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
OptiX Path Tracing Viewer.
Python/OptiX path tracing viewer with DLSS RR support.

This viewer renders a scene using OptiX ray tracing with PBR materials,
displaying raw buffers (radiance, normals, depth, etc.) for debugging.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

import numpy as np
import warp as wp

import warp_optix as wo
from warp_optix import native

from . import pathtracing_warp_kernels as pwk
from .camera import Camera
from .defaults import DEFAULT_VIEWER_HEIGHT, DEFAULT_VIEWER_WIDTH, DLSS_PROJECT_ID
from .denoising import DLSS_INPUTS, DenoiserInputs, OptixDenoiser
from .environment_map import EnvironmentMap
from .lighting import RENDERER_RADIANCE_PER_NIT
from .scene import Scene
from .sky_lut import generate_physical_sky_lut
from .tonemap import Tonemapper
from .volume_composite import VolumeParams, composite_volume

try:
    from PIL import Image
except ImportError:
    Image = None

# Initialize warp
wp.init()

logger = logging.getLogger(__name__)


@wp.kernel
def _reset_accum_buffer(accum: wp.array2d(dtype=wp.vec4)):
    x, y = wp.tid()
    accum[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _accumulate_sample(
    sample: wp.array2d(dtype=wp.vec4),
    accum: wp.array2d(dtype=wp.vec4),
    sample_index: int,
):
    x, y = wp.tid()
    s = sample[y, x]
    a = accum[y, x]
    t = 1.0 / float(sample_index + 1)
    accum[y, x] = a + (s - a) * t


@wp.kernel(enable_backward=False)
def _prepare_device_camera(
    positions: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    up_directions: wp.array(dtype=wp.vec3),
    fovs: wp.array(dtype=wp.float32),
    camera_transform: wp.mat44,
    aspect: float,
    initialized: wp.array(dtype=wp.int32),
    states: wp.array(dtype=pwk.DeviceCameraState),
):
    position4 = camera_transform * wp.vec4(positions[0][0], positions[0][1], positions[0][2], 1.0)
    target4 = camera_transform * wp.vec4(targets[0][0], targets[0][1], targets[0][2], 1.0)
    up4 = camera_transform * wp.vec4(up_directions[0][0], up_directions[0][1], up_directions[0][2], 0.0)
    position = wp.vec3(position4[0], position4[1], position4[2])
    forward = wp.vec3(target4[0], target4[1], target4[2]) - position
    if wp.length(forward) <= 1.0e-8:
        forward = wp.vec3(0.0, 0.0, -1.0)
    else:
        forward = wp.normalize(forward)
    world_up = wp.vec3(up4[0], up4[1], up4[2])
    right = wp.cross(forward, world_up)
    if wp.length(right) <= 1.0e-8:
        right = wp.vec3(1.0, 0.0, 0.0)
    else:
        right = wp.normalize(right)
    up = wp.normalize(wp.cross(right, forward))
    tan_half_fov = wp.tan(wp.radians(wp.clamp(fovs[0], 5.0, 120.0)) * 0.5)

    state = states[0]
    if initialized[0] == 0:
        state.previous_position = position
        state.previous_forward = forward
        state.previous_right = right
        state.previous_up = up
        state.previous_tan_half_fov = tan_half_fov
        state.previous_aspect = aspect
        initialized[0] = 1
    else:
        state.previous_position = state.position
        state.previous_forward = state.forward
        state.previous_right = state.right
        state.previous_up = state.up
        state.previous_tan_half_fov = state.tan_half_fov
        state.previous_aspect = state.aspect
    state.position = position
    state.forward = forward
    state.right = right
    state.up = up
    state.tan_half_fov = tan_half_fov
    state.aspect = aspect
    states[0] = state


class PathTracingViewer:
    """
    OptiX Path Tracing Viewer.

    Renders a scene using hardware ray tracing with PBR materials.
    """

    # Output modes
    OUTPUT_FINAL = 0
    OUTPUT_RADIANCE = 1
    OUTPUT_DEPTH = 2
    OUTPUT_MOTION = 3
    OUTPUT_NORMAL = 4
    OUTPUT_ROUGHNESS = 5
    OUTPUT_DIFFUSE = 6
    OUTPUT_SPECULAR = 7
    OUTPUT_SPEC_HITDIST = 8

    DLSS_QUALITY_MODES = {
        "performance": "MAX_PERF",
        "balanced": "BALANCED",
        "quality": "MAX_QUALITY",
        "ultra_performance": "ULTRA_PERFORMANCE",
        "native": "DLAA",
    }

    def __init__(
        self,
        width: int = DEFAULT_VIEWER_WIDTH,
        height: int = DEFAULT_VIEWER_HEIGHT,
        scene_setup: Callable[[Scene], None] | None = None,
        camera: Camera | None = None,
        accumulate_samples: bool = False,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        russian_roulette_start_bounce: int = 3,
        use_halton_jitter: bool = True,
        enable_dlss_rr: bool = True,
        enable_set: bool = True,
        dlss_quality: str = "quality",
        enable_texture_mipmaps: bool = False,
        backface_culling: bool = True,
        denoiser: str | None = None,
        optix_upscale: bool = False,
    ):
        """
        Initialize the path tracing viewer.

        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
        self._render_width = width
        self._render_height = height
        self.frame_index = 0
        self.sample_index = 0
        self.accumulate_samples = accumulate_samples
        self.samples_per_frame = max(1, int(samples_per_frame))
        self.max_bounces = max(1, int(max_bounces))
        self._pipeline_max_bounces = self.max_bounces
        self.direct_light_samples = max(1, int(direct_light_samples))
        self.russian_roulette_start_bounce = max(1, int(russian_roulette_start_bounce))
        self.use_halton_jitter = bool(use_halton_jitter)
        self.enable_dlss_rr = bool(enable_dlss_rr)
        self._denoiser_explicit = denoiser is not None
        self.denoiser = self._normalize_denoiser(
            denoiser if denoiser is not None else ("dlss" if enable_dlss_rr else "none")
        )
        self.optix_upscale = bool(optix_upscale)
        self._optix_denoiser = None
        self.denoiser_error: str | None = None
        self.enable_set = bool(enable_set)
        self.backface_culling = bool(backface_culling)
        self.enable_texture_mipmaps = bool(enable_texture_mipmaps)
        self.dlss_quality = self._normalize_dlss_quality(dlss_quality)
        self._set_active = False
        self._render_stream = wp.get_stream("cuda")

        # Camera
        if camera is None:
            self.camera = Camera(
                position=(0.0, 0.0, 6.0),
                target=(0.0, 0.0, 0.0),
                fov=45.0,
                aspect_ratio=width / height,
            )
        else:
            self.camera = camera
            self.camera.set_aspect_ratio(width, height)

        # Optional external scene configuration callback
        self._scene_setup = scene_setup

        # Default to path-traced final output.
        self.output_mode = self.OUTPUT_FINAL

        # OptiX state (initialized in build())
        self._optix = None
        self._ctx = None
        self._pipeline = None

        # Scene
        self._scene = None

        # Tonemapper
        self._tonemapper = Tonemapper(width, height)

        # Output buffers
        self._color_buffer = None
        self._sky_lut_buffer = None
        self._sky_lut_state = None
        self._accum_buffer = None
        self._normal_roughness_buffer = None
        self._motion_buffer = None
        self._depth_buffer = None
        self._diffuse_buffer = None
        self._specular_buffer = None
        self._spec_hit_dist_buffer = None
        self._instance_transforms_buffer = None
        self._prev_instance_transforms_buffer = None
        self._prev_instance_transforms_valid = False

        self._launch_params = None
        self._instance_transform_count = 0
        self._device_camera_positions = None
        self._device_camera_targets = None
        self._device_camera_up = None
        self._device_camera_fovs = None
        self._device_camera_transform = wp.mat44(*np.eye(4, dtype=np.float32).reshape(-1))
        self._device_camera_initialized = None
        self._device_camera_state = None
        self._volume = None
        self._volume_min = (0.0, 0.0, 0.0)
        self._volume_max = (0.0, 0.0, 0.0)
        self._volume_density_scale = 1.0
        self._volume_step_size = 0.01
        self._volume_cool_colors = (
            (0.0, 0.025, 0.18),
            (0.0, 0.20, 0.95),
            (0.0, 0.92, 0.82),
        )
        self._volume_warm_colors = (
            (0.30, 0.008, 0.0),
            (1.0, 0.075, 0.0),
            (1.0, 0.62, 0.0),
        )
        self._volume_emission = 0.0
        self._volume_anisotropy = 0.0
        self._volume_transfer_table = None
        self._volume_density_feature = False
        self._volume_params = None

        # CUDA surface objects
        self._color_surface = None
        self._dlss_context = None
        self._dlss_denoiser = None
        self._dlss_color_in_tex = None
        self._dlss_normal_roughness_tex = None
        self._dlss_motion_tex = None
        self._dlss_depth_tex = None
        self._dlss_diffuse_tex = None
        self._dlss_specular_tex = None
        self._dlss_spec_hit_dist_tex = None
        self._dlss_color_out_tex = None
        self._dlss_output_surface = 0
        self._dlss_enabled = False
        self._dlss_reset_history = True
        self._dlss_init_error: str | None = None
        self._dlss_status_reported = False
        self._last_jitter = (0.0, 0.0)

        # Previous-frame camera matrices for motion vectors.
        self._prev_view = None
        self._prev_proj = None
        self._prev_mvp = None
        self._last_accum_view = None
        self._last_accum_proj = None
        self._sync_prev_camera_matrices_to_current()
        self._last_output_mode = self.output_mode

        # Physical sky defaults aligned with the upstream DLSS-RR sample behavior.
        self.sky_rgb_unit_conversion = (
            RENDERER_RADIANCE_PER_NIT,
            RENDERER_RADIANCE_PER_NIT,
            RENDERER_RADIANCE_PER_NIT,
        )
        self.sky_multiplier = 1.0
        self.sky_haze = 0.5
        self.sky_redblueshift = 0.05
        self.sky_saturation = 1.0
        self.sky_horizon_height = 0.0
        self.sky_ground_color = (0.4, 0.4, 0.4)
        self.sky_horizon_blur = 1.0
        self.sky_night_color = (0.0, 0.0, 0.0)
        self.sky_sun_disk_intensity = 1.0
        # MinimalDlssRR PhysicalSky.Afternoon: 45-degree elevation,
        # 240-degree azimuth (west-southwest).
        self.sky_sun_direction = (-0.6123724, 0.7071068, -0.3535534)
        self.sky_sun_disk_scale = 1.0
        self.sky_sun_glow_intensity = 1.0
        self.sky_y_is_up = 1
        self.sky_grayscale = 0.5
        self.env_intensity = (1.0, 1.0, 1.0)
        self.analytic_light_intensity = 1.0
        self.emissive_material_intensity = 1.0
        self.env_rotation = 0.0
        self.use_path_regularization = True
        self.use_psr = True
        self.override_roughness = -1.0
        self.override_metallic = -1.0
        self.bitangent_flip = 1.0

        # Optional HDR environment map (lat-long, RGBA32F).
        self._env_map: EnvironmentMap | None = None

    @classmethod
    def _normalize_dlss_quality(cls, quality: str) -> str:
        value = str(quality).strip().lower().replace("-", "_")
        if value not in cls.DLSS_QUALITY_MODES:
            choices = ", ".join(cls.DLSS_QUALITY_MODES)
            raise ValueError(f"dlss_quality must be one of: {choices}")
        return value

    def set_dlss_quality(self, quality: str) -> None:
        """Select the DLSS input-resolution/quality mode."""
        value = self._normalize_dlss_quality(quality)
        if value == self.dlss_quality:
            return
        self.dlss_quality = value
        if self._optix is not None and self.denoiser in ("dlss", "auto"):
            wp.synchronize_stream(self._render_stream)
            self._init_denoiser()

    @staticmethod
    def _normalize_denoiser(value: str) -> str:
        value = str(value).strip().lower()
        if value not in ("auto", "dlss", "optix", "none"):
            raise ValueError("denoiser must be 'auto', 'dlss', 'optix', or 'none'")
        return value

    @property
    def active_denoiser(self) -> str:
        if self._dlss_enabled:
            return "dlss"
        return "optix" if self._optix_denoiser is not None else "none"

    def set_denoiser(self, denoiser: str, *, optix_upscale: bool | None = None):
        """Switch reconstruction backends without rebuilding the scene.

        Explicit selections require the backend; 'auto' tries DLSS, OptiX,
        then raw rendering. Backend-specific resolution settings are retained.
        """
        mode = self._normalize_denoiser(denoiser)
        upscale = self.optix_upscale if optix_upscale is None else bool(optix_upscale)
        if mode == self.denoiser and upscale == self.optix_upscale and self._denoiser_explicit:
            return
        previous = self.denoiser, self.optix_upscale, self._denoiser_explicit
        self.denoiser, self.optix_upscale, self._denoiser_explicit = mode, upscale, True
        if self._ctx is None:
            return
        wp.synchronize_stream(self._render_stream)
        try:
            self._init_denoiser()
        except Exception:
            self.denoiser, self.optix_upscale, self._denoiser_explicit = previous
            self._init_denoiser()
            raise

    def _destroy_optix_denoiser(self):
        if self._optix_denoiser is not None:
            self._optix_denoiser.close()
            self._optix_denoiser = None

    def _init_denoiser(self):
        # Release old reconstruction resources before replacing render targets
        # to avoid retaining two generations of allocations during large resizes.
        self._destroy_optix_denoiser()
        self._destroy_dlss_rr(restore_resolution=False)
        self.denoiser_error = None
        self.sample_index = 0
        self.frame_index = 0
        self._dlss_reset_history = True
        self._prev_instance_transforms_valid = False
        self._sync_prev_camera_matrices_to_current()
        if self._optix is None:
            self._set_render_resolution(self.width, self.height)
            return
        self.enable_dlss_rr = self.denoiser in ("dlss", "auto")
        if self.enable_dlss_rr:
            self._init_dlss_rr()
            if self._dlss_enabled:
                return
            self.denoiser_error = self._dlss_init_error
            if self.denoiser == "dlss" and self._denoiser_explicit:
                raise RuntimeError(f"DLSS initialization failed: {self.denoiser_error}")
        if self.denoiser in ("optix", "auto"):
            try:
                scale = 2 if self.optix_upscale else 1
                self._set_render_resolution(
                    (self.width + scale - 1) // scale,
                    (self.height + scale - 1) // scale,
                )
                self._optix_denoiser = OptixDenoiser(
                    self._ctx,
                    self._render_width,
                    self._render_height,
                    upscale=self.optix_upscale,
                    stream=self._render_stream,
                    output_size=(self.width, self.height),
                )
            except Exception as exc:
                if self.denoiser == "optix":
                    raise
                self.denoiser_error = f"{self.denoiser_error}; OptiX initialization failed: {exc}"
                logger.warning("Denoiser fallback: %s", self.denoiser_error)
                self._set_render_resolution(self.width, self.height)
        else:
            self._set_render_resolution(self.width, self.height)

    def _denoiser_inputs(self, color=None) -> DenoiserInputs:
        return DenoiserInputs(
            color=self._color_buffer if color is None else color,
            normal=self._normal_roughness_buffer,
            albedo=self._diffuse_buffer,
            motion=self._motion_buffer,
            depth=self._depth_buffer,
            specular_albedo=self._specular_buffer,
            specular_hit_distance=self._spec_hit_dist_buffer,
        )

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        russian_roulette_start_bounce: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust runtime ray budgets without rebuilding the OptiX pipeline."""
        if max_bounces is not None:
            value = int(max_bounces)
            if value < 1 or value > self._pipeline_max_bounces:
                raise ValueError(
                    "max_bounces must be in the range "
                    f"[1, {self._pipeline_max_bounces}]; construct the viewer with a "
                    "larger max_bounces value to raise the compiled limit"
                )
            self.max_bounces = value
        if direct_light_samples is not None:
            value = int(direct_light_samples)
            if value < 1:
                raise ValueError("direct_light_samples must be at least 1")
            self.direct_light_samples = value
        if russian_roulette_start_bounce is not None:
            value = int(russian_roulette_start_bounce)
            if value < 1:
                raise ValueError("russian_roulette_start_bounce must be at least 1")
            self.russian_roulette_start_bounce = value
        if samples_per_frame is not None:
            value = int(samples_per_frame)
            if value < 1:
                raise ValueError("samples_per_frame must be at least 1")
            self.samples_per_frame = value

    def set_backface_culling(self, enabled: bool) -> None:
        """Globally enable or disable triangle backface culling for every ray."""
        value = bool(enabled)
        if value == self.backface_culling:
            return
        self.backface_culling = value
        self.sample_index = 0
        self.frame_index = 0
        self._dlss_reset_history = True

    def set_volume(
        self,
        volume: wp.Volume | None,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(1.0, 1.0, 1.0),
        *,
        density_scale=1.0,
        step_size=0.01,
        cool_colors=(
            (0.0, 0.025, 0.18),
            (0.0, 0.20, 0.95),
            (0.0, 0.92, 0.82),
        ),
        warm_colors=(
            (0.30, 0.008, 0.0),
            (1.0, 0.075, 0.0),
            (1.0, 0.62, 0.0),
        ),
        emission=0.0,
        anisotropy=0.0,
        transfer_table=None,
        density_feature=False,
    ):
        """Set the NanoVDB volume composited by the path tracer."""
        self._volume = volume
        self._volume_min = tuple(float(v) for v in bounds_min)
        self._volume_max = tuple(float(v) for v in bounds_max)
        self._volume_density_scale = float(density_scale)
        self._volume_step_size = float(step_size)
        self._volume_cool_colors = tuple(tuple(float(v) for v in color) for color in cool_colors)
        self._volume_warm_colors = tuple(tuple(float(v) for v in color) for color in warm_colors)
        self._volume_emission = float(emission)
        self._volume_anisotropy = float(anisotropy)
        self._volume_density_feature = bool(density_feature)
        if transfer_table is None:
            self._volume_transfer_table = None
        elif isinstance(transfer_table, wp.array):
            if transfer_table.ndim != 1 or transfer_table.dtype != wp.vec4 or len(transfer_table) < 2:
                raise ValueError("transfer_table must be a 1D Warp vec4 array with at least two entries")
            if not transfer_table.device.is_cuda:
                raise ValueError("transfer_table must be CUDA-resident")
            self._volume_transfer_table = transfer_table
        else:
            table = np.asarray(transfer_table, dtype=np.float32)
            if table.ndim != 2 or table.shape[1] != 4 or len(table) < 2:
                raise ValueError("transfer_table must have shape (N, 4) with N >= 2")
            if not np.all(np.isfinite(table)) or np.any(table < 0.0):
                raise ValueError("transfer_table entries must be finite and nonnegative")
            self._volume_transfer_table = wp.array(table, dtype=wp.vec4, device="cuda")
        self.sample_index = 0
        self.frame_index = 0
        self._dlss_reset_history = True

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        """
        Load an HDR environment map from disk.

        The environment map is used for image-based lighting with importance sampling.

        Args:
            hdr_path: Path to HDR file (.hdr format)
            scaling: Intensity multiplier (default 1.0)
        """
        env_map = EnvironmentMap()
        if env_map.load_from_file(hdr_path, scaling=scaling):
            self._env_map = env_map
        else:
            logger.warning("Failed to load HDR environment: %s", hdr_path)

    def set_environment_color(self, color: tuple[float, float, float]):
        """
        Set a uniform color environment (useful for debugging or simple scenes).

        Args:
            color: RGB color values
        """
        env_map = EnvironmentMap()
        if env_map.load_from_color(color):
            self._env_map = env_map

    def clear_environment_map(self):
        """Clear the HDR environment map and use procedural sky only."""
        self._env_map = None

    @property
    def tonemapped_output(self):
        """Return the tonemapped output buffer used for display/extraction."""
        return self._tonemapper.output

    @property
    def linear_depth_output(self):
        """Return the current positive view-space depth buffer."""
        return self._depth_buffer

    @property
    def render_resolution(self) -> tuple[int, int]:
        """Return the internal render resolution as ``(width, height)``."""
        return self._render_width, self._render_height

    def build(self):
        """Build the OptiX pipeline and scene."""
        logger.info("Initializing OptiX path tracing viewer.")

        if self._ctx is not None:
            self.close()
        self._optix = native
        self._ctx = wo.create_context("cuda")
        try:
            self._set_active = False

            # Create scene
            self._scene = Scene(self._ctx, enable_texture_mipmaps=self.enable_texture_mipmaps)
            if self._scene_setup is not None:
                self._scene_setup(self._scene)
            else:
                self._scene.create_cornell_box()
            self._scene.build(native)

            # Create output buffers
            self._create_buffers()
            self._launch_params = pwk.PathtraceLaunchParams()
            self._init_denoiser()

            self._create_pipeline()

            logger.info("OptiX path tracing viewer build complete.")
            return True
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _create_cuda_texture_2d(
        height: int, width: int, channels: int, *, surface_access: bool = False
    ) -> wp.Texture2D:
        if channels == 1:
            data = np.zeros((height, width), dtype=np.float32)
        else:
            data = np.zeros((height, width, channels), dtype=np.float32)
        return wp.Texture2D(
            data,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.CLAMP,
            device="cuda",
            surface_access=surface_access,
        )

    @staticmethod
    def _half_res(value: int) -> int:
        # Keep dimensions even (where possible) to match common DLSS input expectations.
        v = max(1, int(value) // 2)
        if v > 1 and (v % 2) != 0:
            v -= 1
        return max(1, v)

    def _set_render_resolution(self, render_width: int, render_height: int):
        rw = max(1, int(render_width))
        rh = max(1, int(render_height))
        if rw == self._render_width and rh == self._render_height:
            return
        self._render_width = rw
        self._render_height = rh
        self._create_buffers()
        self.frame_index = 0
        self.sample_index = 0
        self._dlss_reset_history = True

    def _sync_prev_camera_matrices_to_current(self):
        """Initialize previous-frame camera transforms from the current camera pose.

        Mirrors reference first-frame behavior where prevMVP is set to currentMVP to avoid
        spurious large motion vectors after resets/resizes.
        """
        view = self.camera.get_view_matrix().copy()
        proj = self.camera.get_projection_matrix().copy()
        self._prev_view = view
        self._prev_proj = proj
        self._prev_mvp = (view @ proj).astype(np.float32)
        self._last_accum_view = view.copy()
        self._last_accum_proj = proj.copy()

    def _destroy_dlss_rr(self, *, restore_resolution: bool = True):
        if self._dlss_denoiser is not None:
            try:
                self._dlss_denoiser.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS denoiser: %s", exc)
        self._dlss_denoiser = None

        if self._dlss_context is not None:
            try:
                self._dlss_context.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS context: %s", exc)
        self._dlss_context = None
        # Release textures only after NGX has stopped using their CUDA handles.
        self._dlss_output_surface = 0
        for _, attribute, _, _ in DLSS_INPUTS:
            setattr(self, attribute, None)
        self._dlss_color_out_tex = None
        self._dlss_enabled = False
        self._dlss_status_reported = False
        # If DLSS gets disabled at runtime, restore full-resolution rendering.
        if restore_resolution:
            self._set_render_resolution(self.width, self.height)

    def _init_dlss_rr(self):
        self._dlss_init_error = None
        if not self.enable_dlss_rr or self._optix is None:
            self._set_render_resolution(self.width, self.height)
            return

        support_query = getattr(self._optix, "dlss_support_available", None)
        if support_query is None or not support_query():
            self._dlss_init_error = "DLSS support is not compiled into this optix build"
            logger.info("DLSS RR is not compiled into this optix build.")
            self._set_render_resolution(self.width, self.height)
            return

        try:
            context = self._optix.DlssRRContext()
            self._dlss_context = context
            context.init(DLSS_PROJECT_ID)
            if not context.isDlssRRAvailable():
                self._dlss_init_error = "not available on this system"
                logger.info("DLSS RR not available on this system.")
                context.deinit()
                self._dlss_context = None
                self._set_render_resolution(self.width, self.height)
                return

            init_info = self._optix.DlssRRInitInfo()
            render_width = self._half_res(self.width)
            render_height = self._half_res(self.height)
            init_info.inputWidth = int(render_width)
            init_info.inputHeight = int(render_height)
            init_info.outputWidth = int(self.width)
            init_info.outputHeight = int(self.height)
            quality_enum = self._optix.DlssPerfQuality
            quality_name = self.DLSS_QUALITY_MODES[self.dlss_quality]
            if not hasattr(quality_enum, quality_name):
                quality_name = (
                    "MAX_QUALITY"
                    if hasattr(quality_enum, "MAX_QUALITY")
                    else "BALANCED"
                    if hasattr(quality_enum, "BALANCED")
                    else "DLAA"
                )
            init_info.quality = getattr(quality_enum, quality_name)
            preset_enum = self._optix.RayReconstructionHintRenderPreset
            # Preset F selects the DLSS 4.5 Ray Reconstruction transformer
            # (RR2), introduced by SDK 310.9.1.
            init_info.preset = getattr(preset_enum, "F", getattr(preset_enum, "E", preset_enum.DEFAULT))
            # Match reference behavior:
            # - MVJittered=false while still passing per-frame jitter to denoise()
            # - lowResolutionMotionVectors=true (motion vectors provided at render resolution)
            init_info.mvJittered = False
            init_info.lowResolutionMotionVectors = True
            init_info.isContentHDR = True
            init_info.depthInverted = False
            init_info.autoExposure = False
            init_info.useHWDepth = False

            # Ask NGX for the optimal input size for the chosen quality mode,
            # for the selected quality mode, and only fallback to half-res on failure.
            if hasattr(context, "querySupportedDlssInputSizes"):
                try:
                    sizes = context.querySupportedDlssInputSizes(int(self.width), int(self.height), init_info.quality)
                    ow = int(getattr(sizes, "optimalWidth", 0))
                    oh = int(getattr(sizes, "optimalHeight", 0))
                    if ow > 0 and oh > 0:
                        render_width = ow
                        render_height = oh
                        init_info.inputWidth = int(render_width)
                        init_info.inputHeight = int(render_height)
                except Exception as exc:
                    logger.warning(
                        "Failed to query optimal DLSS input size; using half-res fallback: %s",
                        exc,
                    )

            denoiser = context.initDlssRR(init_info, int(self._render_stream.cuda_stream))
            self._set_render_resolution(render_width, render_height)

            self._dlss_context = context
            self._dlss_denoiser = denoiser
            inputs = self._denoiser_inputs()
            for name, attribute, channels, resource in DLSS_INPUTS:
                buffer = getattr(inputs, name)
                texture = self._create_cuda_texture_2d(*buffer.shape, channels)
                setattr(self, attribute, texture)
                denoiser.setResource(getattr(self._optix.DlssRRResource, resource), texture.cuda_texture)
            self._dlss_color_out_tex = self._create_cuda_texture_2d(self.height, self.width, 4, surface_access=True)
            self._dlss_output_surface = self._dlss_color_out_tex.cuda_surface

            denoiser.setResource(self._optix.DlssRRResource.RESOURCE_COLOR_OUT, self._dlss_output_surface)

            self._dlss_context = context
            self._dlss_denoiser = denoiser
            self._dlss_enabled = True
            self._dlss_reset_history = True
            self._dlss_status_reported = False
            logger.info(
                "DLSS Ray Reconstruction enabled (render=%dx%d, output=%dx%d).",
                self._render_width,
                self._render_height,
                self.width,
                self.height,
            )
        except Exception as exc:
            self._dlss_init_error = str(exc)
            logger.warning("Failed to initialize DLSS RR: %s", exc)
            self._destroy_dlss_rr()

    def _copy_linear_to_dlss_textures(self):
        if not self._dlss_enabled:
            return
        inputs = self._denoiser_inputs()
        for name, attribute, _, _ in DLSS_INPUTS:
            getattr(self, attribute).copy_from(getattr(inputs, name))

    def _run_dlss_rr(self, reset: bool):
        if not self._dlss_enabled or self._dlss_denoiser is None:
            return False
        try:
            # Match upstream matrix packing in the Vulkan DLSS-RR sample:
            # output in column-major memory order (m11,m21,m31,m41,...).
            view_m = self.camera.get_view_matrix().astype(np.float32)
            proj_m = self.camera.get_projection_matrix().astype(np.float32)
            view = view_m.T.reshape(-1).tolist()
            proj = proj_m.T.reshape(-1).tolist()
            self._dlss_denoiser.denoise(
                int(self._render_width),
                int(self._render_height),
                float(-self._last_jitter[0]),
                float(-self._last_jitter[1]),
                view,
                proj,
                bool(reset or self._dlss_reset_history),
                0,
                0,
                1.0,
                1.0,
            )
            self._dlss_reset_history = False
            return True
        except Exception as exc:
            logger.warning("DLSS denoise failed; disabling DLSS RR: %s", exc)
            self._destroy_dlss_rr()
            return False

    def _create_buffers(self):
        """Create output buffers."""
        # HDR color buffer
        self._color_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._sky_lut_buffer = wp.empty((1024, 2048), dtype=wp.vec4, device="cuda")
        self._sky_lut_state = None
        self._accum_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")

        # G-buffer outputs
        self._normal_roughness_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.vec4, device="cuda"
        )
        self._motion_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec2, device="cuda")
        self._depth_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.float32, device="cuda")
        self._diffuse_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._specular_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._spec_hit_dist_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.float32, device="cuda"
        )

    def _update_instance_transform_buffers(self):
        """Upload current instance transforms for motion vectors.

        The *previous* transforms buffer is populated by
        :meth:`_snapshot_instance_transforms` which copies the current
        buffer to the previous buffer **after** each rendered frame.
        This guarantees that ``prev`` always holds exactly the transforms
        that were used for the last rendered frame.
        """
        if self._scene is None or self._scene.instance_count == 0:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        count = self._scene.instance_count

        self._instance_transforms_buffer = self._scene._device_instance_transforms
        if self._instance_transforms_buffer is None:
            raise RuntimeError("Scene instance transform buffers have not been built")

        # (Re)allocate GPU buffers when the instance count changes.
        if count != self._instance_transform_count:
            self._prev_instance_transforms_buffer = wp.empty((count, 12), dtype=wp.float32, device="cuda")
            self._instance_transform_count = count
            self._prev_instance_transforms_valid = False

        # First frame (or after resize): prev == current so motion is zero.
        if not self._prev_instance_transforms_valid:
            wp.copy(
                self._prev_instance_transforms_buffer,
                self._instance_transforms_buffer,
                stream=self._render_stream,
            )
            self._prev_instance_transforms_valid = True

    def _snapshot_instance_transforms(self):
        """Copy current instance transforms to the previous-frame buffer on GPU.

        Must be called once per frame **after** the OptiX launch so that the
        next frame sees the correct previous-frame transforms for rigid-body
        motion vectors.
        """
        if (
            self._instance_transforms_buffer is not None
            and self._prev_instance_transforms_buffer is not None
            and self._instance_transforms_buffer.shape == self._prev_instance_transforms_buffer.shape
        ):
            wp.copy(
                self._prev_instance_transforms_buffer,
                self._instance_transforms_buffer,
                stream=self._render_stream,
            )

    def _create_pipeline(self):
        """Create the OptiX pipeline with primary and shadow/visibility ray types.

        Geometry SBT offsets select the hit groups: triangles 0, round linear
        curves 2, and round cubic Bezier curves 4. Bounces iterate in raygen,
        so hit and miss programs never trace recursively.
        """
        linear = native.PRIMITIVE_TYPE_ROUND_LINEAR
        bezier = native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER
        self._pipeline = wo.create_pipeline(
            self._ctx,
            pwk.primary_raygen,
            miss=[pwk.primary_miss, pwk.secondary_miss],
            hit_groups=[
                wo.HitGroup(pwk.primary_closest_hit, pwk.primary_any_hit),
                wo.HitGroup(pwk.secondary_closest_hit, pwk.secondary_any_hit),
                wo.HitGroup(pwk.curve_primary_closest_hit, pwk.primary_any_hit, builtin=linear),
                wo.HitGroup(pwk.secondary_closest_hit, pwk.secondary_any_hit, builtin=linear),
                wo.HitGroup(pwk.curve_primary_closest_hit, pwk.primary_any_hit, builtin=bezier),
                wo.HitGroup(pwk.secondary_closest_hit, pwk.secondary_any_hit, builtin=bezier),
            ],
            ray_types=2,
            payload_words=19,
            graph_flags=native.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            device="cuda",
        )

    @staticmethod
    def _halton(index: int, base: int) -> float:
        f = 1.0
        r = 0.0
        i = max(0, int(index))
        b = max(2, int(base))
        while i > 0:
            f /= float(b)
            r += f * float(i % b)
            i //= b
        return r

    def _compute_camera_basis(self):
        pos = np.asarray(self.camera.position, dtype=np.float32)
        target = np.asarray(self.camera.target, dtype=np.float32)
        world_up = np.asarray(self.camera.up, dtype=np.float32)

        forward = target - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1.0e-8:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= forward_norm

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm

        up = np.cross(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm < 1.0e-8:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            up /= up_norm

        return pos, forward, right, up

    def bind_device_camera(
        self,
        positions: wp.array,
        targets: wp.array,
        *,
        fov: float | wp.array = 45.0,
        up=(0.0, 1.0, 0.0),
        camera_transform: np.ndarray | None = None,
    ) -> None:
        """Bind graph-written CUDA eye and target arrays to the renderer."""
        if positions.dtype != wp.vec3 or targets.dtype != wp.vec3:
            raise TypeError("positions and targets must be wp.array[wp.vec3]")
        if positions.shape[0] < 1 or targets.shape[0] < 1:
            raise ValueError("positions and targets must contain at least one element")
        if positions.device != targets.device or not positions.device.is_cuda:
            raise ValueError("positions and targets must share a CUDA device")
        self._device_camera_positions = positions
        self._device_camera_targets = targets
        self._device_camera_up = wp.array([up], dtype=wp.vec3, device=positions.device)
        if hasattr(fov, "dtype"):
            if fov.dtype != wp.float32 or fov.shape[0] < 1 or fov.device != positions.device:
                raise ValueError("fov must be a non-empty wp.array[float] on the camera device")
            self._device_camera_fovs = fov
        else:
            self._device_camera_fovs = wp.array([float(fov)], dtype=float, device=positions.device)
        transform = np.eye(4, dtype=np.float32) if camera_transform is None else camera_transform
        transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        self._device_camera_transform = wp.mat44(*transform.reshape(-1))
        self._device_camera_initialized = wp.zeros(1, dtype=wp.int32, device=positions.device)
        self._device_camera_state = wp.empty(1, dtype=pwk.DeviceCameraState, device=positions.device)

    def _update_device_camera(self) -> None:
        if self._device_camera_state is None:
            return
        wp.launch(
            _prepare_device_camera,
            dim=1,
            inputs=[
                self._device_camera_positions,
                self._device_camera_targets,
                self._device_camera_up,
                self._device_camera_fovs,
                self._device_camera_transform,
                float(self._render_width) / float(max(1, self._render_height)),
                self._device_camera_initialized,
            ],
            outputs=[self._device_camera_state],
            device=self._device_camera_positions.device,
            stream=self._render_stream,
        )

    def unbind_device_camera(self) -> None:
        """Restore the host-driven camera path."""
        self._device_camera_positions = None
        self._device_camera_targets = None
        self._device_camera_up = None
        self._device_camera_fovs = None
        self._device_camera_initialized = None
        self._device_camera_state = None

    def _update_launch_params(
        self,
        frame_index_override: int | None = None,
        *,
        update_instance_transforms: bool = True,
        view: np.ndarray | None = None,
        proj: np.ndarray | None = None,
        view_inv: np.ndarray | None = None,
        proj_inv: np.ndarray | None = None,
    ):
        """Update launch parameters for the current frame."""
        if update_instance_transforms:
            self._update_instance_transform_buffers()

        if view is None:
            view = self.camera.get_view_matrix()
        if proj is None:
            proj = self.camera.get_projection_matrix()
        if view_inv is None:
            view_inv = np.linalg.inv(view)
        if proj_inv is None:
            proj_inv = np.linalg.inv(proj)

        if self._launch_params is None:
            self._launch_params = pwk.PathtraceLaunchParams()

        p = self._launch_params
        p.tlas = wp.uint64(self._scene.tlas_handle)
        p.width = wp.uint32(self._render_width)
        p.height = wp.uint32(self._render_height)
        frame_index_value = self.sample_index if frame_index_override is None else int(frame_index_override)
        p.frame_index = wp.uint32(frame_index_value)
        p.max_bounces = wp.uint32(self.max_bounces)
        p.direct_light_samples = wp.uint32(self.direct_light_samples)
        p.russian_roulette_start_bounce = wp.uint32(self.russian_roulette_start_bounce)
        p.analytic_light_intensity = float(self.analytic_light_intensity)
        p.emissive_material_intensity = float(self.emissive_material_intensity)
        p.output_mode = int(self.OUTPUT_FINAL)
        p.device_camera = self._device_camera_state
        if self.use_halton_jitter and self._optix_denoiser is None:
            jitter_x = self._halton(frame_index_value, 2) - 0.5
            jitter_y = self._halton(frame_index_value, 3) - 0.5
            self._last_jitter = (float(jitter_x), float(jitter_y))
        else:
            self._last_jitter = (0.0, 0.0)

        pos, forward, right, up = self._compute_camera_basis()
        p.cam_pos = wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        p.cam_forward = wp.vec3(float(forward[0]), float(forward[1]), float(forward[2]))
        p.cam_right = wp.vec3(float(right[0]), float(right[1]), float(right[2]))
        p.cam_up = wp.vec3(float(up[0]), float(up[1]), float(up[2]))
        p.cam_tan_half_fov = float(np.tan(np.deg2rad(self.camera.fov * 0.5)))
        p.cam_aspect = float(self._render_width) / float(max(1, self._render_height))
        p.jitter = wp.vec2(float(self._last_jitter[0]), float(self._last_jitter[1]))
        p.view = tuple(np.asarray(view, dtype=np.float32).reshape(-1))
        p.proj = tuple(np.asarray(proj, dtype=np.float32).reshape(-1))
        p.view_inv = tuple(np.asarray(view_inv, dtype=np.float32).reshape(-1))
        p.proj_inv = tuple(np.asarray(proj_inv, dtype=np.float32).reshape(-1))
        p.prev_mvp = tuple(np.asarray(self._prev_mvp, dtype=np.float32).reshape(-1))
        p.env_intensity = wp.vec3(
            float(self.env_intensity[0]),
            float(self.env_intensity[1]),
            float(self.env_intensity[2]),
        )
        ambient = getattr(self._scene, "usd_ambient_light", (0.0, 0.0, 0.0))
        p.ambient_light = wp.vec3(float(ambient[0]), float(ambient[1]), float(ambient[2]))
        p.env_rotation = float(self.env_rotation)
        flags = 0
        if self._env_map is None:
            flags |= 1
        if self.use_psr:
            flags |= 2
        if self.use_path_regularization:
            flags |= 4
        if self.backface_culling:
            flags |= 8
        p.flags = wp.uint32(flags)
        p.override_roughness = float(self.override_roughness)
        p.override_metallic = float(self.override_metallic)
        p.bitangent_flip = float(self.bitangent_flip)
        p.use_procedural_sky = wp.uint32(1 if self._env_map is None else 0)
        p.env_map = None if self._env_map is None else self._env_map._env_map_buffer
        p.env_map_length = wp.uint32(0 if self._env_map is None else self._env_map.width * self._env_map.height * 4)
        p.env_map_width = wp.uint32(0 if self._env_map is None else self._env_map.width)
        p.env_map_height = wp.uint32(0 if self._env_map is None else self._env_map.height)
        p.env_accel = (
            None
            if self._env_map is None or self._env_map._env_accel_buffer is None
            else wp.array(
                ptr=self._env_map._env_accel_buffer.ptr,
                shape=(self._env_map.accel_count,),
                dtype=pwk.EnvAccel,
            )
        )
        p.env_accel_count = wp.uint32(0 if self._env_map is None else self._env_map.accel_count)

        sky = pwk.PhysicalSkyParams()
        sky.rgb_unit_conversion = wp.vec3(*self.sky_rgb_unit_conversion)
        sky.multiplier = float(self.sky_multiplier)
        sky.haze = float(self.sky_haze)
        sky.redblueshift = float(self.sky_redblueshift)
        sky.saturation = float(self.sky_saturation)
        sky.horizon_height = float(self.sky_horizon_height)
        sky.ground_color = wp.vec3(*self.sky_ground_color)
        sky.horizon_blur = float(self.sky_horizon_blur)
        sky.night_color = wp.vec3(*self.sky_night_color)
        sky.sun_disk_intensity = float(self.sky_sun_disk_intensity)
        sd = self.sky_sun_direction
        sd_len = (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) ** 0.5
        if sd_len > 1.0e-8:
            sd = (sd[0] / sd_len, sd[1] / sd_len, sd[2] / sd_len)
        sky.sun_direction = wp.vec3(*sd)
        sky.sun_disk_scale = float(self.sky_sun_disk_scale)
        sky.sun_glow_intensity = float(self.sky_sun_glow_intensity)
        sky.y_is_up = int(self.sky_y_is_up)
        sky.grayscale = float(self.sky_grayscale)
        p.sky = sky
        sky_state = (
            tuple(self.sky_rgb_unit_conversion),
            float(self.sky_multiplier),
            float(self.sky_haze),
            float(self.sky_redblueshift),
            float(self.sky_saturation),
            float(self.sky_horizon_height),
            tuple(self.sky_ground_color),
            float(self.sky_horizon_blur),
            tuple(self.sky_night_color),
            tuple(sd),
            int(self.sky_y_is_up),
        )
        if self._env_map is None and sky_state != self._sky_lut_state:
            wp.launch(
                generate_physical_sky_lut,
                dim=(self._sky_lut_buffer.shape[1], self._sky_lut_buffer.shape[0]),
                inputs=[sky],
                outputs=[self._sky_lut_buffer],
                device="cuda",
                stream=self._render_stream,
            )
            self._sky_lut_state = sky_state
        p.sky_lut = self._sky_lut_buffer
        p.sky_lut_width = wp.uint32(self._sky_lut_buffer.shape[1])
        p.sky_lut_height = wp.uint32(self._sky_lut_buffer.shape[0])

        p.sphere_lights = (
            None
            if self._scene._sphere_light_data is None
            else wp.array(
                ptr=self._scene._sphere_light_data.ptr,
                shape=(self._scene.light_count,),
                dtype=pwk.SphereLight,
            )
        )
        p.sphere_light_count = wp.uint32(self._scene.light_count)
        p.render_primitives = (
            None
            if self._scene._render_primitives is None or self._scene.geometry_count == 0
            else wp.array(
                ptr=self._scene._render_primitives.ptr,
                shape=(self._scene.geometry_count,),
                dtype=pwk.RenderPrimitive,
            )
        )
        p.render_prim_count = wp.uint32(self._scene.geometry_count)
        p.instance_render_prim_ids = self._scene._instance_render_prim_ids
        p.instance_geometry_types = self._scene._instance_geometry_types
        p.instance_material_ids = self._scene._instance_material_ids
        p.instance_count = wp.uint32(self._scene.instance_count)
        p.instance_transforms = (
            None
            if self._instance_transforms_buffer is None
            else wp.array(
                ptr=self._instance_transforms_buffer.ptr,
                shape=(self._scene.instance_count,),
                dtype=pwk.TransformMatrix3x4,
            )
        )
        p.prev_instance_transforms = (
            None
            if self._prev_instance_transforms_buffer is None
            else wp.array(
                ptr=self._prev_instance_transforms_buffer.ptr,
                shape=(self._scene.instance_count,),
                dtype=pwk.TransformMatrix3x4,
            )
        )
        p.compact_materials = (
            None
            if self._scene._compact_materials is None or self._scene.materials.count == 0
            else wp.array(
                ptr=self._scene._compact_materials.ptr,
                shape=(self._scene.materials.count,),
                dtype=pwk.CompactMaterial,
            )
        )
        p.packed_indices = self._scene._packed_indices
        p.packed_positions = self._scene._packed_positions
        p.packed_radii = self._scene._packed_radii
        p.packed_normals = self._scene._packed_normals
        p.packed_tangents = self._scene._packed_tangents
        p.packed_texcoords0 = self._scene._packed_texcoords0
        p.packed_texcoords1 = self._scene._packed_texcoords1
        p.packed_prev_positions = self._scene._packed_prev_positions
        p.packed_material_ids = self._scene._packed_material_ids
        p.material_count = wp.uint32(self._scene.materials.count)
        p.textures = self._scene._texture_data
        p.texture_count = wp.uint32(self._scene.texture_count)

        p.color_output = self._color_buffer
        p.normal_roughness_output = self._normal_roughness_buffer
        p.motion_output = self._motion_buffer
        p.depth_output = self._depth_buffer
        p.diffuse_output = self._diffuse_buffer
        p.specular_output = self._specular_buffer
        p.spec_hit_dist_output = self._spec_hit_dist_buffer

    def _update_temporal_state(
        self,
        current_view: np.ndarray,
        current_proj: np.ndarray,
        use_external_accum: bool,
    ) -> bool:
        """Update accumulation state and return whether temporal history resets."""
        if self._dlss_enabled or self._optix_denoiser is not None:
            return False

        reset_temporal = (
            self.output_mode != self._last_output_mode
            or (not np.allclose(current_view, self._last_accum_view))
            or (not np.allclose(current_proj, self._last_accum_proj))
        )

        if use_external_accum:
            if reset_temporal:
                wp.launch(
                    _reset_accum_buffer,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._accum_buffer],
                    device="cuda",
                    stream=self._render_stream,
                )
                self.frame_index = 0
            return bool(reset_temporal)

        # No persistent external accumulation for this mode.
        wp.launch(
            _reset_accum_buffer,
            dim=(self._render_width, self._render_height),
            inputs=[self._accum_buffer],
            device="cuda",
            stream=self._render_stream,
        )
        return bool(reset_temporal)

    def _launch_samples(
        self,
        samples_this_frame: int,
        use_external_accum: bool,
        *,
        view: np.ndarray,
        proj: np.ndarray,
        view_inv: np.ndarray,
        proj_inv: np.ndarray,
    ):
        """Launch OptiX path tracing and optional external accumulation kernels."""
        for s in range(samples_this_frame):
            launch_frame_index = self.sample_index + s
            self._update_launch_params(
                frame_index_override=launch_frame_index,
                update_instance_transforms=False,
                view=view,
                proj=proj,
                view_inv=view_inv,
                proj_inv=proj_inv,
            )

            self._launch_optix()

            if self._volume is not None:
                p = VolumeParams()
                p.volume = wp.uint64(self._volume.id)
                p.bounds_min = wp.vec3(*self._volume_min)
                p.bounds_max = wp.vec3(*self._volume_max)
                p.density_scale = self._volume_density_scale
                p.step_size = self._volume_step_size
                p.cool_dark = wp.vec3(*self._volume_cool_colors[0])
                p.cool_mid = wp.vec3(*self._volume_cool_colors[1])
                p.cool_light = wp.vec3(*self._volume_cool_colors[2])
                p.warm_dark = wp.vec3(*self._volume_warm_colors[0])
                p.warm_mid = wp.vec3(*self._volume_warm_colors[1])
                p.warm_light = wp.vec3(*self._volume_warm_colors[2])
                p.emission = self._volume_emission
                p.anisotropy = self._volume_anisotropy
                p.transfer_table = self._volume_transfer_table
                p.transfer_count = wp.uint32(
                    0 if self._volume_transfer_table is None else len(self._volume_transfer_table)
                )
                p.density_feature = wp.uint32(self._volume_density_feature)
                self._volume_params = p
                wp.launch(
                    composite_volume,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._launch_params, self._volume_params],
                    device="cuda",
                    stream=self._render_stream,
                )

            if not self._dlss_enabled:
                accum_sample_index = int(self.frame_index if use_external_accum else s)
                wp.launch(
                    _accumulate_sample,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._color_buffer, self._accum_buffer, accum_sample_index],
                    device="cuda",
                    stream=self._render_stream,
                )

                if use_external_accum:
                    self.frame_index += 1

    def _launch_optix(self):
        wo.launch(
            self._pipeline,
            dim=(self._render_height, self._render_width),
            inputs=[self._launch_params],
            stream=self._render_stream,
        )

    def _process_debug_output(self):
        self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process_debug(
            self.output_mode,
            self._color_buffer,
            self._depth_buffer,
            self._motion_buffer,
            self._normal_roughness_buffer,
            self._diffuse_buffer,
            self._specular_buffer,
            self._spec_hit_dist_buffer,
            self._render_width,
            self._render_height,
        )

    def _process_final_output(self, source_buffer, *, resize_to_render: bool):
        if resize_to_render:
            self._tonemapper.resize(self._render_width, self._render_height)
        else:
            self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process(source_buffer)

    def _process_output(self, source_buffer, *, resize_final_to_render: bool):
        if self.output_mode == self.OUTPUT_FINAL:
            self._process_final_output(source_buffer, resize_to_render=resize_final_to_render)
            return
        self._process_debug_output()

    def render(self):
        """Render a frame."""
        # Texture transfers and tone mapping use Warp's current stream.
        # Order them with OptiX/DLSS, including calls from another stream.
        with wp.ScopedStream(self._render_stream):
            self._render_frame()

    def _render_frame(self):
        if self._pipeline is None:
            logger.error("Pipeline not built. Call build() first.")
            return

        if not self._dlss_status_reported:
            logger.info(
                "Denoiser active: %s (requested: %s).",
                self.active_denoiser,
                self.denoiser,
            )
            if self.denoiser_error:
                logger.warning("Denoiser fallback: %s", self.denoiser_error)
            self._dlss_status_reported = True
        self._update_device_camera()

        current_view = self.camera.get_view_matrix().copy()
        current_proj = self.camera.get_projection_matrix().copy()
        current_view_inv = np.linalg.inv(current_view)
        current_proj_inv = np.linalg.inv(current_proj)
        use_external_accum = (
            self.accumulate_samples
            and not self._dlss_enabled
            and self._optix_denoiser is None
            and self._device_camera_state is None
        )
        samples_this_frame = 1 if self._dlss_enabled else self.samples_per_frame
        reset_temporal = self._update_temporal_state(current_view, current_proj, use_external_accum)
        self._update_instance_transform_buffers()
        self._launch_samples(
            samples_this_frame,
            use_external_accum,
            view=current_view,
            proj=current_proj,
            view_inv=current_view_inv,
            proj_inv=current_proj_inv,
        )

        # Queue the current-to-previous transform snapshot behind the OptiX launch.
        self._snapshot_instance_transforms()

        # Keep previous matrices for next frame's motion-vector calculation.
        self._prev_view = current_view.copy()
        self._prev_proj = current_proj.copy()
        self._prev_mvp = (current_view @ current_proj).astype(np.float32)
        self._last_accum_view = current_view.copy()
        self._last_accum_proj = current_proj.copy()
        self._last_output_mode = self.output_mode

        if self._dlss_enabled:
            # OptiX, texture copies, DLSS, and tone mapping share one stream;
            # stream ordering publishes every producer to its consumer.
            self._copy_linear_to_dlss_textures()
            if self._run_dlss_rr(reset_temporal):
                self._process_output(self._dlss_color_out_tex, resize_final_to_render=False)
            else:
                self._process_output(self._color_buffer, resize_final_to_render=True)
        elif self._optix_denoiser is not None:
            output = self._optix_denoiser.apply(
                self._denoiser_inputs(self._accum_buffer),
                reset=reset_temporal or self._dlss_reset_history,
            )
            self._dlss_reset_history = False
            self._process_output(output, resize_final_to_render=False)
        else:
            self._process_output(self._accum_buffer, resize_final_to_render=True)
        self.sample_index += samples_this_frame

    def get_output(self) -> np.ndarray:
        """Get the current output as a numpy array."""
        wp.synchronize_stream(self._render_stream)
        return self._tonemapper.get_numpy()

    def resize(self, width: int, height: int):
        """Resize the render buffers."""
        if width != self.width or height != self.height:
            wp.synchronize_stream(self._render_stream)
            # Tear down resources tied to the old dimensions first.  Native
            # window/output resolution remains uncapped; this only avoids a
            # transient old+new allocation spike during resize.
            self._destroy_dlss_rr(restore_resolution=False)
            self._destroy_optix_denoiser()
            self.width = width
            self.height = height
            self.camera.set_aspect_ratio(width, height)
            self._sync_prev_camera_matrices_to_current()
            self._init_denoiser()
            self._tonemapper.resize(width, height)
            self.frame_index = 0

    def close(self):
        """Wait for rendering and release reconstruction resources."""
        wp.synchronize_stream(self._render_stream)
        self._destroy_dlss_rr(restore_resolution=False)
        self._destroy_optix_denoiser()
        # Destroying the context releases its pipelines and modules. GPU buffers
        # are dropped afterwards so OptiX never sees them freed while in use.
        if self._ctx is not None:
            wo.destroy_context(self._ctx)
        self._ctx = None
        self._pipeline = None
        self._scene = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def main():
    """Run the path tracing viewer."""
    logger.info("%s", "=" * 60)
    logger.info("OptiX Path Tracing Viewer")
    logger.info("%s", "=" * 60)

    viewer = PathTracingViewer(width=800, height=600)

    if not viewer.build():
        logger.error("Failed to build viewer.")
        return 1

    # Render a few frames
    logger.info("Rendering frames.")
    for i in range(10):
        viewer.render()
        logger.info("Frame %d", i + 1)

    # Get final output
    output = viewer.get_output()
    logger.info("Output shape: %s", output.shape)
    logger.info("Output range: [%.3f, %.3f]", float(output.min()), float(output.max()))

    # Save to file if possible
    if Image is not None:
        img_data = (output[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save("pathtracing_output.png")
        logger.info("Saved output to pathtracing_output.png")
    else:
        logger.info("Pillow not installed; skipping image save.")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
