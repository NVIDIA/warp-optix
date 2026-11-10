# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A physically based OptiX path tracer with denoising and an interactive viewer.

Requires the ``pathtracing`` extra. See ``docs/pathtracing.md`` for a guide.
"""

from warp_optix._src.pathtracing.arrows import ArrowBatch as ArrowBatch
from warp_optix._src.pathtracing.camera import Camera as Camera
from warp_optix._src.pathtracing.defaults import DEFAULT_VIEWER_HEIGHT as DEFAULT_VIEWER_HEIGHT
from warp_optix._src.pathtracing.defaults import DEFAULT_VIEWER_SIZE as DEFAULT_VIEWER_SIZE
from warp_optix._src.pathtracing.defaults import DEFAULT_VIEWER_WIDTH as DEFAULT_VIEWER_WIDTH
from warp_optix._src.pathtracing.denoising import DenoiserInputs as DenoiserInputs
from warp_optix._src.pathtracing.denoising import OptixDenoiser as OptixDenoiser
from warp_optix._src.pathtracing.materials import MaterialManager as MaterialManager
from warp_optix._src.pathtracing.pathtracer_api import PathTracer as PathTracer
from warp_optix._src.pathtracing.pathtracing_viewer import Renderer as Renderer
from warp_optix._src.pathtracing.scene import Curve as Curve
from warp_optix._src.pathtracing.scene import Mesh as Mesh
from warp_optix._src.pathtracing.scene import Scene as Scene
from warp_optix._src.pathtracing.tonemap import Tonemapper as Tonemapper
from warp_optix._src.pathtracing.usd_scene import UsdScene as UsdScene
from warp_optix._src.pathtracing.usd_scene import UsdTransformHandle as UsdTransformHandle
from warp_optix._src.pathtracing.viewer import Viewer as Viewer
from warp_optix._src.pathtracing.viewer import ViewerBackend as ViewerBackend

__all__ = [
    "DEFAULT_VIEWER_HEIGHT",
    "DEFAULT_VIEWER_SIZE",
    "DEFAULT_VIEWER_WIDTH",
    "ArrowBatch",
    "Camera",
    "Curve",
    "DenoiserInputs",
    "MaterialManager",
    "Mesh",
    "OptixDenoiser",
    "PathTracer",
    "Renderer",
    "Scene",
    "Tonemapper",
    "UsdScene",
    "UsdTransformHandle",
    "Viewer",
    "ViewerBackend",
]
