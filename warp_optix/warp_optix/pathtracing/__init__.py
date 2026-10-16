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
OptiX Path Tracing Package.

Python/OptiX path tracing components providing hardware-accelerated ray
tracing with PBR materials.

Components:
    - camera: First-person/orbit camera with matrix generation
    - materials: PBR material management
    - scene: Mesh and acceleration structure management
    - tonemap: HDR to LDR tonemapping
    - pathtracing_viewer: OptiX rendering engine
    - viewer: Standalone and framework-compatible viewer API
"""

from .arrows import ArrowBatch
from .camera import Camera
from .defaults import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_SIZE,
    DEFAULT_VIEWER_WIDTH,
)
from .denoising import DenoiserInputs, OptixDenoiser
from .materials import MaterialManager
from .pathtracer_api import PathTracerAPI
from .pathtracing_viewer import PathTracingViewer as PathTracingRenderer
from .scene import Curve, Mesh, Scene
from .tonemap import Tonemapper
from .usd_scene import USDScene, USDTransformHandle
from .viewer import PathTracingViewer, PathTracingViewerBackend

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
    "PathTracerAPI",
    "PathTracingRenderer",
    "PathTracingViewer",
    "PathTracingViewerBackend",
    "Scene",
    "Tonemapper",
    "USDScene",
    "USDTransformHandle",
]
