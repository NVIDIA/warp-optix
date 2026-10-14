# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp ray tracing using PyOptiX and supplemental native bindings."""

from ._addon import register_with_warp as _register_with_warp

_register_with_warp()

__version__ = "0.2.0"

from ._api import Context as Context
from ._api import AccelerationStructure as AccelerationStructure
from ._api import Pipeline as Pipeline
from ._api import Instance as Instance
from ._api import HitKernel as HitKernel
from ._sbt import SbtBuilder as SbtBuilder
from ._sbt import SbtLayout as SbtLayout
from ._sbt import HitGroupHandle as HitGroupHandle
from ._enums import GeometryFlags as GeometryFlags
from ._enums import PrimitiveType as PrimitiveType
from ._enums import PrimitiveTypeFlags as PrimitiveTypeFlags
from ._enums import InstanceFlags as InstanceFlags
from ._enums import BuildFlags as BuildFlags
from ._enums import TraversableGraphFlags as TraversableGraphFlags
from ._enums import ExceptionFlags as ExceptionFlags
from ._codegen import OptixKernelType as OptixKernelType
from ._codegen import optix_kernel as optix_kernel
from ._runtime.gl_interop import OptixGLInteropViewer as GLInteropViewer
from ._runtime.gl_line_overlay import GLLineOverlay as GLLineOverlay
from ._runtime.runtime import LaunchParamsBuffer as LaunchParamsBuffer
from ._runtime.runtime import compile_warp_module_to_ptx as compile_warp_module_to_ptx
from ._runtime.runtime import create_launch_params_buffer as create_launch_params_buffer
from ._runtime.runtime import get_optix_entry_name as get_entry_name
from ._runtime.runtime import write_launch_params as write_launch_params

__all__ = [
    "Context",
    "AccelerationStructure",
    "Pipeline",
    "Instance",
    "HitKernel",
    "SbtBuilder",
    "SbtLayout",
    "HitGroupHandle",
    "GeometryFlags",
    "PrimitiveType",
    "PrimitiveTypeFlags",
    "InstanceFlags",
    "BuildFlags",
    "TraversableGraphFlags",
    "ExceptionFlags",
    "OptixKernelType",
    "optix_kernel",
    "GLInteropViewer",
    "GLLineOverlay",
    "LaunchParamsBuffer",
    "compile_warp_module_to_ptx",
    "create_launch_params_buffer",
    "get_entry_name",
    "write_launch_params",
]
