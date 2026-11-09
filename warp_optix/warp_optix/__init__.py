# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX ray tracing for Warp.

Acceleration structures are built from Warp device arrays, OptiX programs are
ordinary Warp kernels, and launches mirror ``wp.launch``::

    import warp_optix as wo

    accel = wo.Accel([wo.triangles(points, indices)])
    pipeline = wo.Pipeline(render)
    wo.launch(pipeline, dim=(height, width), inputs=[accel.id, image])

Like Warp's CUDA context, the OptiX context is implicit: one per device,
created on first use with the settings in :mod:`warp_optix.config`.
``warp_optix.native`` gives advanced access to PyOptiX and the supplemental bindings.
"""

from warp_optix.addon import register_with_warp

register_with_warp()

from warp_optix import config as config
from warp_optix._version import __version__ as __version__
from warp_optix.accel import Accel as Accel
from warp_optix.accel import BuildInput as BuildInput
from warp_optix.accel import Instance as Instance
from warp_optix.accel import aabbs as aabbs
from warp_optix.accel import compact as compact
from warp_optix.accel import curves as curves
from warp_optix.accel import instances as instances
from warp_optix.accel import make_instance as make_instance
from warp_optix.accel import triangles as triangles
from warp_optix.context import get_context as get_context
from warp_optix.enums import BuildFlags as BuildFlags
from warp_optix.enums import CompileDebugLevel as CompileDebugLevel
from warp_optix.enums import CompileOptimizationLevel as CompileOptimizationLevel
from warp_optix.enums import ExceptionFlags as ExceptionFlags
from warp_optix.enums import GeometryFlags as GeometryFlags
from warp_optix.enums import HitKind as HitKind
from warp_optix.enums import InstanceFlags as InstanceFlags
from warp_optix.enums import PrimitiveType as PrimitiveType
from warp_optix.enums import RayFlags as RayFlags
from warp_optix.enums import TraversableGraphFlags as TraversableGraphFlags
from warp_optix.pipeline import HitGroup as HitGroup
from warp_optix.pipeline import Launch as Launch
from warp_optix.pipeline import Pipeline as Pipeline
from warp_optix.pipeline import launch as launch
from warp_optix.program import ProgramKind as ProgramKind
from warp_optix.program import kernel as kernel
from warp_optix.rays import AccelQueryRay as AccelQueryRay
from warp_optix.rays import accel_query_ray as accel_query_ray
from warp_optix.rays import accel_query_ray_anyhit as accel_query_ray_anyhit
from warp_optix.sbt import HitGroupHandle as HitGroupHandle
from warp_optix.sbt import SbtBuilder as SbtBuilder
from warp_optix.sbt import SbtLayout as SbtLayout

__all__ = [
    "Accel",
    "AccelQueryRay",
    "BuildFlags",
    "BuildInput",
    "CompileDebugLevel",
    "CompileOptimizationLevel",
    "ExceptionFlags",
    "GeometryFlags",
    "HitGroup",
    "HitGroupHandle",
    "HitKind",
    "Instance",
    "InstanceFlags",
    "Launch",
    "Pipeline",
    "PrimitiveType",
    "ProgramKind",
    "RayFlags",
    "SbtBuilder",
    "SbtLayout",
    "TraversableGraphFlags",
    "aabbs",
    "compact",
    "accel_query_ray",
    "accel_query_ray_anyhit",
    "config",
    "curves",
    "get_context",
    "instances",
    "kernel",
    "launch",
    "make_instance",
    "triangles",
]
