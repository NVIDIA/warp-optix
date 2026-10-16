# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX ray tracing for Warp.

Acceleration structures are built from Warp device arrays, OptiX programs are
ordinary Warp kernels, and launches mirror ``wp.launch``::

    import warp_optix as wo

    ctx = wo.create_context()
    gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)])
    pipeline = wo.create_pipeline(ctx, raygen=render)
    wo.launch(pipeline, dim=(height, width), inputs=[gas.handle, image])
    wo.destroy_context(ctx)

``warp_optix.native`` gives advanced access to PyOptiX and the supplemental bindings.
"""

from warp_optix.addon import register_with_warp

register_with_warp()

__version__ = "0.2.0"

from warp_optix import constants as constants
from warp_optix.accel import Accel as Accel
from warp_optix.accel import Instance as Instance
from warp_optix.accel import aabbs as aabbs
from warp_optix.accel import build_accel as build_accel
from warp_optix.accel import curves as curves
from warp_optix.accel import instances as instances
from warp_optix.accel import make_instance as make_instance
from warp_optix.accel import triangles as triangles
from warp_optix.accel import update_accel as update_accel
from warp_optix.enums import BuildFlags as BuildFlags
from warp_optix.enums import GeometryFlags as GeometryFlags
from warp_optix.enums import InstanceFlags as InstanceFlags
from warp_optix.enums import PrimitiveType as PrimitiveType
from warp_optix.enums import TraversableGraphFlags as TraversableGraphFlags
from warp_optix.pipeline import HitGroup as HitGroup
from warp_optix.pipeline import Pipeline as Pipeline
from warp_optix.pipeline import create_context as create_context
from warp_optix.pipeline import create_pipeline as create_pipeline
from warp_optix.pipeline import destroy_context as destroy_context
from warp_optix.pipeline import launch as launch
from warp_optix.program import ProgramKind as ProgramKind
from warp_optix.rays import MISS as MISS
from warp_optix.rays import Hit as Hit
from warp_optix.rays import any_hit as any_hit
from warp_optix.rays import closest_hit as closest_hit

__all__ = [
    "MISS",
    "Accel",
    "BuildFlags",
    "GeometryFlags",
    "Hit",
    "HitGroup",
    "Instance",
    "InstanceFlags",
    "Pipeline",
    "PrimitiveType",
    "ProgramKind",
    "TraversableGraphFlags",
    "aabbs",
    "any_hit",
    "build_accel",
    "closest_hit",
    "constants",
    "create_context",
    "create_pipeline",
    "curves",
    "destroy_context",
    "instances",
    "launch",
    "make_instance",
    "triangles",
    "update_accel",
]
