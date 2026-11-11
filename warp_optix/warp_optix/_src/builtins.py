# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX device builtins registered with Warp.

Every builtin maps onto a wrapper in ``warp_optix_builtins.h`` and is only
valid inside programs compiled by :class:`warp_optix.Pipeline`. Flags, masks,
SBT offsets, and register indices accept any integer type, so
:class:`warp_optix.RayFlags` members can be passed without a cast.

Warp's external-builtin API has no default arguments, so builtins with
optional trailing arguments register one overload per argument count;
trailing arguments may be omitted, but not skipped by keyword.
"""

from __future__ import annotations

import re
from typing import Any

import warp as wp

_mat24 = wp.types.matrix(shape=(2, 4), dtype=wp.float32)
_mat34 = wp.types.matrix(shape=(3, 4), dtype=wp.float32)
_Int = wp.Int

# Argument-free state queries: name -> (return type, doc).
_QUERIES = {
    "optix_get_cluster_id": (wp.uint32, "User-defined cluster ID of the current hit."),
    "optix_hit_object_get_cluster_id": (wp.uint32, "User-defined cluster ID of the hit object."),
    "optix_get_launch_index": (wp.vec3ui, "Launch index (x, y, z) of this thread."),
    "optix_get_launch_dimensions": (wp.vec3ui, "Launch dimensions (x, y, z)."),
    "optix_get_world_ray_origin": (wp.vec3, "World-space origin of the current ray."),
    "optix_get_world_ray_direction": (wp.vec3, "World-space direction of the current ray."),
    "optix_get_object_ray_origin": (wp.vec3, "Object-space origin of the current ray."),
    "optix_get_object_ray_direction": (wp.vec3, "Object-space direction of the current ray."),
    "optix_get_ray_tmin": (wp.float32, "Minimum distance of the current ray."),
    "optix_get_ray_tmax": (wp.float32, "Maximum distance of the current ray; the hit distance in closest-hit."),
    "optix_get_ray_time": (wp.float32, "Motion-blur time of the current ray."),
    "optix_get_ray_flags": (wp.uint32, "RayFlags of the current ray."),
    "optix_get_ray_visibility_mask": (wp.uint32, "Visibility mask of the current ray."),
    "optix_get_triangle_barycentrics": (wp.vec2, "Barycentric (u, v) of the current triangle hit."),
    "optix_get_triangle_vertex_data": (wp.mat33, "Object-space vertices of the hit triangle, one per row."),
    "optix_is_triangle_hit": (wp.bool, "Whether the current hit is a built-in triangle."),
    "optix_is_triangle_front_face_hit": (wp.bool, "Whether the current hit is a triangle front face."),
    "optix_is_triangle_back_face_hit": (wp.bool, "Whether the current hit is a triangle back face."),
    "optix_is_front_face_hit": (wp.bool, "Whether the current hit is a front face."),
    "optix_is_back_face_hit": (wp.bool, "Whether the current hit is a back face."),
    "optix_get_linear_curve_vertex_data": (_mat24, "Control points (xyz, width) of the hit linear curve segment."),
    "optix_get_quadratic_bspline_vertex_data": (_mat34, "Control points of the hit quadratic B-spline segment."),
    "optix_get_ribbon_vertex_data": (_mat34, "Control points of the hit flat quadratic B-spline (ribbon) segment."),
    "optix_get_cubic_bspline_vertex_data": (wp.mat44, "Control points of the hit cubic B-spline segment."),
    "optix_get_catmull_rom_vertex_data": (wp.mat44, "Control points of the hit Catmull-Rom segment."),
    "optix_get_cubic_bezier_vertex_data": (wp.mat44, "Control points of the hit cubic Bezier segment."),
    "optix_get_ribbon_parameters": (wp.vec2, "Surface parameters (u, v) of the current ribbon hit."),
    "optix_get_curve_parameter": (wp.float32, "Curve parameter of the current curve hit."),
    "optix_get_primitive_index": (wp.uint32, "Primitive index of the current hit within its build input."),
    "optix_get_instance_id": (wp.uint32, "User instance id of the current hit."),
    "optix_get_instance_index": (wp.uint32, "Index of the current hit's instance in its instance build input."),
    "optix_get_sbt_gas_index": (wp.uint32, "SBT index of the hit geometry within its acceleration structure."),
    "optix_get_primitive_type": (wp.uint32, "PrimitiveType of the current hit."),
    "optix_get_gas_traversable_handle": (wp.uint64, "Id of the geometry acceleration structure of the current hit."),
    "optix_get_hit_kind": (wp.uint32, "Hit kind of the current hit (HitKind for triangles, user value for custom)."),
    "optix_get_object_to_world_matrix": (wp.mat44, "Object-to-world matrix of the current hit."),
    "optix_get_world_to_object_matrix": (wp.mat44, "World-to-object matrix of the current hit."),
    "optix_get_transform_list_size": (wp.uint32, "Number of transforms between the current hit and the root."),
    "optix_get_exception_code": (wp.int32, "Code of the exception being handled."),
    # Hit-object queries, valid after optix_traverse() in raygen, closest-hit, or miss.
    "optix_hit_object_is_hit": (wp.bool, "Whether the hit object holds a hit."),
    "optix_hit_object_is_miss": (wp.bool, "Whether the hit object holds a miss."),
    "optix_hit_object_get_world_ray_origin": (wp.vec3, "World-space ray origin stored in the hit object."),
    "optix_hit_object_get_world_ray_direction": (wp.vec3, "World-space ray direction stored in the hit object."),
    "optix_hit_object_get_ray_tmin": (wp.float32, "Minimum ray distance stored in the hit object."),
    "optix_hit_object_get_ray_tmax": (wp.float32, "Hit distance stored in the hit object."),
    "optix_hit_object_get_primitive_index": (wp.uint32, "Primitive index stored in the hit object."),
    "optix_hit_object_get_instance_id": (wp.uint32, "User instance id stored in the hit object."),
    "optix_hit_object_get_instance_index": (wp.uint32, "Instance index stored in the hit object."),
    "optix_hit_object_get_sbt_gas_index": (wp.uint32, "Geometry SBT index stored in the hit object."),
    "optix_hit_object_get_hit_kind": (wp.uint32, "Hit kind stored in the hit object."),
    "optix_hit_object_get_triangle_barycentrics": (wp.vec2, "Triangle barycentrics stored in the hit object."),
    "optix_hit_object_get_curve_parameter": (wp.float32, "Curve parameter stored in the hit object (curve hits only)."),
    "optix_hit_object_get_ribbon_parameters": (wp.vec2, "Ribbon (u, v) stored in the hit object (ribbon hits only)."),
    "optix_hit_object_get_object_to_world_matrix": (wp.mat44, "Object-to-world matrix of the hit object."),
    "optix_hit_object_get_world_to_object_matrix": (wp.mat44, "World-to-object matrix of the hit object."),
}

_DEPRECATED_NUMBERED = "Deprecated numbered form of ``{base}(index)``; prefer the indexed builtin."

# Builtins with arguments: name -> (input types, return type, doc).
_FUNCTIONS = {
    "optix_float_as_uint32": ({"x": wp.float32}, wp.uint32, "Reinterpret the bits of a float32 as uint32."),
    "optix_uint32_as_float": ({"u": wp.uint32}, wp.float32, "Reinterpret the bits of a uint32 as float32."),
    "optix_get_ribbon_normal": ({"parameters": wp.vec2}, wp.vec3, "Object-space ribbon normal at (u, v)."),
    "optix_get_transform_list_handle": ({"index": _Int}, wp.uint64, "Handle of transform ``index`` on the path."),
    "optix_get_transform_type_from_handle": ({"handle": wp.uint64}, wp.uint32, "Transform type of a handle."),
    "optix_get_primitive_type_from_hit_kind": (
        {"hit_kind": _Int},
        wp.uint32,
        "PrimitiveType of a hit kind, e.g. ``optix_hit_object_get_hit_kind()``.",
    ),
    "optix_terminate_ray": ({}, None, "Stop traversal of the current ray (any-hit only)."),
    "optix_ignore_intersection": ({}, None, "Discard the current intersection (any-hit only)."),
    "optix_direct_call": ({"sbt_index": _Int}, None, "Call the direct-callable program at ``sbt_index``."),
    "optix_continuation_call": ({"sbt_index": _Int}, None, "Call the continuation-callable program at ``sbt_index``."),
    "optix_load_payload": ({"payload": Any}, None, "Read the payload registers into a Warp value."),
    "optix_store_payload": ({"payload": Any}, None, "Write a Warp value into the payload registers."),
    "optix_load_sbt_data": (
        {"data": Any},
        None,
        "Read this program's SBT record data into a Warp struct of the type given as ``data=`` on the host.",
    ),
    "optix_get_payload": ({"index": _Int}, wp.uint32, "Read payload register ``index`` (0-31; use a constant)."),
    "optix_set_payload": ({"index": _Int, "value": wp.uint32}, None, "Write payload register ``index`` (0-31)."),
    "optix_get_attribute": ({"index": _Int}, wp.uint32, "Read intersection attribute ``index`` (0-7)."),
    "optix_hit_object_get_attribute": ({"index": _Int}, wp.uint32, "Read hit-object attribute ``index`` (0-7)."),
    "optix_get_exception_detail": ({"index": _Int}, wp.uint32, "Read exception detail ``index`` (0-7)."),
    **{
        f"optix_get_payload_{i}": ({}, wp.uint32, _DEPRECATED_NUMBERED.format(base="optix_get_payload"))
        for i in range(32)
    },
    **{
        f"optix_set_payload_{i}": ({"value": wp.uint32}, None, _DEPRECATED_NUMBERED.format(base="optix_set_payload"))
        for i in range(32)
    },
    **{
        f"optix_get_attribute_{i}": ({}, wp.uint32, _DEPRECATED_NUMBERED.format(base="optix_get_attribute"))
        for i in range(8)
    },
    **{
        f"optix_hit_object_get_attribute_{i}": (
            {},
            wp.uint32,
            _DEPRECATED_NUMBERED.format(base="optix_hit_object_get_attribute"),
        )
        for i in range(8)
    },
    **{
        f"optix_get_exception_detail_{i}": (
            {},
            wp.uint32,
            _DEPRECATED_NUMBERED.format(base="optix_get_exception_detail"),
        )
        for i in range(8)
    },
    **{
        f"optix_{space}transform_{kind}_{src}_to_{dst}": (
            {kind: wp.vec3},
            wp.vec3,
            f"Transform a {kind} from {src} to {dst} space"
            + (" using the hit object's transforms." if space else " at the current hit."),
        )
        for space in ("", "hit_object_")
        for kind in ("point", "vector", "normal")
        for src, dst in (("object", "world"), ("world", "object"))
        if not (space and src == "world")
    },
}

# Required ray arguments, the optional tail (defaults live in the C++ wrapper), and the payload.
_RAY = {"id": wp.uint64, "start": wp.vec3, "dir": wp.vec3, "max_t": wp.float32}
_RAY_TAIL = {
    "min_t": wp.float32,
    "time": wp.float32,
    "mask": _Int,
    "flags": _Int,
    "sbt_offset": _Int,
    "sbt_stride": _Int,
    "miss_index": _Int,
}
_TRACE_DOC = (
    "Trace a ray against acceleration structure ``id``, invoking hit and miss programs. Optional trailing "
    "arguments default to min_t=0, time=0, mask=0xFF, flags=RayFlags.NONE, sbt_offset=0, sbt_stride=1, "
    "miss_index=0. ``payload`` is any Warp value of 1-32 words, updated in place."
)
_TRAVERSE_DOC = (
    "Traverse like optix_trace without invoking hit or miss programs; read the result with the "
    "``optix_hit_object_*`` builtins, or run the programs with optix_invoke()."
)


def _tails():
    names = list(_RAY_TAIL)
    for n in range(len(names) + 1):
        yield {name: _RAY_TAIL[name] for name in names[:n]}


_COOP_DOC = (
    "OptiX cooperative-vector operation. The output is an explicit Warp vector argument. Named matrix "
    "operations use the matching element interpretation, inference-optimal layout, and a non-transposed matrix."
)


def _coop_vec_builtins():
    yield "optix_coop_vec_load", {"address": wp.uint64, "output": Any}, None
    yield "optix_coop_vec_cvt", {"input": Any, "output": Any}, None
    for name in ("exp2", "log2", "tanh"):
        yield f"optix_coop_vec_{name}", {"input": Any, "output": Any}, None
    for name in ("min", "max", "mul", "add", "sub", "step", "min_scalar", "max_scalar"):
        yield f"optix_coop_vec_{name}", {"a": Any, "b": Any, "output": Any}, None
    yield "optix_coop_vec_ffma", {"a": Any, "b": Any, "c": Any, "output": Any}, None
    for elem in ("fp16", "fp8_e4m3", "fp8_e5m2", "int8", "uint8"):
        matrix = {"matrix": wp.uint64, "matrix_offset": wp.uint32}
        stride = {"row_column_stride": wp.uint32, "output": Any}
        yield f"optix_coop_vec_matmul_{elem}", {"input": Any} | matrix | stride, None
        bias = {"bias": wp.uint64, "bias_offset": wp.uint32}
        yield f"optix_coop_vec_matmul_bias_{elem}", {"input": Any} | matrix | bias | stride, None
        yield f"optix_coop_vec_matrix_size_{elem}", {"input_shape": Any, "output_shape": Any}, wp.uint32
    yield "optix_coop_vec_reduce_sum_accumulate", {"input": Any, "output": wp.uint64, "offset": wp.uint32}, None
    yield (
        "optix_coop_vec_outer_product_accumulate",
        {"a": Any, "b": Any, "output": wp.uint64, "offset": wp.uint32},
        None,
    )


def register_addon_builtins() -> None:
    """Register every OptiX builtin with Warp."""
    add = wp.build_experimental.add_builtin
    for name, (value_type, doc) in _QUERIES.items():
        add(name, input_types={}, value_type=value_type, doc=doc)
    for name, (input_types, value_type, doc) in _FUNCTIONS.items():
        add(name, input_types=input_types, value_type=value_type, doc=doc)
    # Payload-free overloads first, so a float in the fifth position is min_t rather than a payload.
    for tail in _tails():
        add("optix_traverse", input_types=_RAY | tail, value_type=None, doc=_TRAVERSE_DOC)
    for tail in _tails():
        add("optix_trace", input_types=_RAY | {"payload": Any} | tail, value_type=None, doc=_TRACE_DOC)
        add(
            "optix_traverse",
            input_types=_RAY | {"payload": Any} | tail,
            value_type=None,
            native_name="wp::optix_traverse_payload",
            doc=_TRAVERSE_DOC,
        )
    invoke_doc = "Run the closest-hit or miss program of the current hit object, optionally with a payload."
    add("optix_invoke", input_types={}, value_type=None, doc=invoke_doc)
    add("optix_invoke", input_types={"payload": Any}, value_type=None, doc=invoke_doc)
    reorder_doc = "Reorder threads by their hit objects (shader execution reordering), optionally with a hint."
    add("optix_reorder", input_types={}, value_type=None, doc=reorder_doc)
    add(
        "optix_reorder",
        input_types={"coherence_hint": _Int, "num_coherence_hint_bits_from_lsb": _Int},
        value_type=None,
        doc=reorder_doc,
    )
    # Exception details are optional trailing arguments (0-8).
    for n in range(9):
        add(
            "optix_throw_exception",
            input_types={"exception_code": _Int} | {f"detail_{i}": _Int for i in range(n)},
            value_type=None,
            doc="Throw a user exception with code >= 0 and up to eight optional details.",
        )
    # OptiX accepts zero to eight 32-bit attributes per reported intersection.
    for n in range(9):
        inputs = {"hit_t": wp.float32, "hit_kind": _Int} | {f"attribute_{i}": wp.uint32 for i in range(n)}
        add(
            "optix_report_intersection",
            input_types=inputs,
            value_type=wp.bool,
            doc="Report a custom intersection at ``hit_t`` (intersection programs); returns whether it was accepted.",
        )
    for name, input_types, value_type in _coop_vec_builtins():
        add(name, input_types=input_types, value_type=value_type, doc=_COOP_DOC)


def builtin_names() -> list[str]:
    """Names of every registered builtin, for documentation and tests."""
    names = [*_QUERIES, *_FUNCTIONS, "optix_trace", "optix_traverse", "optix_invoke", "optix_reorder"]
    names += ["optix_throw_exception", "optix_report_intersection", *(n for n, _, _ in _coop_vec_builtins())]
    return sorted(set(names), key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])
