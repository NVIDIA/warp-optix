# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptiX device builtins registered with Warp.

Every builtin maps one-to-one onto a wrapper in ``warp_optix_builtins.h`` and
is only valid inside programs compiled by :func:`warp_optix.create_pipeline`.
"""

from __future__ import annotations

from typing import Any

import warp as wp

_mat24 = wp.types.matrix(shape=(2, 4), dtype=wp.float32)
_mat34 = wp.types.matrix(shape=(3, 4), dtype=wp.float32)

# Argument-free state queries: name -> return type.
_QUERIES = {
    "optix_get_launch_index": wp.vec3ui,
    "optix_get_launch_dimensions": wp.vec3ui,
    "optix_get_world_ray_origin": wp.vec3,
    "optix_get_world_ray_direction": wp.vec3,
    "optix_get_object_ray_origin": wp.vec3,
    "optix_get_object_ray_direction": wp.vec3,
    "optix_get_ray_tmin": wp.float32,
    "optix_get_ray_tmax": wp.float32,
    "optix_get_ray_time": wp.float32,
    "optix_get_ray_flags": wp.uint32,
    "optix_get_ray_visibility_mask": wp.uint32,
    "optix_get_triangle_barycentrics": wp.vec2,
    "optix_get_triangle_vertex_data": wp.mat33,
    "optix_is_triangle_hit": wp.bool,
    "optix_is_triangle_front_face_hit": wp.bool,
    "optix_is_triangle_back_face_hit": wp.bool,
    "optix_is_front_face_hit": wp.bool,
    "optix_is_back_face_hit": wp.bool,
    "optix_get_linear_curve_vertex_data": _mat24,
    "optix_get_quadratic_bspline_vertex_data": _mat34,
    "optix_get_ribbon_vertex_data": _mat34,
    "optix_get_cubic_bspline_vertex_data": wp.mat44,
    "optix_get_catmull_rom_vertex_data": wp.mat44,
    "optix_get_cubic_bezier_vertex_data": wp.mat44,
    "optix_get_ribbon_parameters": wp.vec2,
    "optix_get_curve_parameter": wp.float32,
    "optix_get_primitive_index": wp.uint32,
    "optix_get_instance_id": wp.uint32,
    "optix_get_instance_index": wp.uint32,
    "optix_get_sbt_gas_index": wp.uint32,
    "optix_get_primitive_type": wp.uint32,
    "optix_get_gas_traversable_handle": wp.uint64,
    "optix_get_hit_kind": wp.uint32,
    "optix_get_object_to_world_transform_matrix": wp.mat44,
    "optix_get_world_to_object_transform_matrix": wp.mat44,
    "optix_get_transform_list_size": wp.uint32,
    "optix_get_exception_code": wp.int32,
    # Hit-object queries, valid after optix_traverse() in raygen, closest-hit, or miss.
    "optix_hit_object_is_hit": wp.bool,
    "optix_hit_object_is_miss": wp.bool,
    "optix_hit_object_get_world_ray_origin": wp.vec3,
    "optix_hit_object_get_world_ray_direction": wp.vec3,
    "optix_hit_object_get_ray_tmin": wp.float32,
    "optix_hit_object_get_ray_tmax": wp.float32,
    "optix_hit_object_get_primitive_index": wp.uint32,
    "optix_hit_object_get_instance_id": wp.uint32,
    "optix_hit_object_get_instance_index": wp.uint32,
    "optix_hit_object_get_sbt_gas_index": wp.uint32,
    "optix_hit_object_get_hit_kind": wp.uint32,
    "optix_hit_object_get_triangle_barycentrics": wp.vec2,
    "optix_hit_object_get_object_to_world_transform_matrix": wp.mat44,
    "optix_hit_object_get_world_to_object_transform_matrix": wp.mat44,
    **{f"optix_get_attribute_{i}": wp.uint32 for i in range(8)},
    **{f"optix_hit_object_get_attribute_{i}": wp.uint32 for i in range(8)},
    **{f"optix_get_exception_detail_{i}": wp.uint32 for i in range(8)},
    **{f"optix_get_payload_{i}": wp.uint32 for i in range(32)},
}

# Builtins with arguments: name -> (input types, return type).
_FUNCTIONS = {
    "float_to_uint32": ({"x": wp.float32}, wp.uint32),
    "uint32_to_float": ({"u": wp.uint32}, wp.float32),
    "optix_get_ribbon_normal": ({"parameters": wp.vec2}, wp.vec3),
    "optix_get_transform_list_handle": ({"index": wp.uint32}, wp.uint64),
    "optix_get_transform_type_from_handle": ({"handle": wp.uint64}, wp.uint32),
    "optix_terminate_ray": ({}, None),
    "optix_ignore_intersection": ({}, None),
    "optix_direct_call": ({"sbt_index": wp.uint32}, None),
    "optix_continuation_call": ({"sbt_index": wp.uint32}, None),
    "optix_load_payload": ({"payload": Any}, None),
    "optix_store_payload": ({"payload": Any}, None),
    **{f"optix_set_payload_{i}": ({"value": wp.uint32}, None) for i in range(32)},
    **{
        f"optix_{space}transform_{kind}_from_{src}_to_{dst}_space": ({kind: wp.vec3}, wp.vec3)
        for space in ("", "hit_object_")
        for kind in ("point", "vector", "normal")
        for src, dst in (("object", "world"), ("world", "object"))
        if not (space and src == "world")
    },
}

_RAY = {
    "traversable": wp.uint64,
    "ray_origin": wp.vec3,
    "ray_direction": wp.vec3,
    "tmin": wp.float32,
    "tmax": wp.float32,
    "ray_time": wp.float32,
    "visibility_mask": wp.uint32,
    "ray_flags": wp.uint32,
    "sbt_offset": wp.uint32,
    "sbt_stride": wp.uint32,
    "miss_sbt_index": wp.uint32,
}

# Builtins with several overloads: name -> list of input types.
_OVERLOADS = {
    "optix_trace": [_RAY | {"payload": Any}],
    "optix_traverse": [_RAY, _RAY | {"payload": Any}],
    "optix_invoke": [{}, {"payload": Any}],
    "optix_reorder": [{}, {"coherence_hint": wp.uint32, "num_coherence_hint_bits_from_lsb": wp.uint32}],
    "optix_throw_exception": [
        {"exception_code": wp.int32} | {f"detail_{i}": wp.uint32 for i in range(n)} for n in range(9)
    ],
}

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
    for name, value_type in _QUERIES.items():
        add(name, input_types={}, value_type=value_type)
    for name, (input_types, value_type) in _FUNCTIONS.items():
        add(name, input_types=input_types, value_type=value_type)
    for name, overloads in _OVERLOADS.items():
        for input_types in overloads:
            add(name, input_types=input_types, value_type=None)
    # OptiX accepts zero to eight 32-bit attributes per reported intersection.
    for n in range(9):
        inputs = {"hit_t": wp.float32, "hit_kind": wp.uint32} | {f"attribute_{i}": wp.uint32 for i in range(n)}
        add("optix_report_intersection", input_types=inputs, value_type=wp.bool)
    for name, input_types, value_type in _coop_vec_builtins():
        add(name, input_types=input_types, value_type=value_type, doc=_COOP_DOC)
