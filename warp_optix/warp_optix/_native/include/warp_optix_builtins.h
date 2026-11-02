// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OptiX device wrappers for Warp-generated programs.
//
// warp_optix injects this header through the CUDA preamble of OptiX modules
// only, so every wrapper may call OptiX device intrinsics unconditionally.

#pragma once

// The preamble precedes Warp's own module header, so CRT-free compilation must
// be requested here before builtin.h is pulled in under NVRTC.
#ifndef WP_NO_CRT
#define WP_NO_CRT
#endif

#include "optix_device.h"
#include "builtin.h"

namespace wp {

inline CUDA_CALLABLE_DEVICE vec3 optix_vec3(const float3& v) { return vec3(v.x, v.y, v.z); }
inline CUDA_CALLABLE_DEVICE float3 optix_float3(const vec3& v) { return make_float3(v[0], v[1], v[2]); }

inline CUDA_CALLABLE_DEVICE mat44 optix_mat44(const float (&m)[12])
{
    return mat44(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], 0.0f, 0.0f, 0.0f, 1.0f);
}

template <unsigned Count>
inline CUDA_CALLABLE_DEVICE mat_t<Count, 4, float> optix_mat_rows(const float4 (&rows)[Count])
{
    mat_t<Count, 4, float> result;
    for (unsigned i = 0; i < Count; ++i) {
        result.data[i][0] = rows[i].x;
        result.data[i][1] = rows[i].y;
        result.data[i][2] = rows[i].z;
        result.data[i][3] = rows[i].w;
    }
    return result;
}

inline CUDA_CALLABLE_DEVICE mat33 optix_mat33_rows(const float3 (&rows)[3])
{
    return mat33(rows[0].x, rows[0].y, rows[0].z, rows[1].x, rows[1].y, rows[1].z, rows[2].x, rows[2].y, rows[2].z);
}

// Bit casts used to move floats through 32-bit payload and attribute registers.
inline CUDA_CALLABLE uint32 float_to_uint32(float x) { return __float_as_uint(x); }
inline CUDA_CALLABLE float uint32_to_float(uint32 u) { return __uint_as_float(u); }

// ---------------------------------------------------------------------------
// Scalar state queries. Each wrapper forwards to the identically named OptiX
// intrinsic and converts CUDA vector types to Warp types.
// ---------------------------------------------------------------------------

#define WP_OPTIX_SCALAR(TYPE, NAME, OPTIX)                                                                             \
    inline CUDA_CALLABLE_DEVICE TYPE NAME() { return static_cast<TYPE>(OPTIX()); }
#define WP_OPTIX_VEC3(NAME, OPTIX)                                                                                     \
    inline CUDA_CALLABLE_DEVICE vec3 NAME() { return optix_vec3(OPTIX()); }
#define WP_OPTIX_VEC2(NAME, OPTIX)                                                                                     \
    inline CUDA_CALLABLE_DEVICE vec2 NAME()                                                                            \
    {                                                                                                                  \
        const float2 v = OPTIX();                                                                                      \
        return vec2(v.x, v.y);                                                                                         \
    }

inline CUDA_CALLABLE_DEVICE vec3ui optix_get_launch_index()
{
    const uint3 i = optixGetLaunchIndex();
    return vec3ui(i.x, i.y, i.z);
}

inline CUDA_CALLABLE_DEVICE vec3ui optix_get_launch_dimensions()
{
    const uint3 d = optixGetLaunchDimensions();
    return vec3ui(d.x, d.y, d.z);
}

WP_OPTIX_VEC3(optix_get_world_ray_origin, optixGetWorldRayOrigin)
WP_OPTIX_VEC3(optix_get_world_ray_direction, optixGetWorldRayDirection)
WP_OPTIX_VEC3(optix_get_object_ray_origin, optixGetObjectRayOrigin)
WP_OPTIX_VEC3(optix_get_object_ray_direction, optixGetObjectRayDirection)
WP_OPTIX_SCALAR(float, optix_get_ray_tmin, optixGetRayTmin)
WP_OPTIX_SCALAR(float, optix_get_ray_tmax, optixGetRayTmax)
WP_OPTIX_SCALAR(float, optix_get_ray_time, optixGetRayTime)
WP_OPTIX_SCALAR(uint32, optix_get_ray_flags, optixGetRayFlags)
WP_OPTIX_SCALAR(uint32, optix_get_ray_visibility_mask, optixGetRayVisibilityMask)
WP_OPTIX_VEC2(optix_get_triangle_barycentrics, optixGetTriangleBarycentrics)
WP_OPTIX_SCALAR(bool, optix_is_triangle_hit, optixIsTriangleHit)
WP_OPTIX_SCALAR(bool, optix_is_triangle_front_face_hit, optixIsTriangleFrontFaceHit)
WP_OPTIX_SCALAR(bool, optix_is_triangle_back_face_hit, optixIsTriangleBackFaceHit)
WP_OPTIX_SCALAR(bool, optix_is_front_face_hit, optixIsFrontFaceHit)
WP_OPTIX_SCALAR(bool, optix_is_back_face_hit, optixIsBackFaceHit)
WP_OPTIX_SCALAR(float, optix_get_curve_parameter, optixGetCurveParameter)
WP_OPTIX_VEC2(optix_get_ribbon_parameters, optixGetRibbonParameters)
WP_OPTIX_SCALAR(uint32, optix_get_primitive_index, optixGetPrimitiveIndex)
WP_OPTIX_SCALAR(uint32, optix_get_instance_id, optixGetInstanceId)
WP_OPTIX_SCALAR(uint32, optix_get_instance_index, optixGetInstanceIndex)
WP_OPTIX_SCALAR(uint32, optix_get_sbt_gas_index, optixGetSbtGASIndex)
WP_OPTIX_SCALAR(uint32, optix_get_primitive_type, optixGetPrimitiveType)
WP_OPTIX_SCALAR(uint64, optix_get_gas_traversable_handle, optixGetGASTraversableHandle)
WP_OPTIX_SCALAR(uint32, optix_get_hit_kind, optixGetHitKind)
WP_OPTIX_SCALAR(uint32, optix_get_transform_list_size, optixGetTransformListSize)
WP_OPTIX_SCALAR(int32, optix_get_exception_code, optixGetExceptionCode)

WP_OPTIX_SCALAR(bool, optix_hit_object_is_hit, optixHitObjectIsHit)
WP_OPTIX_SCALAR(bool, optix_hit_object_is_miss, optixHitObjectIsMiss)
WP_OPTIX_VEC3(optix_hit_object_get_world_ray_origin, optixHitObjectGetWorldRayOrigin)
WP_OPTIX_VEC3(optix_hit_object_get_world_ray_direction, optixHitObjectGetWorldRayDirection)
WP_OPTIX_SCALAR(float, optix_hit_object_get_ray_tmin, optixHitObjectGetRayTmin)
WP_OPTIX_SCALAR(float, optix_hit_object_get_ray_tmax, optixHitObjectGetRayTmax)
WP_OPTIX_SCALAR(uint32, optix_hit_object_get_primitive_index, optixHitObjectGetPrimitiveIndex)
WP_OPTIX_SCALAR(uint32, optix_hit_object_get_instance_id, optixHitObjectGetInstanceId)
WP_OPTIX_SCALAR(uint32, optix_hit_object_get_instance_index, optixHitObjectGetInstanceIndex)
WP_OPTIX_SCALAR(uint32, optix_hit_object_get_sbt_gas_index, optixHitObjectGetSbtGASIndex)
WP_OPTIX_SCALAR(uint32, optix_hit_object_get_hit_kind, optixHitObjectGetHitKind)
WP_OPTIX_VEC2(optix_hit_object_get_triangle_barycentrics, optixHitObjectGetTriangleBarycentrics)

#undef WP_OPTIX_SCALAR
#undef WP_OPTIX_VEC3
#undef WP_OPTIX_VEC2

#define WP_OPTIX_INDEXED_8(MACRO) MACRO(0) MACRO(1) MACRO(2) MACRO(3) MACRO(4) MACRO(5) MACRO(6) MACRO(7)

#define WP_OPTIX_ATTRIBUTE(I)                                                                                          \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_attribute_##I() { return optixGetAttribute_##I(); }                   \
    inline CUDA_CALLABLE_DEVICE uint32 optix_hit_object_get_attribute_##I() { return optixHitObjectGetAttribute_##I(); }
#define WP_OPTIX_EXCEPTION_DETAIL(I)                                                                                   \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_exception_detail_##I() { return optixGetExceptionDetail_##I(); }
WP_OPTIX_INDEXED_8(WP_OPTIX_ATTRIBUTE)
WP_OPTIX_INDEXED_8(WP_OPTIX_EXCEPTION_DETAIL)
#undef WP_OPTIX_ATTRIBUTE
#undef WP_OPTIX_EXCEPTION_DETAIL
#undef WP_OPTIX_INDEXED_8

// ---------------------------------------------------------------------------
// Geometry and transform queries.
// ---------------------------------------------------------------------------

inline CUDA_CALLABLE_DEVICE mat33 optix_get_triangle_vertex_data()
{
    float3 v[3];
    optixGetTriangleVertexData(v);
    return optix_mat33_rows(v);
}

#define WP_OPTIX_CURVE(NAME, OPTIX, COUNT)                                                                             \
    inline CUDA_CALLABLE_DEVICE mat_t<COUNT, 4, float> NAME()                                                          \
    {                                                                                                                  \
        float4 v[COUNT];                                                                                               \
        OPTIX(v);                                                                                                      \
        return optix_mat_rows(v);                                                                                      \
    }
WP_OPTIX_CURVE(optix_get_linear_curve_vertex_data, optixGetLinearCurveVertexData, 2)
WP_OPTIX_CURVE(optix_get_quadratic_bspline_vertex_data, optixGetQuadraticBSplineVertexData, 3)
WP_OPTIX_CURVE(optix_get_cubic_bspline_vertex_data, optixGetCubicBSplineVertexData, 4)
WP_OPTIX_CURVE(optix_get_catmull_rom_vertex_data, optixGetCatmullRomVertexData, 4)
WP_OPTIX_CURVE(optix_get_cubic_bezier_vertex_data, optixGetCubicBezierVertexData, 4)
WP_OPTIX_CURVE(optix_get_ribbon_vertex_data, optixGetRibbonVertexData, 3)
#undef WP_OPTIX_CURVE

inline CUDA_CALLABLE_DEVICE vec3 optix_get_ribbon_normal(const vec2& parameters)
{
    return optix_vec3(optixGetRibbonNormal(make_float2(parameters[0], parameters[1])));
}

#define WP_OPTIX_TRANSFORM(NAME, OPTIX, ARG)                                                                           \
    inline CUDA_CALLABLE_DEVICE vec3 NAME(const vec3& ARG) { return optix_vec3(OPTIX(optix_float3(ARG))); }
WP_OPTIX_TRANSFORM(optix_transform_point_from_object_to_world_space, optixTransformPointFromObjectToWorldSpace, point)
WP_OPTIX_TRANSFORM(optix_transform_vector_from_object_to_world_space, optixTransformVectorFromObjectToWorldSpace, vector)
WP_OPTIX_TRANSFORM(optix_transform_normal_from_object_to_world_space, optixTransformNormalFromObjectToWorldSpace, normal)
WP_OPTIX_TRANSFORM(optix_transform_point_from_world_to_object_space, optixTransformPointFromWorldToObjectSpace, point)
WP_OPTIX_TRANSFORM(optix_transform_vector_from_world_to_object_space, optixTransformVectorFromWorldToObjectSpace, vector)
WP_OPTIX_TRANSFORM(optix_transform_normal_from_world_to_object_space, optixTransformNormalFromWorldToObjectSpace, normal)
WP_OPTIX_TRANSFORM(optix_hit_object_transform_point_from_object_to_world_space, optixHitObjectTransformPointFromObjectToWorldSpace, point)
WP_OPTIX_TRANSFORM(optix_hit_object_transform_vector_from_object_to_world_space, optixHitObjectTransformVectorFromObjectToWorldSpace, vector)
WP_OPTIX_TRANSFORM(optix_hit_object_transform_normal_from_object_to_world_space, optixHitObjectTransformNormalFromObjectToWorldSpace, normal)
#undef WP_OPTIX_TRANSFORM

#define WP_OPTIX_MATRIX(NAME, OPTIX)                                                                                   \
    inline CUDA_CALLABLE_DEVICE mat44 NAME()                                                                           \
    {                                                                                                                  \
        float m[12];                                                                                                   \
        OPTIX(m);                                                                                                      \
        return optix_mat44(m);                                                                                         \
    }
WP_OPTIX_MATRIX(optix_get_object_to_world_transform_matrix, optixGetObjectToWorldTransformMatrix)
WP_OPTIX_MATRIX(optix_get_world_to_object_transform_matrix, optixGetWorldToObjectTransformMatrix)
WP_OPTIX_MATRIX(optix_hit_object_get_object_to_world_transform_matrix, optixHitObjectGetObjectToWorldTransformMatrix)
WP_OPTIX_MATRIX(optix_hit_object_get_world_to_object_transform_matrix, optixHitObjectGetWorldToObjectTransformMatrix)
#undef WP_OPTIX_MATRIX

inline CUDA_CALLABLE_DEVICE uint64 optix_get_transform_list_handle(uint32 index)
{
    return optixGetTransformListHandle(index);
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_transform_type_from_handle(uint64 handle)
{
    return optixGetTransformTypeFromHandle(static_cast<OptixTraversableHandle>(handle));
}

// ---------------------------------------------------------------------------
// Control flow.
// ---------------------------------------------------------------------------

inline CUDA_CALLABLE_DEVICE void optix_terminate_ray() { optixTerminateRay(); }
inline CUDA_CALLABLE_DEVICE void optix_ignore_intersection() { optixIgnoreIntersection(); }
inline CUDA_CALLABLE_DEVICE void optix_direct_call(uint32 sbt_index) { optixDirectCall<void>(sbt_index); }
inline CUDA_CALLABLE_DEVICE void optix_continuation_call(uint32 sbt_index) { optixContinuationCall<void>(sbt_index); }
inline CUDA_CALLABLE_DEVICE void optix_reorder() { optixReorder(); }
inline CUDA_CALLABLE_DEVICE void optix_reorder(uint32 hint, uint32 num_hint_bits) { optixReorder(hint, num_hint_bits); }

template <typename... Details>
inline CUDA_CALLABLE_DEVICE void optix_throw_exception(int32 exception_code, Details... details)
{
    optixThrowException(exception_code, static_cast<uint32>(details)...);
}

template <typename... Attributes>
inline CUDA_CALLABLE_DEVICE bool optix_report_intersection(float hit_t, uint32 hit_kind, Attributes... attributes)
{
    return optixReportIntersection(hit_t, hit_kind, static_cast<uint32>(attributes)...);
}

// ---------------------------------------------------------------------------
// Tracing. A payload is any trivially copyable Warp value whose size is a
// multiple of four bytes; it is passed through OptiX's 32-bit payload registers.
// ---------------------------------------------------------------------------

template <typename Payload> union optix_payload_words {
    static_assert(__is_trivially_copyable(Payload), "OptiX payload must be trivially copyable");
    static_assert(sizeof(Payload) % sizeof(uint32) == 0, "OptiX payload size must be a multiple of 4 bytes");
    static constexpr size_t count = sizeof(Payload) / sizeof(uint32);
    static_assert(count > 0 && count <= 32, "OptiX payload must use between 1 and 32 registers");

    Payload payload;
    uint32 words[count];
};

template <size_t... I> struct optix_index_sequence { };
template <size_t N, size_t... I> struct optix_make_index_sequence : optix_make_index_sequence<N - 1, N - 1, I...> { };
template <size_t... I> struct optix_make_index_sequence<0, I...> {
    using type = optix_index_sequence<I...>;
};

#define WP_OPTIX_RAY_PARAMS                                                                                            \
    uint64 traversable, const vec3 &ray_origin, const vec3 &ray_direction, float tmin, float tmax, float ray_time,     \
        uint32 visibility_mask, uint32 ray_flags, uint32 sbt_offset, uint32 sbt_stride, uint32 miss_sbt_index
#define WP_OPTIX_RAY_ARGS                                                                                              \
    static_cast<OptixTraversableHandle>(traversable), optix_float3(ray_origin), optix_float3(ray_direction), tmin,     \
        tmax, ray_time, static_cast<OptixVisibilityMask>(visibility_mask), ray_flags, sbt_offset, sbt_stride,          \
        miss_sbt_index
#define WP_OPTIX_RAY_NAMES                                                                                             \
    traversable, ray_origin, ray_direction, tmin, tmax, ray_time, visibility_mask, ray_flags, sbt_offset, sbt_stride,  \
        miss_sbt_index

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_trace_words(WP_OPTIX_RAY_PARAMS, uint32 (&w)[N], optix_index_sequence<I...>)
{
    optixTrace(WP_OPTIX_RAY_ARGS, w[I]...);
}

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_traverse_words(WP_OPTIX_RAY_PARAMS, uint32 (&w)[N], optix_index_sequence<I...>)
{
    optixTraverse(WP_OPTIX_RAY_ARGS, w[I]...);
}

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_invoke_words(uint32 (&w)[N], optix_index_sequence<I...>)
{
    optixInvoke(w[I]...);
}

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_trace(WP_OPTIX_RAY_PARAMS, Payload& payload)
{
    using Words = optix_payload_words<Payload>;
    Words packed = {};
    packed.payload = payload;
    optix_trace_words(WP_OPTIX_RAY_NAMES, packed.words, typename optix_make_index_sequence<Words::count>::type {});
    payload = packed.payload;
}

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_traverse(WP_OPTIX_RAY_PARAMS, Payload& payload)
{
    using Words = optix_payload_words<Payload>;
    Words packed = {};
    packed.payload = payload;
    optix_traverse_words(WP_OPTIX_RAY_NAMES, packed.words, typename optix_make_index_sequence<Words::count>::type {});
    payload = packed.payload;
}

// Payload-free traversal: the result is read from the hit object in the caller.
inline CUDA_CALLABLE_DEVICE void optix_traverse(WP_OPTIX_RAY_PARAMS) { optixTraverse(WP_OPTIX_RAY_ARGS); }

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_invoke(Payload& payload)
{
    using Words = optix_payload_words<Payload>;
    Words packed = {};
    packed.payload = payload;
    optix_invoke_words(packed.words, typename optix_make_index_sequence<Words::count>::type {});
    payload = packed.payload;
}

inline CUDA_CALLABLE_DEVICE void optix_invoke() { optixInvoke(); }

#undef WP_OPTIX_RAY_PARAMS
#undef WP_OPTIX_RAY_ARGS
#undef WP_OPTIX_RAY_NAMES

// OptiX payload registers are only addressable by compile-time index; the
// switch folds away whenever the index is a constant, as in the loops below.
inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_index(uint32 index)
{
    switch (index) {
#define WP_OPTIX_GET(I)                                                                                                \
    case I:                                                                                                            \
        return optixGetPayload_##I();
        WP_OPTIX_GET(0) WP_OPTIX_GET(1) WP_OPTIX_GET(2) WP_OPTIX_GET(3) WP_OPTIX_GET(4) WP_OPTIX_GET(5)
        WP_OPTIX_GET(6) WP_OPTIX_GET(7) WP_OPTIX_GET(8) WP_OPTIX_GET(9) WP_OPTIX_GET(10) WP_OPTIX_GET(11)
        WP_OPTIX_GET(12) WP_OPTIX_GET(13) WP_OPTIX_GET(14) WP_OPTIX_GET(15) WP_OPTIX_GET(16) WP_OPTIX_GET(17)
        WP_OPTIX_GET(18) WP_OPTIX_GET(19) WP_OPTIX_GET(20) WP_OPTIX_GET(21) WP_OPTIX_GET(22) WP_OPTIX_GET(23)
        WP_OPTIX_GET(24) WP_OPTIX_GET(25) WP_OPTIX_GET(26) WP_OPTIX_GET(27) WP_OPTIX_GET(28) WP_OPTIX_GET(29)
        WP_OPTIX_GET(30) WP_OPTIX_GET(31)
#undef WP_OPTIX_GET
    default:
        return 0u;
    }
}

inline CUDA_CALLABLE_DEVICE void optix_set_payload_index(uint32 index, uint32 value)
{
    switch (index) {
#define WP_OPTIX_SET(I)                                                                                                \
    case I:                                                                                                            \
        optixSetPayload_##I(value);                                                                                    \
        return;
        WP_OPTIX_SET(0) WP_OPTIX_SET(1) WP_OPTIX_SET(2) WP_OPTIX_SET(3) WP_OPTIX_SET(4) WP_OPTIX_SET(5)
        WP_OPTIX_SET(6) WP_OPTIX_SET(7) WP_OPTIX_SET(8) WP_OPTIX_SET(9) WP_OPTIX_SET(10) WP_OPTIX_SET(11)
        WP_OPTIX_SET(12) WP_OPTIX_SET(13) WP_OPTIX_SET(14) WP_OPTIX_SET(15) WP_OPTIX_SET(16) WP_OPTIX_SET(17)
        WP_OPTIX_SET(18) WP_OPTIX_SET(19) WP_OPTIX_SET(20) WP_OPTIX_SET(21) WP_OPTIX_SET(22) WP_OPTIX_SET(23)
        WP_OPTIX_SET(24) WP_OPTIX_SET(25) WP_OPTIX_SET(26) WP_OPTIX_SET(27) WP_OPTIX_SET(28) WP_OPTIX_SET(29)
        WP_OPTIX_SET(30) WP_OPTIX_SET(31)
#undef WP_OPTIX_SET
    default:
        return;
    }
}

#define WP_OPTIX_PAYLOAD(I)                                                                                            \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_##I() { return optixGetPayload_##I(); }                       \
    inline CUDA_CALLABLE_DEVICE void optix_set_payload_##I(uint32 value) { optixSetPayload_##I(value); }
WP_OPTIX_PAYLOAD(0) WP_OPTIX_PAYLOAD(1) WP_OPTIX_PAYLOAD(2) WP_OPTIX_PAYLOAD(3) WP_OPTIX_PAYLOAD(4)
WP_OPTIX_PAYLOAD(5) WP_OPTIX_PAYLOAD(6) WP_OPTIX_PAYLOAD(7) WP_OPTIX_PAYLOAD(8) WP_OPTIX_PAYLOAD(9)
WP_OPTIX_PAYLOAD(10) WP_OPTIX_PAYLOAD(11) WP_OPTIX_PAYLOAD(12) WP_OPTIX_PAYLOAD(13) WP_OPTIX_PAYLOAD(14)
WP_OPTIX_PAYLOAD(15) WP_OPTIX_PAYLOAD(16) WP_OPTIX_PAYLOAD(17) WP_OPTIX_PAYLOAD(18) WP_OPTIX_PAYLOAD(19)
WP_OPTIX_PAYLOAD(20) WP_OPTIX_PAYLOAD(21) WP_OPTIX_PAYLOAD(22) WP_OPTIX_PAYLOAD(23) WP_OPTIX_PAYLOAD(24)
WP_OPTIX_PAYLOAD(25) WP_OPTIX_PAYLOAD(26) WP_OPTIX_PAYLOAD(27) WP_OPTIX_PAYLOAD(28) WP_OPTIX_PAYLOAD(29)
WP_OPTIX_PAYLOAD(30) WP_OPTIX_PAYLOAD(31)
#undef WP_OPTIX_PAYLOAD

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_load_payload(Payload& payload)
{
    using Words = optix_payload_words<Payload>;
    Words packed = {};
#pragma unroll
    for (uint32 i = 0; i < Words::count; ++i)
        packed.words[i] = optix_get_payload_index(i);
    payload = packed.payload;
}

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_store_payload(const Payload& payload)
{
    using Words = optix_payload_words<Payload>;
    Words packed = {};
    packed.payload = payload;
#pragma unroll
    for (uint32 i = 0; i < Words::count; ++i)
        optix_set_payload_index(i, packed.words[i]);
}

// ---------------------------------------------------------------------------
// Cooperative vectors. OptiX uses CUDA's global half type, while Warp uses its
// own ABI-compatible wp::half; the adapters below keep Warp code on wp.vector.
// ---------------------------------------------------------------------------

template <typename A, typename B> struct optix_coop_is_same { static constexpr bool value = false; };
template <typename A> struct optix_coop_is_same<A, A> { static constexpr bool value = true; };

template <typename T> struct optix_coop_scalar;
#define WP_OPTIX_COOP_SCALAR(T)                                                                                        \
    template <> struct optix_coop_scalar<T> {                                                                          \
        using type = T;                                                                                                \
        static inline CUDA_CALLABLE_DEVICE type to_optix(T value) { return value; }                                   \
        static inline CUDA_CALLABLE_DEVICE T from_optix(type value) { return value; }                                 \
    };
WP_OPTIX_COOP_SCALAR(float)
WP_OPTIX_COOP_SCALAR(int8)
WP_OPTIX_COOP_SCALAR(uint8)
WP_OPTIX_COOP_SCALAR(int32)
WP_OPTIX_COOP_SCALAR(uint32)
#undef WP_OPTIX_COOP_SCALAR
template <> struct optix_coop_scalar<half> {
    using type = ::half;
    static inline CUDA_CALLABLE_DEVICE type to_optix(half value) { return __ushort_as_half(value.u); }
    static inline CUDA_CALLABLE_DEVICE half from_optix(type value)
    {
        half result;
        result.u = __half_as_ushort(value);
        return result;
    }
};

template <typename T> struct optix_coop_elem_type;
template <> struct optix_coop_elem_type<half> {
    static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
};
template <> struct optix_coop_elem_type<float> {
    static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32;
};

template <typename Vec> struct optix_coop_vec_traits;
template <unsigned Length, typename T> struct optix_coop_vec_traits<vec_t<Length, T>> {
    using scalar_type = T;
    using optix_type = OptixCoopVec<typename optix_coop_scalar<T>::type, Length>;
    static constexpr unsigned size = Length;
};

template <typename Vec>
inline CUDA_CALLABLE_DEVICE typename optix_coop_vec_traits<Vec>::optix_type optix_coop_to_native(const Vec& value)
{
    using Traits = optix_coop_vec_traits<Vec>;
    typename Traits::optix_type result;
#pragma unroll
    for (unsigned i = 0; i < Traits::size; ++i)
        result[i] = optix_coop_scalar<typename Traits::scalar_type>::to_optix(value[i]);
    return result;
}

template <typename Vec>
inline CUDA_CALLABLE_DEVICE Vec optix_coop_from_native(const typename optix_coop_vec_traits<Vec>::optix_type& value)
{
    using Traits = optix_coop_vec_traits<Vec>;
    Vec result;
#pragma unroll
    for (unsigned i = 0; i < Traits::size; ++i)
        result[i] = optix_coop_scalar<typename Traits::scalar_type>::from_optix(value[i]);
    return result;
}

template <typename Vec> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_load(uint64 address, Vec& output)
{
    using Native = typename optix_coop_vec_traits<Vec>::optix_type;
    output = optix_coop_from_native<Vec>(optixCoopVecLoad<Native>(static_cast<CUdeviceptr>(address)));
}

#define WP_OPTIX_COOP_UNARY(NAME, OPTIX)                                                                               \
    template <typename Vec> inline CUDA_CALLABLE_DEVICE void NAME(const Vec& input, Vec& output)                       \
    {                                                                                                                  \
        output = optix_coop_from_native<Vec>(OPTIX(optix_coop_to_native(input)));                                     \
    }
WP_OPTIX_COOP_UNARY(optix_coop_vec_exp2, optixCoopVecExp2)
WP_OPTIX_COOP_UNARY(optix_coop_vec_log2, optixCoopVecLog2)
WP_OPTIX_COOP_UNARY(optix_coop_vec_tanh, optixCoopVecTanh)
#undef WP_OPTIX_COOP_UNARY

#define WP_OPTIX_COOP_BINARY(NAME, OPTIX)                                                                              \
    template <typename Vec> inline CUDA_CALLABLE_DEVICE void NAME(const Vec& a, const Vec& b, Vec& output)             \
    {                                                                                                                  \
        output = optix_coop_from_native<Vec>(OPTIX(optix_coop_to_native(a), optix_coop_to_native(b)));                \
    }
WP_OPTIX_COOP_BINARY(optix_coop_vec_min, optixCoopVecMin)
WP_OPTIX_COOP_BINARY(optix_coop_vec_max, optixCoopVecMax)
WP_OPTIX_COOP_BINARY(optix_coop_vec_mul, optixCoopVecMul)
WP_OPTIX_COOP_BINARY(optix_coop_vec_add, optixCoopVecAdd)
WP_OPTIX_COOP_BINARY(optix_coop_vec_sub, optixCoopVecSub)
WP_OPTIX_COOP_BINARY(optix_coop_vec_step, optixCoopVecStep)
#undef WP_OPTIX_COOP_BINARY

#define WP_OPTIX_COOP_SCALAR_BINARY(NAME, OPTIX)                                                                       \
    template <typename Vec>                                                                                            \
    inline CUDA_CALLABLE_DEVICE void NAME(const Vec& a, typename optix_coop_vec_traits<Vec>::scalar_type b, Vec& output) \
    {                                                                                                                  \
        using Scalar = optix_coop_scalar<typename optix_coop_vec_traits<Vec>::scalar_type>;                           \
        output = optix_coop_from_native<Vec>(OPTIX(optix_coop_to_native(a), Scalar::to_optix(b)));                     \
    }
WP_OPTIX_COOP_SCALAR_BINARY(optix_coop_vec_min_scalar, optixCoopVecMin)
WP_OPTIX_COOP_SCALAR_BINARY(optix_coop_vec_max_scalar, optixCoopVecMax)
#undef WP_OPTIX_COOP_SCALAR_BINARY

template <typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_cvt(const VecIn& input, VecOut& output)
{
    static_assert(optix_coop_vec_traits<VecIn>::size == optix_coop_vec_traits<VecOut>::size,
                  "optix_coop_vec_cvt input and output must have equal lengths");
    using NativeOut = typename optix_coop_vec_traits<VecOut>::optix_type;
    output = optix_coop_from_native<VecOut>(optixCoopVecCvt<NativeOut>(optix_coop_to_native(input)));
}

template <typename Vec>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_ffma(const Vec& a, const Vec& b, const Vec& c, Vec& output)
{
    output = optix_coop_from_native<Vec>(
        optixCoopVecFFMA(optix_coop_to_native(a), optix_coop_to_native(b), optix_coop_to_native(c)));
}

template <OptixCoopVecElemType MatrixType, typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void
optix_coop_vec_matmul_impl(const VecIn& input, uint64 matrix, uint32 matrix_offset, uint32 stride, VecOut& output)
{
    using IT = optix_coop_vec_traits<VecIn>;
    using OT = optix_coop_vec_traits<VecOut>;
    output = optix_coop_from_native<VecOut>(
        optixCoopVecMatMul<typename OT::optix_type, typename IT::optix_type, MatrixType,
                           OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, false, OT::size, IT::size, MatrixType>(
            optix_coop_to_native(input), static_cast<CUdeviceptr>(matrix), matrix_offset, stride));
}

template <OptixCoopVecElemType MatrixType, typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_bias_impl(
    const VecIn& input, uint64 matrix, uint32 matrix_offset, uint64 bias, uint32 bias_offset, uint32 stride,
    VecOut& output)
{
    using IT = optix_coop_vec_traits<VecIn>;
    using OT = optix_coop_vec_traits<VecOut>;
    constexpr OptixCoopVecElemType BiasType = optix_coop_elem_type<typename OT::scalar_type>::value;
    output = optix_coop_from_native<VecOut>(
        optixCoopVecMatMul<typename OT::optix_type, typename IT::optix_type, MatrixType,
                           OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, false, OT::size, IT::size, MatrixType,
                           BiasType>(
            optix_coop_to_native(input), static_cast<CUdeviceptr>(matrix), matrix_offset,
            static_cast<CUdeviceptr>(bias), bias_offset, stride));
}

#define WP_OPTIX_COOP_MATMUL(SUFFIX, TYPE)                                                                             \
    template <typename VI, typename VO>                                                                                \
    inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_##SUFFIX(                                                   \
        const VI& input, uint64 matrix, uint32 offset, uint32 stride, VO& output)                                      \
    {                                                                                                                  \
        optix_coop_vec_matmul_impl<TYPE>(input, matrix, offset, stride, output);                                       \
    }                                                                                                                  \
    template <typename VI, typename VO>                                                                                \
    inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_bias_##SUFFIX(                                              \
        const VI& input, uint64 matrix, uint32 offset, uint64 bias, uint32 bias_offset, uint32 stride, VO& output)     \
    {                                                                                                                  \
        optix_coop_vec_matmul_bias_impl<TYPE>(input, matrix, offset, bias, bias_offset, stride, output);               \
    }                                                                                                                  \
    template <typename VI, typename VO>                                                                                \
    inline CUDA_CALLABLE_DEVICE uint32 optix_coop_vec_matrix_size_##SUFFIX(const VI&, const VO&)                       \
    {                                                                                                                  \
        return optixCoopVecGetMatrixSize<optix_coop_vec_traits<VO>::size, optix_coop_vec_traits<VI>::size, TYPE,       \
                                         OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, 0>();                       \
    }
WP_OPTIX_COOP_MATMUL(fp16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16)
WP_OPTIX_COOP_MATMUL(fp8_e4m3, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3)
WP_OPTIX_COOP_MATMUL(fp8_e5m2, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2)
WP_OPTIX_COOP_MATMUL(int8, OPTIX_COOP_VEC_ELEM_TYPE_INT8)
WP_OPTIX_COOP_MATMUL(uint8, OPTIX_COOP_VEC_ELEM_TYPE_UINT8)
#undef WP_OPTIX_COOP_MATMUL

template <typename Vec>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_reduce_sum_accumulate(const Vec& input, uint64 output, uint32 offset)
{
    using Scalar = typename optix_coop_vec_traits<Vec>::scalar_type;
    static_assert(optix_coop_is_same<Scalar, half>::value || optix_coop_is_same<Scalar, float>::value,
                  "optix_coop_vec_reduce_sum_accumulate requires a float16 or float32 vector");
    optixCoopVecReduceSumAccumulate(optix_coop_to_native(input), static_cast<CUdeviceptr>(output), offset);
}

template <typename VecA, typename VecB>
inline CUDA_CALLABLE_DEVICE void
optix_coop_vec_outer_product_accumulate(const VecA& a, const VecB& b, uint64 output, uint32 offset)
{
    static_assert(optix_coop_is_same<typename optix_coop_vec_traits<VecA>::scalar_type, half>::value
                      && optix_coop_is_same<typename optix_coop_vec_traits<VecB>::scalar_type, half>::value,
                  "optix_coop_vec_outer_product_accumulate requires two float16 vectors");
    optixCoopVecOuterProductAccumulate<typename optix_coop_vec_traits<VecA>::optix_type,
                                       typename optix_coop_vec_traits<VecB>::optix_type,
                                       OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL>(
        optix_coop_to_native(a), optix_coop_to_native(b), static_cast<CUdeviceptr>(output), offset, 0);
}

}  // namespace wp
