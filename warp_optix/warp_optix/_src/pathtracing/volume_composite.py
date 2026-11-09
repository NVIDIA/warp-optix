# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from .pathtracing_warp_kernels import (
    MISS_DISTANCE,
    PathtraceLaunchParams,
    _compute_ray_dir,
    _compute_ray_origin,
    _mul_mat3x3_cm,
    _pcg_rand01,
    _sample_environment,
    _xxhash32,
)


@wp.struct
class VolumeIntegration:
    radiance: wp.vec3
    transmittance: wp.float32


@wp.struct
class VolumeParams:
    volume: wp.uint64
    bounds_min: wp.vec3
    bounds_max: wp.vec3
    density_scale: wp.float32
    step_size: wp.float32
    cool_dark: wp.vec3
    cool_mid: wp.vec3
    cool_light: wp.vec3
    warm_dark: wp.vec3
    warm_mid: wp.vec3
    warm_light: wp.vec3
    emission: wp.float32
    anisotropy: wp.float32
    transfer_table: wp.array(dtype=wp.vec4)
    transfer_count: wp.uint32
    density_feature: wp.uint32


@wp.func
def _sample_volume(params: VolumeParams, index: wp.vec3):
    if params.density_feature != wp.uint32(0):
        return wp.volume_sample_v(params.volume, index, wp.Volume.LINEAR)
    field = wp.volume_sample_f(params.volume, index, wp.Volume.LINEAR)
    feature = -1.0
    if field > 0.0:
        feature = 1.0
    return wp.vec3(wp.abs(field), feature, 1.0)


@wp.func
def _volume_palette(params: VolumeParams, field: float, ramp: float):
    if params.transfer_count > wp.uint32(1):
        coordinate = wp.clamp(0.5 + 0.5 * field, 0.0, 1.0)
        table_position = coordinate * float(params.transfer_count - wp.uint32(1))
        low = wp.min(wp.int32(table_position), wp.int32(params.transfer_count) - 2)
        blend = table_position - float(low)
        return params.transfer_table[low] * (1.0 - blend) + params.transfer_table[low + 1] * blend
    dark = params.cool_dark
    mid = params.cool_mid
    light = params.cool_light
    if field > 0.0:
        dark = params.warm_dark
        mid = params.warm_mid
        light = params.warm_light
    color = dark * (1.0 - 2.0 * ramp) + mid * (2.0 * ramp)
    if ramp > 0.5:
        t = 2.0 * ramp - 1.0
        color = mid * (1.0 - t) + light * t
    return wp.vec4(color[0], color[1], color[2], 1.0)


@wp.func
def _integrate_volume(
    params: PathtraceLaunchParams,
    volume_params: VolumeParams,
    origin: wp.vec3,
    direction: wp.vec3,
    surface_t: wp.float32,
    jitter: wp.float32,
) -> VolumeIntegration:
    result = VolumeIntegration()
    result.radiance = wp.vec3(0.0)
    result.transmittance = 1.0
    inv_dir = wp.vec3(1.0e20)
    for axis in range(3):
        if wp.abs(direction[axis]) > 1.0e-10:
            inv_dir[axis] = 1.0 / direction[axis]
    lo = wp.cw_mul(volume_params.bounds_min - origin, inv_dir)
    hi = wp.cw_mul(volume_params.bounds_max - origin, inv_dir)
    t0 = wp.max(wp.min(lo, hi))
    t1 = wp.min(wp.max(lo, hi))
    t0 = wp.max(t0, 0.0)
    t1 = wp.min(t1, surface_t)
    step_size = wp.max(volume_params.step_size, 1.0e-5)
    if t1 <= t0:
        return result
    t = t0 + jitter * step_size
    sun_direction = wp.normalize(params.sky.sun_direction)
    sun_radiance = _sample_environment(params, sun_direction)
    g = wp.clamp(volume_params.anisotropy, -0.95, 0.95)
    phase_base = 1.0 + g * g - 2.0 * g * wp.dot(sun_direction, -direction)
    phase = (1.0 - g * g) / (4.0 * wp.pi * phase_base * wp.sqrt(phase_base))
    ambient = params.ambient_light + _sample_environment(params, -direction) * 0.08
    steps = wp.int32(0)
    while t < t1 and steps < wp.int32(2048) and result.transmittance > 1.0e-3:
        position = origin + direction * t
        index = wp.volume_world_to_index(volume_params.volume, position)
        sample = _sample_volume(volume_params, index)
        density = wp.max(sample[0], 0.0)
        field = sample[1]
        if density < 1.0e-4:
            t += 3.0 * step_size
            steps += wp.int32(1)
            continue
        base_sigma_t = density * volume_params.density_scale
        ramp = wp.sqrt(wp.clamp(base_sigma_t, 0.0, 1.0))
        transfer = _volume_palette(volume_params, field, ramp)
        sigma_t = base_sigma_t * transfer[3]
        if sigma_t < 1.0e-6:
            t += step_size
            steps += wp.int32(1)
            continue
        alpha = 1.0 - wp.exp(-sigma_t * step_size)
        color = wp.vec3(transfer[0], transfer[1], transfer[2])
        gradient = wp.vec3(
            wp.abs(_sample_volume(volume_params, index + wp.vec3(1.0, 0.0, 0.0))[0])
            - wp.abs(_sample_volume(volume_params, index - wp.vec3(1.0, 0.0, 0.0))[0]),
            wp.abs(_sample_volume(volume_params, index + wp.vec3(0.0, 1.0, 0.0))[0])
            - wp.abs(_sample_volume(volume_params, index - wp.vec3(0.0, 1.0, 0.0))[0]),
            wp.abs(_sample_volume(volume_params, index + wp.vec3(0.0, 0.0, 1.0))[0])
            - wp.abs(_sample_volume(volume_params, index - wp.vec3(0.0, 0.0, 1.0))[0]),
        )
        edge_light = 0.45
        if wp.dot(gradient, gradient) > 1.0e-8:
            edge_light += 0.9 * wp.abs(wp.dot(wp.normalize(gradient), sun_direction))
        lighting = (ambient + sun_radiance * phase * wp.clamp(sample[2], 0.0, 1.0)) * edge_light
        source = wp.cw_mul(color, lighting) + color * volume_params.emission
        result.radiance += result.transmittance * alpha * source
        result.transmittance *= 1.0 - alpha
        t += step_size
        steps += wp.int32(1)
    return result


@wp.kernel(enable_backward=False)
def composite_volume(params: PathtraceLaunchParams, volume_params: VolumeParams):
    y, x = wp.tid()
    if x >= int(params.width) or y >= int(params.height):
        return
    origin = _compute_ray_origin(params)
    direction = _compute_ray_dir(params, x, y)
    depth = params.depth_output[y, x]
    surface_t = wp.float32(1.0e32)
    if depth < MISS_DISTANCE:
        view_direction = _mul_mat3x3_cm(params.view, direction)
        surface_t = depth / wp.max(-view_direction[2], 1.0e-5)
    rng = _xxhash32(wp.uint32(x), wp.uint32(y), params.frame_index)
    volume = _integrate_volume(params, volume_params, origin, direction, surface_t, _pcg_rand01(rng))
    color = params.color_output[y, x]
    rgb = volume.radiance + volume.transmittance * wp.vec3(color[0], color[1], color[2])
    params.color_output[y, x] = wp.vec4(rgb[0], rgb[1], rgb[2], color[3])
