# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from warp_optix.pathtracing.pathtracing_warp_kernels import (
    PathtraceLaunchParams,
    PhysicalSkyParams,
    SphereLight,
    ShadedHitData,
    _set_incident_medium_for_hit,
    _bsdf_evaluate,
    _eval_physical_sky,
    _bsdf_sample,
    _filter_roughness_for_normal_map,
    _intersect_sphere_lights,
    _material_f0,
    _sky_star_radiance,
    _sample_sphere_light,
)


@wp.kernel
def _integrate_lambertian_sphere_light(
    params: PathtraceLaunchParams,
    albedo: wp.float32,
    output: wp.array(dtype=wp.vec3),
):
    sample_index = wp.tid()
    sample_count = float(output.shape[0])
    sample = _sample_sphere_light(
        params,
        wp.vec3(0.0, 0.0, 0.0),
        0.5,
        (float(sample_index) + 0.5) / sample_count,
        0.5,
    )
    cosine = wp.max(sample.direction[2], 0.0)
    output[sample_index] = sample.radiance * (albedo * cosine / wp.pi) / sample.pdf


@wp.kernel
def _evaluate_night_sky(
    sky: PhysicalSkyParams,
    moon_direction: wp.vec3,
    output: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    direction = moon_direction
    if index == 1:
        direction = wp.normalize(moon_direction + wp.vec3(0.02, 0.0, 0.0))
    output[index] = _eval_physical_sky(sky, direction, wp.bool(True), wp.bool(True))


@wp.kernel
def _sample_star_field(output: wp.array(dtype=wp.float32)):
    index = wp.tid()
    count = wp.float32(output.shape[0])
    z = (wp.float32(index) + 0.5) / count
    radius = wp.sqrt(wp.max(1.0 - z * z, 0.0))
    phi = wp.float32(index) * 2.39996323
    direction = wp.vec3(radius * wp.cos(phi), radius * wp.sin(phi), z)
    star = _sky_star_radiance(direction)
    output[index] = wp.max(star[0], wp.max(star[1], star[2]))


@wp.kernel
def _sample_clear_thin_transmission(
    roughness: wp.float32, output: wp.array(dtype=wp.vec4)
):
    sample = _bsdf_sample(
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.2, 0.1, 0.05),
        wp.vec3(1.0, 1.0, 1.0),
        wp.vec3(1.0, 1.0, 1.0),
        roughness,
        0.0,
        0.0,
        0.0,
        wp.vec3(0.0, 0.0, 0.0),
        1.0,
        1.0,
        1.0,
        0.0,
        wp.vec3(1.0, 1.0, 1.0),
        0.0,
        0.1,
        1,
        0.2,
        0.7,
        0.3,
    )
    output[0] = wp.vec4(
        sample.direction[0],
        sample.direction[1],
        sample.direction[2],
        float(sample.event_type),
    )
    output[1] = wp.vec4(
        sample.bsdf_over_pdf[0],
        sample.bsdf_over_pdf[1],
        sample.bsdf_over_pdf[2],
        sample.pdf,
    )


@wp.kernel
def _evaluate_backside_thin_transmission(output: wp.array(dtype=wp.vec4)):
    evaluated = _bsdf_evaluate(
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.5, 0.5, 0.5),
        wp.vec3(1.0, 1.0, 1.0),
        wp.vec3(1.0, 1.0, 1.0),
        0.04,
        0.0,
        1.0,
        0.0,
        wp.vec3(0.0, 0.0, 0.0),
        1.0,
        1.2,
        1.0,
        0.0,
        wp.vec3(1.0, 1.0, 1.0),
        0.0,
        0.1,
        1.0,
        1,
        0.5,
    )
    value = evaluated.bsdf_diffuse + evaluated.bsdf_glossy
    output[0] = wp.vec4(value[0], value[1], value[2], evaluated.pdf)


@wp.kernel
def _trace_authored_lamp_light(
    params: PathtraceLaunchParams, output: wp.array(dtype=wp.vec4)
):
    hit = _intersect_sphere_lights(
        params,
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        100.0,
    )
    output[0] = wp.vec4(hit.radiance[0], hit.radiance[1], hit.radiance[2], hit.distance)
    output[1] = wp.vec4(hit.pdf, 0.0, 0.0, 0.0)


@wp.kernel
def _filter_wool_normal_roughness(output: wp.array(dtype=wp.float32)):
    output[0] = _filter_roughness_for_normal_map(0.04, wp.vec3(0.0, 0.0, 1.0))
    output[1] = _filter_roughness_for_normal_map(0.04, wp.vec3(0.1, 0.0, 0.7))


@wp.kernel
def _evaluate_material_f0(output: wp.array(dtype=wp.vec3)):
    output[0] = _material_f0(wp.vec3(1.0), wp.vec3(1.0), 1.0, 0.0, 1.0, 1.5, 0.0)
    output[1] = _material_f0(wp.vec3(1.0), wp.vec3(1.0), 0.5, 0.0, 1.0, 1.1, 0.0)
    output[2] = _material_f0(
        wp.vec3(0.2, 0.4, 0.8), wp.vec3(1.0), 1.0, 1.0, 1.0, 1.5, 0.0
    )


def test_material_f0_matches_dielectric_glass_and_metal_limits():
    """Supply physical normal-incidence reflectance to reconstruction."""
    output = wp.zeros(3, dtype=wp.vec3, device="cpu")

    wp.launch(_evaluate_material_f0, dim=1, inputs=[output], device="cpu")

    default_dielectric, low_ior_glass, metal = output.numpy()
    np.testing.assert_allclose(default_dielectric, 0.04, rtol=1.0e-5)
    expected_glass = 0.5 * ((1.1 - 1.0) / (1.1 + 1.0)) ** 2
    np.testing.assert_allclose(low_ior_glass, expected_glass, rtol=1.0e-5)
    np.testing.assert_allclose(metal, (0.2, 0.4, 0.8), rtol=1.0e-5)


def test_sphere_light_matches_lambertian_solid_angle_integral():
    """Match the analytic radiance of a centered sphere light on diffuse white."""
    distance = 2.0
    radius = 0.5
    albedo = 0.8
    radiance = np.asarray((5.0, 4.0, 3.0), dtype=np.float32)

    light = SphereLight()
    light.position_radius = wp.vec4(0.0, 0.0, distance, radius)
    light.radiance = wp.vec3(*radiance)
    light.pad = 0.0
    params = PathtraceLaunchParams()
    params.sphere_lights = wp.array([light], dtype=SphereLight, device="cpu")
    params.sphere_light_count = 1
    params.analytic_light_intensity = 1.0

    output = wp.zeros(4096, dtype=wp.vec3, device="cpu")
    wp.launch(
        _integrate_lambertian_sphere_light,
        dim=output.shape[0],
        inputs=[params, albedo, output],
        device="cpu",
    )

    # The cosine-weighted solid-angle integral of a centered spherical light
    # is pi*r^2/d^2; Lambert's 1/pi factor cancels pi.
    expected = radiance * albedo * (radius / distance) ** 2
    np.testing.assert_allclose(output.numpy().mean(axis=0), expected, rtol=2.0e-5)


def test_night_sky_contains_antipodal_moon_disk():
    """Emit a finite moon disk opposite a below-horizon sun."""
    sky = PhysicalSkyParams()
    sky.rgb_unit_conversion = wp.vec3(1.0 / 80000.0)
    sky.multiplier = 0.01
    sky.haze = 0.0
    sky.redblueshift = 0.0
    sky.saturation = 0.5
    sky.horizon_height = 0.0
    sky.ground_color = wp.vec3(0.0)
    sky.horizon_blur = 1.0
    sky.night_color = wp.vec3(0.002, 0.004, 0.01)
    sky.sun_disk_intensity = 1.0
    sky.sun_direction = wp.vec3(0.0, -0.5, 0.8660254)
    sky.sun_disk_scale = 1.0
    sky.sun_glow_intensity = 0.15
    sky.y_is_up = 1
    sky.grayscale = 0
    moon_direction = wp.vec3(0.0, 0.5, -0.8660254)
    output = wp.zeros(2, dtype=wp.vec3, device="cpu")

    wp.launch(
        _evaluate_night_sky,
        dim=2,
        inputs=[sky, moon_direction, output],
        device="cpu",
    )

    center, outside = output.numpy()
    assert center.min() > outside.max()
    np.testing.assert_allclose(
        center - outside, (0.04, 0.044, 0.05), rtol=0.02, atol=5.0e-4
    )


def test_procedural_star_field_is_sparse_and_nonblack():
    """Generate a stable sparse star field over the night hemisphere."""
    output = wp.zeros(32768, dtype=wp.float32, device="cpu")

    wp.launch(
        _sample_star_field,
        dim=output.shape[0],
        inputs=[output],
        device="cpu",
    )

    stars = output.numpy()
    visible_count = np.count_nonzero(stars > 0.0)
    assert 5 <= visible_count <= 80
    assert stars.max() >= 0.02


def test_clear_thin_transmission_stays_straight():
    """Keep smooth thin-wall transmission straight and stable."""
    output = wp.zeros(2, dtype=wp.vec4, device="cpu")

    wp.launch(
        _sample_clear_thin_transmission,
        dim=1,
        inputs=[0.0, output],
        device="cpu",
    )

    direction_event, throughput_pdf = output.numpy()
    np.testing.assert_allclose(direction_event[:3], (0.0, 0.0, -1.0), atol=1.0e-6)
    assert direction_event[3] == 18.0
    np.testing.assert_allclose(throughput_pdf[:3], (1.0, 1.0, 1.0), atol=1.0e-6)
    assert throughput_pdf[3] == 1.0


def test_rough_thin_transmission_retains_microfacet_scattering():
    """Preserve the reference GGX pseudo-BTDF for rough thin surfaces."""
    output = wp.zeros(2, dtype=wp.vec4, device="cpu")

    wp.launch(
        _sample_clear_thin_transmission,
        dim=1,
        inputs=[0.36, output],
        device="cpu",
    )

    direction_event, throughput_pdf = output.numpy()
    assert np.linalg.norm(direction_event[:2]) > 0.05
    assert direction_event[2] < 0.0
    assert direction_event[3] == 18.0
    assert np.all(np.isfinite(throughput_pdf))
    assert throughput_pdf[3] > 0.0


def test_backside_light_contributes_through_thin_transmission():
    """Evaluate direct light arriving through the back of a thin dielectric."""
    output = wp.zeros(1, dtype=wp.vec4, device="cpu")

    wp.launch(
        _evaluate_backside_thin_transmission,
        dim=1,
        inputs=[output],
        device="cpu",
    )

    assert np.all(output.numpy()[0] > 0.0)


def test_authored_counter_lamp_is_visible_through_secondary_rays():
    """Intersect the finite emitter used inside each authored counter lamp."""
    radius = 0.04
    distance = 0.20
    color = np.asarray((0.84942085, 0.64010745, 0.47554448), dtype=np.float32)
    radiance = color * 5.0

    light = SphereLight()
    light.position_radius = wp.vec4(0.0, 0.0, distance, radius)
    light.radiance = wp.vec3(*radiance)
    light.pad = 0.0
    params = PathtraceLaunchParams()
    params.sphere_lights = wp.array([light], dtype=SphereLight, device="cpu")
    params.sphere_light_count = 1
    params.analytic_light_intensity = 1.0
    output = wp.zeros(2, dtype=wp.vec4, device="cpu")

    wp.launch(
        _trace_authored_lamp_light,
        dim=1,
        inputs=[params, output],
        device="cpu",
    )

    hit, pdf = output.numpy()
    np.testing.assert_allclose(hit[:3], radiance, rtol=1.0e-6)
    np.testing.assert_allclose(hit[3], distance - radius, rtol=1.0e-6)
    cos_theta_max = np.sqrt(1.0 - radius * radius / (distance * distance))
    expected_pdf = 1.0 / (2.0 * np.pi * (1.0 - cos_theta_max))
    np.testing.assert_allclose(pdf[0], expected_pdf, rtol=1.0e-5)


def test_filtered_wool_normal_broadens_unresolved_specular_lobe():
    """Broaden roughness when a wool normal-map mip contains unresolved variance."""
    output = wp.zeros(2, dtype=wp.float32, device="cpu")

    wp.launch(
        _filter_wool_normal_roughness,
        dim=1,
        inputs=[output],
        device="cpu",
    )

    resolved, filtered = output.numpy()
    np.testing.assert_allclose(resolved, 0.04, atol=1.0e-7)
    assert filtered > resolved


@wp.kernel
def _select_backface_incident_medium(output: wp.array(dtype=wp.vec4)):
    hit = ShadedHitData()
    hit.valid = wp.uint32(1)
    hit.normal = wp.vec3(0.1, 0.2, 0.3)
    hit.Ng = wp.vec3(0.0, 0.0, 1.0)
    hit.T = wp.vec3(1.0, 0.0, 0.0)
    hit.B = wp.vec3(0.0, 1.0, 0.0)
    hit.Nc = wp.vec3(0.0, 0.0, 1.0)
    hit.ior1 = 1.0
    hit.ior2 = 1.5
    hit.is_thin_walled = 0

    back = _set_incident_medium_for_hit(hit, wp.uint32(0))
    output[0] = wp.vec4(back.normal[0], back.normal[1], back.normal[2], back.ior1)
    output[1] = wp.vec4(back.Ng[0], back.Ng[1], back.Ng[2], back.ior2)
    output[2] = wp.vec4(back.T[0], back.T[1], back.T[2], 0.0)
    output[3] = wp.vec4(back.B[0], back.B[1], back.B[2], 0.0)
    output[4] = wp.vec4(back.Nc[0], back.Nc[1], back.Nc[2], 0.0)

    hit.is_thin_walled = 1
    thin = _set_incident_medium_for_hit(hit, wp.uint32(0))
    output[5] = wp.vec4(thin.ior1, thin.ior2, 0.0, 0.0)


def test_backface_keeps_faceforward_frame_and_selects_volume_medium():
    """Do not double-flip frames that hit geometry already face-forwarded."""
    output = wp.zeros(6, dtype=wp.vec4, device="cpu")

    wp.launch(_select_backface_incident_medium, dim=1, inputs=[output], device="cpu")

    normal, geometric, tangent, bitangent, clearcoat, thin_iors = output.numpy()
    np.testing.assert_allclose(normal[:3], (0.1, 0.2, 0.3), atol=1.0e-7)
    np.testing.assert_allclose(geometric[:3], (0.0, 0.0, 1.0), atol=1.0e-7)
    np.testing.assert_allclose(tangent[:3], (1.0, 0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(bitangent[:3], (0.0, 1.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(clearcoat[:3], (0.0, 0.0, 1.0), atol=1.0e-7)
    np.testing.assert_allclose((normal[3], geometric[3]), (1.5, 1.0), atol=1.0e-7)
    np.testing.assert_allclose(thin_iors[:2], (1.0, 1.5), atol=1.0e-7)
