# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named layouts and GPU dispatch, ported from the public SBT builder."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import warp as wp

import warp_optix as wo


@wp.struct
class Payload:
    value: wp.uint32


@wp.kernel(module="unique")
def raygen(output: wp.array[wp.uint32], scene: wp.uint64, stride: wp.uint32, radiance: wp.uint32, shadow: wp.uint32):
    index = wp.tid()
    ray = radiance
    if index % 2 == 1:
        ray = shadow
    payload = Payload()
    wp.optix_trace(
        scene,
        wp.vec3(float(index // 2) * 2.0 + 0.25, 0.25, 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.0,
        100.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        ray,
        stride,
        ray,
        payload,
    )
    output[index] = payload.value


@wp.kernel(module="unique")
def radiance_miss():
    payload = Payload()
    payload.value = wp.uint32(90)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def shadow_miss():
    payload = Payload()
    payload.value = wp.uint32(91)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def glass_hit():
    payload = Payload()
    payload.value = wp.uint32(10)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def glass_shadow():
    payload = Payload()
    payload.value = wp.uint32(11)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def metal_hit():
    payload = Payload()
    payload.value = wp.uint32(20)
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def metal_shadow():
    payload = Payload()
    payload.value = wp.uint32(21)
    wp.optix_store_payload(payload)


def fill(builder):
    builder.set_miss("radiance", radiance_miss)
    builder.set_miss("shadow", shadow_miss)
    return builder.add_hit_group(
        "glass", radiance=wo.HitGroup(closest_hit=glass_hit), shadow=wo.HitGroup(closest_hit=glass_shadow)
    )


@pytest.mark.parametrize("names", [[], [""], ["a", "a"], [None], [str(i) for i in range(16)]])
def test_invalid_ray_types(names):
    with pytest.raises(ValueError):
        wo.SbtBuilder(names)


def test_named_layout_snapshot_and_foreign_handles():
    builder = wo.SbtBuilder(("radiance", "shadow"))
    glass = fill(builder)
    layout = builder.snapshot()
    other = fill(wo.SbtBuilder(("radiance", "shadow")))
    assert layout.get_sbt_offset(glass) == 0
    assert layout.ray_type_index("shadow") == 1
    assert layout.trace_stride == 2
    with pytest.raises(ValueError, match="belong"):
        layout.get_sbt_offset(other)
    with pytest.raises(KeyError, match="Unknown ray"):
        layout.ray_type_index("typo")
    with pytest.raises(FrozenInstanceError):
        layout.ray_types = ("changed",)
    builder.set_miss("shadow", radiance_miss)
    metal = builder.add_hit_group(
        "metal", radiance=wo.HitGroup(closest_hit=metal_hit), shadow=wo.HitGroup(closest_hit=metal_shadow)
    )
    assert metal.sbt_offset == 2
    assert len(layout.hit_groups) == 1
    assert "shadow_miss" in layout.describe()
    assert "shadow_miss" not in builder.describe()
    assert "glass" in layout.describe() and "glass_shadow" in layout.describe()
    with pytest.raises(ValueError, match="belong"):
        layout.get_sbt_offset(metal)


def test_missing_and_incompatible_assignments():
    builder = wo.SbtBuilder(("radiance", "shadow"))
    with pytest.raises(ValueError, match="Missing miss.*radiance.*shadow"):
        builder.describe()
    with pytest.raises(KeyError, match="Unknown ray"):
        builder.set_miss("typo", None)
    with pytest.raises(ValueError, match="ordinary Warp kernel"):
        builder.set_miss("radiance", "__miss__entry")
    with pytest.raises(ValueError, match="glass.*shadow.*typo"):
        builder.add_hit_group("glass", radiance=wo.HitGroup(), typo=wo.HitGroup())
    with pytest.raises(ValueError, match="same geometry kind"):
        builder.add_hit_group(
            "glass", radiance=wo.HitGroup(builtin=wo.PrimitiveType.ROUND_LINEAR), shadow=wo.HitGroup()
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        builder.add_hit_group(
            "glass",
            radiance=wo.HitGroup(intersection=glass_hit, builtin=wo.PrimitiveType.ROUND_LINEAR),
            shadow=wo.HitGroup(),
        )
    fill(builder)
    with pytest.raises(ValueError, match="Duplicate"):
        fill(builder)


def test_empty_programs_for_hit_object_traversal():
    builder = wo.SbtBuilder()
    builder.set_miss("radiance", None)
    handle = builder.add_hit_group("triangles", radiance=wo.HitGroup())
    assert builder.snapshot().get_sbt_offset(handle) == 0
    assert "(none)" in builder.describe()


@pytest.mark.parametrize("options", [{"miss": (None,)}, {"hit_groups": ()}, {"ray_types": 1}])
def test_builder_rejects_conflicting_options_before_compiling(options):
    with pytest.raises(ValueError, match="cannot be combined"):
        wo.create_pipeline(None, raygen, sbt=wo.SbtBuilder(), **options)


@pytest.mark.parametrize("count", [0, -1, 16, 1.5, True])
def test_invalid_trace_stride_is_rejected(count):
    with pytest.raises(ValueError, match="ray_types"):
        wo.create_pipeline(None, raygen, ray_types=count)


@wp.kernel(module="unique")
def write_instances(gas: wp.uint64, offsets: wp.array[wp.uint32], records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(
        wp.identity(3, float),
        wp.vec3(float(i) * 2.0, 0.0, 0.0),
        gas,
        wp.uint32(i),
        offsets[i],
        wp.uint32(255),
        wp.uint32(0),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("rays", [("radiance", "shadow"), ("shadow", "radiance")])
@pytest.mark.parametrize("use_snapshot", [False, True])
def test_gpu_named_ray_and_instance_dispatch(rays, use_snapshot):
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    device = wp.get_device("cuda")
    ctx = wo.create_context(device)
    try:
        builder = wo.SbtBuilder(rays)
        glass = fill(builder)
        metal = builder.add_hit_group(
            "metal", radiance=wo.HitGroup(closest_hit=metal_hit), shadow=wo.HitGroup(closest_hit=metal_shadow)
        )
        pipeline = wo.create_pipeline(
            ctx, raygen, sbt=builder.snapshot() if use_snapshot else builder, payload_words=1, device=device
        )
        layout = pipeline.sbt_layout
        assert layout.get_sbt_offset(metal) == 2
        vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device=device)
        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
        gas = wo.build_accel(ctx, [wo.triangles(vertices, indices)], device=device)
        offsets = wp.array([layout.get_sbt_offset(glass), layout.get_sbt_offset(metal)], dtype=wp.uint32, device=device)
        records = wp.empty(2, dtype=wo.Instance, device=device)
        wp.launch(write_instances, dim=2, inputs=[gas.handle, offsets, records], device=device)
        ias = wo.build_accel(ctx, [wo.instances(records)], device=device)
        # Builder edits must not change the existing pipeline's layout or programs.
        builder.set_miss("radiance", shadow_miss)
        output = wp.zeros(6, dtype=wp.uint32, device=device)
        wo.launch(
            pipeline,
            dim=(6,),
            inputs=[
                output,
                ias.handle,
                layout.trace_stride,
                layout.ray_type_index("radiance"),
                layout.ray_type_index("shadow"),
            ],
        )
        np.testing.assert_array_equal(output.numpy(), [10, 11, 20, 21, 90, 91])
    finally:
        wo.destroy_context(ctx, device)
