# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named SBT layouts, ownership, and actual multi-ray GPU dispatch."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import warp_optix as wo


def builder(ray_types=("radiance", "shadow")):
    return wo.SbtBuilder(SimpleNamespace(_check_open=lambda: None), ray_types)


def fill(sbt):
    for ray in sbt.ray_types:
        sbt.set_miss(ray, f"__miss__{ray}")
    return sbt.add_hit_group(
        "glass",
        **{
            ray: wo.HitKernel(closest_hit=f"__closesthit__{ray}")
            for ray in sbt.ray_types
        },
    )


@pytest.mark.parametrize(
    "names", [[], [""], ["a", "a"], [None], [str(i) for i in range(16)]]
)
def test_invalid_ray_types(names):
    with pytest.raises(ValueError):
        builder(names)


def test_named_layout_snapshot_and_foreign_handles():
    sbt = builder()
    glass = fill(sbt)
    layout = sbt._snapshot()
    other = fill(builder())
    with pytest.raises(ValueError, match="belong"):
        layout.get_sbt_offset(other)
    assert layout.get_sbt_offset(glass) == 0
    assert layout.ray_type_index("shadow") == 1
    assert layout.trace_stride == 2
    with pytest.raises(KeyError, match="Unknown ray type"):
        layout.ray_type_index("typo")
    with pytest.raises(FrozenInstanceError):
        layout.ray_types = ("changed",)
    sbt.set_miss("shadow", "__miss__new")
    metal = sbt.add_hit_group(
        "metal",
        radiance=wo.HitKernel(closest_hit="__closesthit__metal"),
        shadow=wo.HitKernel(any_hit="__anyhit__metal"),
    )
    assert metal.sbt_offset == 2
    assert len(layout.hit_groups) == 1
    assert "__miss__new" not in layout.describe()
    assert "__miss__new" in sbt.describe()
    assert "glass" in layout.describe() and "__closesthit__shadow" in layout.describe()
    with pytest.raises(ValueError, match="belong"):
        layout.get_sbt_offset(metal)


def test_missing_and_incompatible_assignments():
    sbt = builder()
    with pytest.raises(ValueError, match="Missing miss.*radiance.*shadow"):
        sbt.describe()
    with pytest.raises(KeyError, match="Unknown ray"):
        sbt.set_miss("typo", "__miss__ms")
    with pytest.raises(ValueError, match="glass.*shadow.*typo"):
        sbt.add_hit_group(
            "glass",
            radiance=wo.HitKernel(closest_hit="ch"),
            typo=wo.HitKernel(any_hit="ah"),
        )
    with pytest.raises(ValueError, match="glass.*radiance.*at least one"):
        sbt.add_hit_group(
            "glass", radiance=wo.HitKernel(), shadow=wo.HitKernel(any_hit="ah")
        )
    with pytest.raises(ValueError, match="same geometry kind"):
        sbt.add_hit_group(
            "glass",
            radiance=wo.HitKernel(intersection="is"),
            shadow=wo.HitKernel(any_hit="ah"),
        )
    fill(sbt)
    with pytest.raises(ValueError, match="Duplicate"):
        fill(sbt)


@wp.struct
class Params:
    output: wp.array(dtype=wp.uint32)
    traversable: wp.uint64
    stride: wp.uint32
    radiance: wp.uint32
    shadow: wp.uint32


@wp.struct
class Payload:
    value: wp.uint32


@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def raygen(params: Params):
    index = int(wp.optix_get_launch_index()[0])
    ray_type = params.radiance
    if index % 2 == 1:
        ray_type = params.shadow
    payload = Payload()
    payload.value = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        wp.vec3(float(index // 2) * 2.0 + 0.25, 0.25, 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.0,
        100.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        ray_type,
        params.stride,
        ray_type,
        payload,
    )
    params.output[index] = payload.value


@wo.optix_kernel(wo.OptixKernelType.MISS)
def radiance_miss(params: Params):
    wp.optix_set_payload_0(wp.uint32(90))


@wo.optix_kernel(wo.OptixKernelType.MISS)
def shadow_miss(params: Params):
    wp.optix_set_payload_0(wp.uint32(91))


@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def glass_hit(params: Params):
    wp.optix_set_payload_0(wp.uint32(10))


@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def glass_shadow(params: Params):
    wp.optix_set_payload_0(wp.uint32(11))


@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def metal_hit(params: Params):
    wp.optix_set_payload_0(wp.uint32(20))


@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def metal_shadow(params: Params):
    wp.optix_set_payload_0(wp.uint32(21))


def test_kernel_validation_and_preview_before_compilation():
    sbt = builder(("name",))
    sbt.set_miss("name", radiance_miss)
    sbt.add_hit_group("material", name=wo.HitKernel(closest_hit=glass_hit))
    assert "glass_hit" in sbt.describe()
    with pytest.raises(ValueError, match="ray type 'name'"):
        sbt.set_miss("name", glass_hit)
    with pytest.raises(ValueError, match="bad.*name"):
        sbt.add_hit_group("bad", name=wo.HitKernel(closest_hit=radiance_miss))


@pytest.fixture
def context():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    with wo.Context("cuda:0", log_level=1) as ctx:
        yield ctx


@pytest.mark.parametrize("rays", [("radiance", "shadow"), ("shadow", "radiance")])
def test_gpu_named_ray_and_instance_dispatch(context, rays):
    sbt = context.create_sbt_builder(rays)
    sbt.set_miss("radiance", radiance_miss)
    sbt.set_miss("shadow", shadow_miss)
    glass = sbt.add_hit_group(
        "glass",
        radiance=wo.HitKernel(closest_hit=glass_hit),
        shadow=wo.HitKernel(closest_hit=glass_shadow),
    )
    metal = sbt.add_hit_group(
        "metal",
        radiance=wo.HitKernel(closest_hit=metal_hit),
        shadow=wo.HitKernel(closest_hit=metal_shadow),
    )
    ptx = wo.compile_warp_module_to_ptx(
        wp.get_module(__name__), "", "sbt_builder", __file__, context.device
    )
    pipeline = context.create_pipeline(
        ptx,
        raygen,
        sbt=sbt,
        num_payload_values=1,
        num_attribute_values=2,
        traversable_graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
    )
    assert pipeline.hit_group_offsets == (0, 2)
    assert pipeline.sbt.get_sbt_offset(metal) == 2
    geometry = context.create_triangle_gas(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
    )
    ias = context.create_instance_acceleration_structure(
        [
            wo.Instance(geometry, hit_group=glass),
            wo.Instance(
                geometry,
                hit_group=metal,
                transform=(1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0),
            ),
        ]
    )
    # Mutating the builder must not change this pipeline's shaders or layout.
    sbt.set_miss("radiance", shadow_miss)
    params = Params()
    params.output = wp.zeros(6, dtype=wp.uint32, device=context.device)
    params.traversable = ias.handle
    params.stride = pipeline.sbt.trace_stride
    params.radiance = pipeline.sbt.ray_type_index("radiance")
    params.shadow = pipeline.sbt.ray_type_index("shadow")
    buffer = wo.create_launch_params_buffer(Params, context.device)
    wo.write_launch_params(buffer, params)
    pipeline.launch(buffer, width=6)
    np.testing.assert_array_equal(params.output.numpy(), [10, 11, 20, 21, 90, 91])


def test_context_ownership_and_conflicting_options(context):
    sbt = context.create_sbt_builder()
    handle = fill(sbt)
    with pytest.raises(ValueError, match="cannot be combined"):
        context.create_pipeline(
            b"",
            raygen,
            radiance_miss,
            sbt=sbt,
            num_payload_values=1,
            num_attribute_values=2,
        )
    with wo.Context(context.device, log_level=1) as other:
        with pytest.raises(ValueError, match="belong"):
            other.create_pipeline(
                b"", raygen, sbt=sbt, num_payload_values=1, num_attribute_values=2
            )
        geometry = other.create_triangle_gas(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )
        with pytest.raises(ValueError, match="belong"):
            other.create_instance_acceleration_structure(
                [wo.Instance(geometry, hit_group=handle)]
            )
        with pytest.raises(ValueError, match="cannot be combined"):
            other.create_instance_acceleration_structure(
                [wo.Instance(geometry, hit_group=handle, sbt_offset=2)]
            )
        first = fill(other.create_sbt_builder())
        second = fill(other.create_sbt_builder())
        with pytest.raises(ValueError, match="same SBT builder"):
            other.create_instance_acceleration_structure(
                [
                    wo.Instance(geometry, hit_group=first),
                    wo.Instance(geometry, hit_group=second),
                ]
            )
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        sbt.set_miss("radiance", radiance_miss)
