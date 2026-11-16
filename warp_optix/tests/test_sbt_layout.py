# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named SBT layouts and GPU dispatch."""

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
        100.0,
        payload,
        0.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        ray,
        stride,
        ray,
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


def layout(rays=("radiance", "shadow"), **groups):
    glass = {"radiance": wo.HitGroup(closest_hit=glass_hit), "shadow": wo.HitGroup(closest_hit=glass_shadow)}
    return wo.SbtLayout(
        ray_types=rays,
        miss={"radiance": radiance_miss, "shadow": shadow_miss},
        hit_groups={"glass": glass, **groups},
    )


METAL = {"radiance": wo.HitGroup(closest_hit=metal_hit), "shadow": wo.HitGroup(closest_hit=metal_shadow)}


def test_named_offsets_and_description():
    sbt = layout(metal=METAL)
    assert sbt.get_sbt_offset("glass") == 0
    assert sbt.get_sbt_offset("metal") == 2
    assert sbt.ray_type_index("shadow") == 1
    assert sbt.trace_stride == 2
    with pytest.raises(KeyError, match="Unknown hit group"):
        sbt.get_sbt_offset("typo")
    with pytest.raises(KeyError, match="Unknown ray"):
        sbt.ray_type_index("typo")
    with pytest.raises(FrozenInstanceError):
        sbt.ray_types = ("changed",)
    with pytest.raises(TypeError):
        sbt.hit_groups["copper"] = METAL
    description = sbt.describe()
    assert "shadow_miss" in description and "glass_shadow" in description and "metal" in description
    # Indices follow the declared ray order; offsets count records per group.
    reversed_sbt = layout(("shadow", "radiance"), metal=METAL)
    assert reversed_sbt.ray_type_index("radiance") == 1 and reversed_sbt.get_sbt_offset("metal") == 2


def test_bare_entries_apply_to_every_ray_type():
    sky = Payload()
    sbt = wo.SbtLayout(
        ("radiance", "shadow"), miss=wo.Record(radiance_miss, data=sky), hit_groups={"empty": wo.HitGroup()}
    )
    assert list(sbt.miss) == ["radiance", "shadow"]
    assert all(entry.data is sky for entry in sbt.miss.values())
    assert list(sbt.hit_groups["empty"]) == ["radiance", "shadow"]
    assert "radiance_miss" in sbt.describe()
    default = wo.SbtLayout()
    assert default.ray_types == ("radiance",)
    assert list(default.hit_groups) == ["default"]
    assert default.miss == {"radiance": None}
    assert "(none)" in default.describe()


def test_callable_kind_is_part_of_compiled_module_identity(monkeypatch):
    from warp_optix._src.pipeline import _module_name
    from warp_optix._src.program import params_struct

    params = params_struct(raygen)
    direct = _module_name(params, raygen, [], [], [radiance_miss], None)
    monkeypatch.setattr(radiance_miss, "_optix_kind", wo.ProgramKind.CONTINUATION_CALLABLE, raising=False)
    continuation = _module_name(params, raygen, [], [], [radiance_miss], None)
    assert continuation != direct


def test_missing_and_incompatible_entries():
    for names in ([], [""], ["a", "a"], [None], [str(i) for i in range(16)]):
        with pytest.raises(ValueError):
            wo.SbtLayout(names, miss=None, hit_groups={})
    with pytest.raises(ValueError, match="Miss.*missing.*shadow"):
        wo.SbtLayout(("radiance", "shadow"), miss={"radiance": None})
    with pytest.raises(TypeError, match="Warp kernel"):
        wo.SbtLayout(miss="__miss__entry")
    with pytest.raises(ValueError, match="glass.*shadow.*typo"):
        layout(glass={"radiance": wo.HitGroup(), "typo": wo.HitGroup()})
    with pytest.raises(TypeError, match="glass.*HitGroup"):
        wo.SbtLayout(hit_groups={"glass": radiance_miss})
    with pytest.raises(TypeError, match="Warp kernels"):
        wo.SbtLayout(hit_groups={"glass": wo.HitGroup(closest_hit="__closesthit__glass")})
    with pytest.raises(ValueError, match="same geometry kind"):
        layout(glass={"radiance": wo.HitGroup(builtin=wo.PrimitiveType.ROUND_LINEAR), "shadow": wo.HitGroup()})
    with pytest.raises(ValueError, match="cannot be combined"):
        wo.SbtLayout(hit_groups={"glass": wo.HitGroup(intersection=glass_hit, builtin=wo.PrimitiveType.ROUND_LINEAR)})
    with pytest.raises(ValueError, match="nonempty"):
        wo.SbtLayout(hit_groups={"": wo.HitGroup()})
    with pytest.raises(TypeError, match="SbtLayout"):
        wo.Pipeline(raygen, sbt={"miss": None})


@wp.kernel(module="unique")
def write_instances(gas: wp.uint64, offsets: wp.array[wp.uint32], records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(
        wp.identity(3, float), wp.vec3(float(i) * 2.0, 0.0, 0.0), gas, wp.uint32(i), offsets[i], wp.uint32(255)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")
def test_gpu_named_ray_and_instance_dispatch():
    device = wp.get_device("cuda")
    # Reversed ray order, so the dispatch relies on the layout's indices rather than declaration order.
    pipeline = wo.Pipeline(raygen, sbt=layout(("shadow", "radiance"), metal=METAL), payload_words=1, device=device)
    sbt = pipeline.sbt_layout
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=wp.uint32, device=device)
    gas = wo.Accel([wo.triangles(vertices, indices)], device=device)
    offsets = wp.array([sbt.get_sbt_offset("glass"), sbt.get_sbt_offset("metal")], dtype=wp.uint32, device=device)
    records = wp.empty(2, dtype=wo.Instance, device=device)
    wp.launch(write_instances, dim=2, inputs=[gas.id, offsets, records], device=device)
    ias = wo.Accel([wo.instances(records, [gas])], device=device)
    output = wp.empty(6, dtype=wp.uint32, device=device)
    wo.launch(
        pipeline,
        dim=(6,),
        inputs=[
            output,
            ias,
            sbt.trace_stride,
            sbt.ray_type_index("radiance"),
            sbt.ray_type_index("shadow"),
        ],
    )
    np.testing.assert_array_equal(output.numpy(), [10, 11, 20, 21, 90, 91])
