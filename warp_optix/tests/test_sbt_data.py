# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-record SBT data read with wp.optix_load_sbt_data()."""

import gc
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]


@wp.struct
class Payload:
    color: wp.vec3


@wp.struct
class Flat:
    color: wp.vec3


@wp.struct
class Inner:
    flag: wp.uint8
    value: wp.float64


# Mixed sizes and alignments; the array field also checks that referenced arrays stay alive.
@wp.struct
class Textured:
    flag: wp.bool
    inner: Inner
    scale: wp.float32
    tint: wp.mat22
    palette: wp.array(dtype=wp.vec3)
    index: wp.int64


@wp.struct
class Sky:
    color: wp.vec3


@wp.kernel(module="unique")
def raygen(scene: wp.uint64, image: wp.array[wp.vec3]):
    i = wp.tid()
    payload = Payload()
    wp.optix_trace(scene, wp.vec3(0.25 + 2.0 * float(i), 0.25, 1.0), wp.vec3(0.0, 0.0, -1.0), 10.0, payload)
    image[i] = payload.color


@wp.kernel(module="unique")
def flat_hit():
    data = Flat()
    wp.optix_load_sbt_data(data)
    payload = Payload()
    payload.color = data.color
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def textured_hit():
    data = Textured()
    wp.optix_load_sbt_data(data)
    payload = Payload()
    if data.flag and data.inner.flag == wp.uint8(7):
        color = data.palette[int(data.index)] * data.scale
        payload.color = color + wp.vec3(float(data.inner.value), data.tint[0, 1], data.tint[1, 0])
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def sky_miss():
    data = Sky()
    wp.optix_load_sbt_data(data)
    payload = Payload()
    payload.color = data.color
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def callable_raygen(output: wp.array[wp.vec3]):
    wp.optix_direct_call(1)


@wp.kernel(module="unique")
def data_callable(output: wp.array[wp.vec3]):
    data = Sky()
    wp.optix_load_sbt_data(data)
    output[0] = data.color


@wp.kernel(module="unique")
def other_callable(output: wp.array[wp.vec3]):
    output[0] = wp.vec3(-1.0)


def two_triangle_gas(device):
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    inputs = []
    for dx in (0.0, 2.0):
        vertices = wp.array(triangle + [dx, 0.0, 0.0], dtype=wp.vec3, device=device)
        inputs.append(wo.triangles(vertices, wp.array([0, 1, 2], dtype=wp.int32, device=device)))
    return wo.Accel(inputs, device=device)


def textured(device):
    data = Textured()
    data.flag = True
    data.inner.flag = 7
    data.inner.value = 0.5
    data.scale = 2.0
    data.tint = wp.mat22(0.0, 0.25, 0.125, 0.0)
    data.palette = wp.array([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
    data.index = 1
    return data


def test_records_carry_their_own_data():
    device = wp.get_device("cuda")
    gas = two_triangle_gas(device)
    flat = Flat()
    flat.color = wp.vec3(0.1, 0.2, 0.3)
    sky = Sky()
    sky.color = wp.vec3(0.0, 0.0, 1.0)
    data = textured(device)
    palette = weakref.ref(data.palette)
    pipeline = wo.Pipeline(
        raygen,
        miss=[wo.Record(sky_miss, data=sky)],
        hit_groups=[wo.HitGroup(closest_hit=flat_hit, data=flat), wo.HitGroup(closest_hit=textured_hit, data=data)],
        payload_words=3,
        device=device,
    )
    del data
    gc.collect()
    assert palette() is not None
    image = wp.zeros(3, dtype=wp.vec3, device=device)
    wo.launch(pipeline, 3, [gas, image])
    np.testing.assert_allclose(image.numpy(), [[0.1, 0.2, 0.3], [2.5, 4.25, 6.125], [0.0, 0.0, 1.0]])

    # Records of different sizes share their section's 16-byte aligned stride.
    assert pipeline._sbt.hitgroupRecordStrideInBytes % 16 == 0
    assert pipeline._sbt.hitgroupRecordStrideInBytes >= 32 + wp.types.type_size_in_bytes(Textured)
    assert pipeline._sbt.missRecordStrideInBytes == 48

    # Updating data rewrites the records of the existing pipeline.
    flat.color = wp.vec3(1.0, 0.0, 0.0)
    sky.color = wp.vec3(0.0, 1.0, 0.0)
    pipeline.update_sbt_data()
    wo.launch(pipeline, 3, [gas, image])
    np.testing.assert_allclose(image.numpy(), [[1.0, 0.0, 0.0], [2.5, 4.25, 6.125], [0.0, 1.0, 0.0]])


def test_callable_records_carry_data():
    device = wp.get_device("cuda")
    sky = Sky()
    sky.color = wp.vec3(4.0, 5.0, 6.0)
    output = wp.zeros(1, dtype=wp.vec3, device=device)
    pipeline = wo.Pipeline(
        callable_raygen, callables=[other_callable, wo.Record(data_callable, data=sky)], device=device
    )
    wo.launch(pipeline, 1, [output])
    np.testing.assert_allclose(output.numpy(), [[4.0, 5.0, 6.0]])


def test_invalid_record_data():
    with pytest.raises(TypeError, match="Warp kernel"):
        wo.Record(None)
    with pytest.raises(TypeError, match="Warp struct instance"):
        wo.Pipeline(raygen, hit_groups=[wo.HitGroup(closest_hit=flat_hit, data=wp.vec3(1.0))])
    data = textured("cuda")
    data.palette = wp.zeros(2, dtype=wp.vec3, device="cpu")
    with pytest.raises(ValueError, match="same CUDA device"):
        wo.Pipeline(raygen, hit_groups=[wo.HitGroup(closest_hit=textured_hit, data=data)], payload_words=3)


def test_sbt_builder_miss_records_carry_data():
    sky = Sky()
    sky.color = wp.vec3(0.0, 0.0, 1.0)
    builder = wo.SbtBuilder(("radiance",))
    builder.set_miss("radiance", wo.Record(sky_miss, data=sky))
    builder.add_hit_group("left", radiance=wo.HitGroup())
    builder.add_hit_group("right", radiance=wo.HitGroup())
    assert "sky_miss" in builder.describe()
    image = wp.zeros(3, dtype=wp.vec3, device="cuda")
    wo.launch(wo.Pipeline(raygen, sbt=builder, payload_words=3), 3, [two_triangle_gas("cuda"), image])
    np.testing.assert_allclose(image.numpy(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
