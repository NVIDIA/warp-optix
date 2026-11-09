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
class Color:
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


@wp.kernel(module="unique")
def raygen(scene: wp.uint64, image: wp.array[wp.vec3]):
    i = wp.tid()
    if i == 3:
        wp.optix_continuation_call(0)
        wp.optix_direct_call(1)
        return
    payload = Payload()
    wp.optix_trace(scene, wp.vec3(0.25 + 2.0 * float(i), 0.25, 1.0), wp.vec3(0.0, 0.0, -1.0), 10.0, payload)
    image[i] = payload.color


@wp.kernel(module="unique")
def flat_hit():
    data = Color()
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
    data = Color()
    wp.optix_load_sbt_data(data)
    payload = Payload()
    payload.color = data.color
    wp.optix_store_payload(payload)


@wo.kernel(kind=wo.ProgramKind.CONTINUATION_CALLABLE, module="unique")
def continuation_callable(image: wp.array[wp.vec3]):
    image[4] = wp.vec3(-1.0)


@wp.kernel(module="unique")
def data_callable(image: wp.array[wp.vec3]):
    data = Color()
    wp.optix_load_sbt_data(data)
    image[3] = data.color


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
    flat, sky, ground = Color(), Color(), Color()
    flat.color = wp.vec3(0.1, 0.2, 0.3)
    sky.color = wp.vec3(0.0, 0.0, 1.0)
    ground.color = wp.vec3(4.0, 5.0, 6.0)
    data = textured(device)
    palette = weakref.ref(data.palette)
    pipeline = wo.Pipeline(
        raygen,
        sbt=wo.SbtLayout(
            miss=wo.Record(sky_miss, data=sky),
            hit_groups={
                "flat": wo.HitGroup(closest_hit=flat_hit, data=flat),
                "textured": wo.HitGroup(closest_hit=textured_hit, data=data),
            },
        ),
        # Direct and continuation callables share one list, indexed by position.
        callables=[continuation_callable, wo.Record(data_callable, data=ground)],
        payload_words=3,
        device=device,
    )
    del data
    gc.collect()
    assert palette() is not None
    image = wp.empty(5, dtype=wp.vec3, device=device)
    wo.launch(pipeline, 4, [gas, image])
    expected = [[0.1, 0.2, 0.3], [2.5, 4.25, 6.125], [0.0, 0.0, 1.0], [4.0, 5.0, 6.0], [-1.0, -1.0, -1.0]]
    np.testing.assert_allclose(image.numpy(), expected)

    # Records of different sizes share their section's 16-byte aligned stride.
    assert pipeline._sbt.hitgroupRecordStrideInBytes % 16 == 0
    assert pipeline._sbt.hitgroupRecordStrideInBytes >= 32 + wp.types.type_size_in_bytes(Textured)
    assert pipeline._sbt.missRecordStrideInBytes == 48

    # Updating data rewrites the records of the existing pipeline.
    flat.color = wp.vec3(1.0, 0.0, 0.0)
    sky.color = wp.vec3(0.0, 1.0, 0.0)
    pipeline.update_sbt_data()
    wo.launch(pipeline, 4, [gas, image])
    expected[0], expected[2] = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    np.testing.assert_allclose(image.numpy(), expected)


def test_invalid_record_data():
    with pytest.raises(TypeError, match="Warp kernel"):
        wo.Record(None)
    with pytest.raises(TypeError, match="Warp struct instance"):
        wo.Pipeline(raygen, sbt=wo.SbtLayout(hit_groups={"flat": wo.HitGroup(closest_hit=flat_hit, data=wp.vec3(1.0))}))
    data = textured("cuda")
    data.palette = wp.zeros(2, dtype=wp.vec3, device="cpu")
    with pytest.raises(ValueError, match="same CUDA device"):
        hit_groups = {"textured": wo.HitGroup(closest_hit=textured_hit, data=data)}
        wo.Pipeline(raygen, sbt=wo.SbtLayout(hit_groups=hit_groups), payload_words=3)


def test_record_array_snapshots_update_and_close():
    data = textured("cuda")
    original = weakref.ref(data.palette)
    sky = Color()
    layout = wo.SbtLayout(
        miss=wo.Record(sky_miss, data=sky), hit_groups={"textured": wo.HitGroup(closest_hit=textured_hit, data=data)}
    )
    assert "sky_miss" in layout.describe()
    pipeline = wo.Pipeline(raygen, sbt=layout, payload_words=3)
    del layout, sky

    # Mutating the host struct must not release the array still referenced by
    # the uploaded record, including when an attempted update is rejected.
    data.palette = wp.zeros(2, dtype=wp.vec3, device="cpu")
    gc.collect()
    assert original() is not None
    with pytest.raises(ValueError, match="same CUDA device"):
        pipeline.update_sbt_data()
    assert original() is not None
    data.palette = wp.ones(2, dtype=wp.vec3, device="cuda")
    replacement = weakref.ref(data.palette)
    stream = wp.Stream("cuda")
    pipeline.update_sbt_data(stream=stream)
    wp.synchronize_stream(stream)
    gc.collect()
    assert original() is None

    # close() releases both the array snapshot and the host struct/layout that
    # keeps its current fields alive, even if the Pipeline itself remains held.
    del data
    pipeline.close()
    gc.collect()
    assert replacement() is None
    assert pipeline.closed
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.update_sbt_data()
