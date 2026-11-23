# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public cluster API: real traversal, rebuilds, validation and ownership."""

import ctypes
import weakref

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")]


@wp.kernel(module="unique")
def query_grids(scene: wp.uint64, output: wp.array[wp.vec3]):
    i = wp.tid()
    q = wo.accel_query_ray(scene, wp.vec3(4.0 * float(i), 0.0, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    output[i] = wp.vec3(-1.0)
    if q.result:
        output[i] = wp.vec3(q.t, float(q.cluster_id), float(q.instance_id))


@wp.kernel(module="unique")
def query_device_handle(handle: wp.array[wp.uint64], output: wp.array[wp.vec3]):
    i = wp.tid()
    q = wo.accel_query_ray(handle[0], wp.vec3(4.0 * float(i), 0.0, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    output[i] = wp.vec3(q.t, float(q.cluster_id), 0.0) if q.result else wp.vec3(-1.0)


@wp.kernel(module="unique")
def query_loop(scene: wp.uint64, output: wp.array[wp.vec2ui]):
    # Hit-object state must be read anew after each traversal, even inside a loop.
    for i in range(2):
        q = wo.accel_query_ray(scene, wp.vec3(4.0 * float(i), 0.0, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
        output[i] = wp.vec2ui(q.cluster_id, wp.optix_hit_object_get_cluster_id())


@wp.kernel(module="unique")
def place_grid(child: wp.uint64, records: wp.array[wo.Instance]):
    records[0] = wo.make_instance(wp.transform_identity(), child, 7)


@wp.kernel(enable_backward=False)
def init_batch_records(records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(wp.transform_identity(), wp.uint64(0), 40 + i)


@wp.struct
class Payload:
    value: wp.uint32


@wp.kernel(module="unique")
def trace_materials(scene: wp.uint64, output: wp.array[wp.uint32]):
    i = wp.tid()
    payload = Payload()
    wp.optix_trace(scene, wp.vec3(4.0 * float(i), 0.0, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0, payload)
    output[i] = payload.value


@wp.kernel(module="unique")
def material_a():
    wp.optix_set_payload(0, wp.optix_get_cluster_id() + wp.uint32(100))


@wp.kernel(module="unique")
def material_b():
    wp.optix_set_payload(0, wp.optix_get_cluster_id() + wp.uint32(200))


@wp.kernel(module="unique")
def ignore_grid():
    wp.optix_ignore_intersection()


@pytest.fixture
def supported():
    if not wo.cluster_capabilities()["flags"]:
        pytest.skip("Cluster API unsupported")


def grid_points():
    return np.array([[x + 4 * c, y, 0] for c in range(2) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=np.float32)


def test_owned_clusters_rebuild_and_survive_shared_template_and_child_close(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    ids = wp.array([42, 99], dtype=wp.uint32)
    templates = wo.ClusterTemplates(2)
    accel = wo.ClusterAccel(points, grids[:1], offsets[:1], cluster_ids=ids[:1], templates=templates)
    with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
        result = wp.empty(2, dtype=wp.vec3)
        wo.launch(pipeline, dim=2, inputs=[accel, result])
        values = result.numpy()
        np.testing.assert_allclose(values[0, :2], [4, 42])
        assert values[1, 0] == -1
        # Change both the grid count and vertex positions.
        moved = grid_points()
        moved[:, 2] = 0.5
        wp.copy(points, wp.array(moved, dtype=wp.vec3))
        accel.rebuild(grid_sizes=grids, vertex_offsets=offsets, cluster_ids=ids)
        wo.launch(pipeline, dim=2, inputs=[accel, result])
        np.testing.assert_allclose(result.numpy()[:, :2], [[3.5, 42], [3.5, 99]])
        with wo.Pipeline(query_loop, clustered_geometry=True) as loop_pipeline:
            loop_result = wp.empty(2, dtype=wp.vec2ui)
            wo.launch(loop_pipeline, dim=1, inputs=[accel, loop_result])
            np.testing.assert_array_equal(loop_result.numpy(), [[42, 42], [99, 99]])
        # Closing the shared template owner must not invalidate existing builds or rebuilds.
        templates.close()
        accel.rebuild(grid_sizes=grids[:1], vertex_offsets=offsets[:1], cluster_ids=ids[:1])
        records = wp.empty(1, dtype=wo.Instance)
        wp.launch(place_grid, 1, inputs=[wp.uint64(accel.id), records])
        with wo.Accel([wo.instances(records, [accel])]) as scene:
            accel.close()
            wo.launch(pipeline, dim=2, inputs=[scene, result])
            np.testing.assert_allclose(result.numpy()[0], [3.5, 42, 7])
        with pytest.raises(RuntimeError, match="closed"):
            _ = accel.id


def test_cluster_inputs_are_validated_before_gpu_build(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    with wo.ClusterTemplates(2) as templates:
        with pytest.raises(ValueError, match="edge sizes"):
            wo.ClusterAccel(points, wp.array([[3, 2]], dtype=wp.vec2i), wp.array([0], dtype=int), templates=templates)
        with pytest.raises(ValueError, match="outside points"):
            wo.ClusterAccel(points, wp.array([[2, 2]], dtype=wp.vec2i), wp.array([15], dtype=int), templates=templates)
        with pytest.raises(ValueError, match="invalid cluster ID"):
            wo.ClusterAccel(
                points,
                wp.array([[2, 2]], dtype=wp.vec2i),
                wp.array([0], dtype=int),
                cluster_ids=wp.array([0xFFFFFFFF], dtype=wp.uint32),
                templates=templates,
            )


def test_cluster_sbt_offsets_quantization_and_explicit_stream(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    ids = wp.array([42, 99], dtype=wp.uint32)
    layout = wo.SbtLayout(
        hit_groups={"a": wo.HitGroup(closest_hit=material_a), "b": wo.HitGroup(closest_hit=material_b)}
    )
    sbt = wp.array([layout.get_sbt_offset("a"), layout.get_sbt_offset("b")], dtype=wp.uint32)
    producer, consumer = wp.Stream(), wp.Stream()
    with wo.ClusterTemplates(2, position_truncate_bits=8, max_sbt_index=1, stream=producer) as templates:
        with wo.ClusterAccel(
            points, grids, offsets, templates=templates, cluster_ids=ids, sbt_offsets=sbt, stream=consumer
        ) as accel:
            with wo.Pipeline(trace_materials, sbt=layout, payload_words=1, clustered_geometry=True) as pipeline:
                result = wp.empty(2, dtype=wp.uint32)
                wo.launch(pipeline, dim=2, inputs=[accel, result], stream=consumer)
                np.testing.assert_array_equal(result.numpy(), [142, 299])


@pytest.mark.parametrize(
    "flags, expected", [(wo.GeometryFlags.NONE, [-1.0, -1.0]), (wo.GeometryFlags.DISABLE_ANYHIT, [4.0, 4.0])]
)
def test_cluster_geometry_flags_control_anyhit(supported, flags, expected):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    with wo.ClusterAccel(points, grids, offsets, max_edge_segments=2, flags=flags) as accel:
        with wo.Pipeline(
            query_grids,
            sbt=wo.SbtLayout(hit_groups={"grid": wo.HitGroup(any_hit=ignore_grid)}),
            clustered_geometry=True,
        ) as pipeline:
            result = wp.empty(2, dtype=wp.vec3)
            wo.launch(pipeline, dim=2, inputs=[accel, result])
            np.testing.assert_allclose(result.numpy()[:, 0], expected)


def fine_points():
    # Two 8x8-quad patches; their first four vertices also form a 1x1-quad grid.
    axis = np.linspace(-1.0, 1.0, 9)
    return np.array([[x + 4 * c, y, 0] for c in range(2) for y in axis for x in axis], dtype=np.float32)


def overwrite_freed_memory(sizes):
    # Reuse freed blocks, so that traversal of released memory produces wrong results.
    return [wp.full(size, 0xFF, dtype=wp.uint8) for size in sizes for _ in range(8)]


def test_instanced_cluster_rebuilds_keep_memory_alive_after_close(supported):
    points = wp.array(fine_points(), dtype=wp.vec3)
    offsets = wp.array([0, 81], dtype=int)
    templates = wo.ClusterTemplates(8)
    accel = wo.ClusterAccel(points, wp.array([[1, 1], [1, 1]], dtype=wp.vec2i), offsets, templates=templates)
    records = wp.empty(1, dtype=wo.Instance)
    wp.launch(place_grid, 1, inputs=[wp.uint64(accel.id), records])
    scene = wo.Accel([wo.instances(records, [accel])])
    with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
        result = wp.empty(2, dtype=wp.vec3)
        # Larger grids with the same count reuse the buffers, but may change the handle.
        if accel.rebuild(grid_sizes=wp.array([[8, 8], [8, 8]], dtype=wp.vec2i)):
            wp.launch(place_grid, 1, inputs=[wp.uint64(accel.id), records])
        scene.rebuild()
        wo.launch(pipeline, dim=2, inputs=[scene, result])
        np.testing.assert_allclose(result.numpy(), [[4, 0, 7], [4, 1, 7]])
        # More clusters reallocate: rewrite the records and rebuild the instance structure.
        handle = accel.id
        assert accel.rebuild(
            grid_sizes=wp.array([[8, 8], [8, 8], [1, 1]], dtype=wp.vec2i),
            vertex_offsets=wp.array([0, 81, 0], dtype=int),
            cluster_ids=wp.array([5, 6, 7], dtype=wp.uint32),
        )
        assert accel.id != handle
        wp.launch(place_grid, 1, inputs=[wp.uint64(accel.id), records])
        scene.rebuild()
        retained = [buffer.size for buffer in accel._memory]
        accel.close()
        templates.close()
        junk = overwrite_freed_memory(retained)
        wo.launch(pipeline, dim=2, inputs=[scene, result])
        np.testing.assert_allclose(result.numpy(), [[4, 5, 7], [4, 6, 7]])
        del junk
    scene.close()


def test_cluster_capacity_reuses_buffers_and_reads_handle(supported):
    points = wp.array(fine_points(), dtype=wp.vec3)
    grids = wp.array([[8, 8], [8, 8]], dtype=wp.vec2i)
    offsets = wp.array([0, 81], dtype=int)
    with pytest.raises(ValueError, match="capacity"):
        wo.ClusterAccel(points, grids, offsets, capacity=0)
    with wo.ClusterAccel(points, grids[:1], offsets[:1], capacity=2) as accel:
        assert accel.capacity == 2
        handle = accel.id
        with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
            result = wp.empty(2, dtype=wp.vec3)
            wo.launch(pipeline, dim=2, inputs=[accel, result])
            np.testing.assert_allclose(result.numpy()[:, 0], [4, -1])

            # Growing within the reserved capacity reuses buffers and reads the returned handle.
            build = accel._build
            changed = accel.rebuild(grid_sizes=grids, vertex_offsets=offsets, validate=False)
            assert accel._build is build
            assert changed == (accel.id != handle)
            assert accel.id == int(build.gas.handles.numpy()[0])
            wo.launch(pipeline, dim=2, inputs=[accel, result])
            np.testing.assert_allclose(result.numpy()[:, :2], [[4, 0], [4, 1]])

            handle = accel.id
            assert accel.rebuild(
                grid_sizes=wp.array([[8, 8], [8, 8], [1, 1]], dtype=wp.vec2i),
                vertex_offsets=wp.array([0, 81, 0], dtype=int),
            )
            assert accel.capacity >= 3 and accel.id != handle


@pytest.mark.parametrize("validate", [False, True])
def test_cluster_rebuild_reports_moved_handle_within_capacity(supported, monkeypatch, validate):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    ids = wp.array([42, 99], dtype=wp.uint32)
    with wo.ClusterAccel(points, grids[:1], offsets[:1], capacity=2) as accel:
        build = accel._build
        original_handle = accel.id
        moved_handle = original_handle ^ 0x80
        gas_build = build.gas.build

        def move_handle(args, count=None):
            gas_build(args, count)
            # Simulate a driver returning a different handle from the device build.
            # Do not traverse this synthetic handle; the next real build restores it.
            build.gas.handles.fill_(moved_handle)

        with monkeypatch.context() as patch:
            patch.setattr(build.gas, "build", move_handle)
            assert accel.rebuild(grid_sizes=grids, vertex_offsets=offsets, cluster_ids=ids, validate=validate)
        assert accel._build is build
        assert accel.id == moved_handle
        assert accel.grid_sizes is grids and accel.vertex_offsets is offsets and accel.cluster_ids is ids

        # A subsequent build consumes the actual handle again, and the new inputs are usable.
        assert accel.rebuild(validate=validate)
        assert accel.id == int(build.gas.handles.numpy()[0])
        with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
            result = wp.empty(2, dtype=wp.vec3)
            wo.launch(pipeline, dim=2, inputs=[accel, result])
            np.testing.assert_allclose(result.numpy()[:, :2], [[4, 42], [4, 99]])


def test_async_cluster_rebuild_feeds_gpu_handle_across_capacity_growth(supported, monkeypatch):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    ids = wp.array([42, 99], dtype=wp.uint32)
    first = wp.empty(2, dtype=wp.vec3)
    second = wp.empty_like(first)
    producer, consumer = wp.Stream(), wp.Stream()
    producer.wait_stream(wp.get_stream())
    with (
        wo.ClusterAccel(points, grids[:1], offsets[:1], stream=producer) as accel,
        wo.Pipeline(query_device_handle, clustered_geometry=True) as pipeline,
    ):
        handle = accel.device_handle
        recorded = wo.launch(pipeline, dim=2, inputs=[handle, first], record_cmd=True)

        def no_readback(*args, **kwargs):
            pytest.fail("Async rebuild with trusted inputs read device memory on the host")

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "numpy", no_readback)
            patch.setattr(wp, "synchronize_stream", no_readback)
            accel.rebuild_async(cluster_ids=ids[:1], validate=False, stream=producer)
            recorded.launch(stream=producer)
            accel.rebuild_async(
                grid_sizes=grids, vertex_offsets=offsets, cluster_ids=ids, validate=False, stream=producer
            )
            assert accel.capacity >= 2 and accel.device_handle is handle
            consumer.wait_stream(producer)
            recorded.set_param_by_name("output", second)
            recorded.launch(stream=consumer)
        # Read the latest handle through the host API while a different stream is current.
        with wp.ScopedStream(consumer, sync_enter=False):
            assert accel.id == int(handle.numpy()[0])
            np.testing.assert_allclose(first.numpy()[:, :2], [[4, 42], [-1, -1]])
            np.testing.assert_allclose(second.numpy()[:, :2], [[4, 42], [4, 99]])


def test_cluster_batch_builds_and_rebuilds_instanced_gases(supported, monkeypatch):
    verts = np.array([[x + 4 * c, y, 0] for c in range(3) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=np.float32)
    points = wp.array(verts, dtype=wp.vec3)
    grids = wp.array([[2, 2]] * 3, dtype=wp.vec2i)
    offsets = wp.array([0, 9, 18], dtype=wp.int32)
    records = wp.empty(2, dtype=wo.Instance)
    result = wp.empty(3, dtype=wp.vec3)
    wp.launch(init_batch_records, records.size, inputs=[records])

    def no_readback(*args, **kwargs):
        pytest.fail("Batch fast path waited for the host")

    with wo.ClusterTemplates(2) as templates:
        with wo.ClusterAccelBatch(points, grids, offsets, (1, 2), templates=templates) as batch:
            templates.close()
            handles = batch.device_handles
            with pytest.raises(ValueError, match="one contiguous Instance record"):
                batch.write_instance_handles(records[:1])
            with monkeypatch.context() as patch:
                patch.setattr(wp.array, "numpy", no_readback)
                patch.setattr(wp, "synchronize_stream", no_readback)
                batch.write_instance_handles(records)
                scene = wo.Accel([wo.instances(records, [batch])], flags=wo.BuildFlags.ALLOW_UPDATE)
            with scene, wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
                wo.launch(pipeline, dim=3, inputs=[scene, result])
                np.testing.assert_allclose(result.numpy(), [[4, 0, 40], [4, 1, 41], [4, 2, 41]])
                verts[:, 2] = [0.5] * 9 + [-0.5] * 18
                wp.copy(points, wp.array(verts, dtype=wp.vec3))
                consumer = wp.get_stream()
                producer = wp.Stream()
                producer.wait_stream(consumer)
                with monkeypatch.context() as patch:
                    patch.setattr(wp.array, "numpy", no_readback)
                    patch.setattr(wp, "synchronize_stream", no_readback)
                    batch.rebuild_async(validate=False, stream=producer)
                    consumer.wait_stream(producer)
                    batch.write_instance_handles(records)
                    scene.rebuild()
                    wo.launch(pipeline, dim=3, inputs=[scene, result])
                assert batch.device_handles is handles
                np.testing.assert_allclose(result.numpy(), [[3.5, 0, 40], [4.5, 1, 41], [4.5, 2, 41]])
                previous = batch.ids
                assert len(set(previous)) == 2 and all(previous)
                assert batch.rebuild() == (batch.ids != previous)
                batch.write_instance_handles(records)
                monkeypatch.setattr(wo.config, "verify_instances", True)
                with wo.Accel([wo.instances(records, [batch])]) as verified:
                    batch.close()
                    wo.launch(pipeline, dim=3, inputs=[verified, result])
                    np.testing.assert_allclose(result.numpy(), [[3.5, 0, 40], [4.5, 1, 41], [4.5, 2, 41]])


def test_cluster_batch_rejects_invalid_layout(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=int)
    with pytest.raises(ValueError, match="positive count"):
        wo.ClusterAccelBatch(points, grids, offsets, (0, 2))
    with pytest.raises(ValueError, match="one entry per cluster"):
        wo.ClusterAccelBatch(points, grids, offsets, (1, 2))
    with pytest.raises(ValueError, match="outside points"):
        wo.ClusterAccelBatch(points, grids, wp.array([0, 12], dtype=int), (1, 1))


def test_cluster_growth_retains_pending_build_buffers(supported, gate):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2], [2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0, 9], dtype=wp.int32)
    with wo.ClusterAccel(points, grids[:1], offsets[:1], max_edge_segments=2) as accel:
        # Warm the kernels before holding a stream: module loading can synchronize the device.
        with wo.ClusterAccel(points, grids, offsets, max_edge_segments=2):
            wp.synchronize_device(points.device)
        handle = ctypes.c_void_p()
        gate._check(gate._cuda.cuStreamCreate(ctypes.byref(handle), 1))  # CU_STREAM_NON_BLOCKING
        stream = wp.Stream(points.device, cuda_stream=handle.value)
        frees = []

        def track_free(array, name):
            original = array.deleter

            def postpone(ptr, size):
                # Observe premature release without allowing unsafe GPU memory reuse on failure.
                frees.append((name, original, ptr, size))

            array.deleter = postpone

        track_free(accel._build.clas.scratch, "CLAS scratch")
        track_free(accel._build.clas.handles, "CLAS handles")
        track_free(accel._build.args, "CLAS descriptors")
        track_free(accel._build.count, "CLAS count")
        track_free(accel._build.gas.scratch, "GAS scratch")
        track_free(accel._build.gas_args, "GAS descriptors")
        try:
            pending = gate.block(stream)
            accel.rebuild_async(validate=False, stream=stream)
            accel.rebuild_async(grid_sizes=grids, vertex_offsets=offsets, validate=False, stream=stream)
            assert not pending.is_complete, "The gate expired before inspecting the pending rebuild"
            assert not frees, f"Released pending build buffers: {[entry[0] for entry in frees]}"
            gate.open()
            wp.synchronize_stream(stream)
            wo.get_context(points.device).collect()
            assert len(frees) == 6
            assert accel.id == int(accel.device_handle.numpy()[0])
        finally:
            gate.open()
            wp.synchronize_stream(stream)
            wo.get_context(points.device).collect()
            for _, free, ptr, size in frees:
                free(ptr, size)
            del stream
            gate._check(gate._cuda.cuStreamDestroy_v2(handle))


def test_cluster_setup_retains_arguments_on_nonblocking_stream(supported, gate, monkeypatch):
    from warp_optix._src.clusters import _batch_gas_args, _grid_args

    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0], dtype=wp.int32)
    device = points.device
    # Warm up before holding the stream, since loading Warp kernels can synchronize the device.
    with wo.ClusterTemplates(2) as templates, wo.ClusterAccelBatch(points, grids, offsets, (1,), templates=templates):
        wp.synchronize_device(device)

    original_launch = wp.launch
    arguments = []

    def track_arguments(kernel, *args, **kwargs):
        if kernel is _grid_args:
            arguments.append(weakref.ref(kwargs["inputs"][-1]))
        elif kernel is _batch_gas_args:
            arguments.append(weakref.ref(kwargs["inputs"][0]))
        return original_launch(kernel, *args, **kwargs)

    handle = ctypes.c_void_p()
    with wp.ScopedDevice(device):
        gate._check(gate._cuda.cuStreamCreate(ctypes.byref(handle), 1))  # CU_STREAM_NON_BLOCKING
        stream = wp.Stream(device, cuda_stream=handle.value)
        monkeypatch.setattr(wp, "launch", track_arguments)
        try:
            gate.block(stream)
            with wo.ClusterTemplates(2, stream=stream) as templates:
                with wo.ClusterAccelBatch(
                    points, grids, offsets, (1,), templates=templates, validate=False, stream=stream
                ) as batch:
                    try:
                        assert len(arguments) == 2 and all(arg() is not None for arg in arguments)
                    finally:
                        gate.open()
                        wp.synchronize_stream(stream)
                    assert all(batch.ids)
        finally:
            gate.open()
            wp.synchronize_stream(stream)
            wo.get_context(device).collect()
            del stream
            gate._check(gate._cuda.cuStreamDestroy_v2(handle))


@pytest.fixture
def pending_cluster_build(supported, gate):
    device = wp.get_device()
    handle = ctypes.c_void_p()
    gate._check(gate._cuda.cuStreamCreate(ctypes.byref(handle), 1))  # CU_STREAM_NON_BLOCKING
    stream = wp.Stream(device, cuda_stream=handle.value)
    frees = []

    def track_free(*arrays):
        for array in arrays:
            original = array.deleter

            def postpone(ptr, size, free=original):
                # Detect an early release without letting CUDA reuse the memory.
                frees.append((free, ptr, size))

            array.deleter = postpone

    def check(pending, expected):
        assert not pending.is_complete, "The gate expired before inspecting the pending build"
        assert not frees, "Released buffers before the queued build could run"
        gate.open()
        wp.synchronize_stream(stream)
        wo.get_context(device).collect()
        assert len(frees) == expected

    try:
        yield stream, track_free, check
    finally:
        gate.open()
        wp.synchronize_stream(stream)
        wo.get_context(device).collect()
        for free, ptr, size in frees:
            free(ptr, size)
        gate._check(gate._cuda.cuStreamDestroy_v2(handle))


@pytest.mark.parametrize("kind", ["templates", "grid", "batch", "triangle"])
@pytest.mark.parametrize("explicit_close", [True, False])
def test_cluster_close_retains_pending_build_buffers(gate, pending_cluster_build, kind, explicit_close):
    stream, track_free, check = pending_cluster_build
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0], dtype=wp.int32)
    if kind == "templates":
        with wo.ClusterTemplates(2, stream=stream):
            wp.synchronize_stream(stream)
        pending = gate.block(stream)
        owner = wo.ClusterTemplates(2, stream=stream)
        track_free(owner._state.output, owner._state.handles)
        count = 2
    else:
        if kind == "grid":
            owner = wo.ClusterAccel(points, grids, offsets, max_edge_segments=2)
            track_free(owner._host_handle, owner._build.count, owner._build.gas_args)
            builds = (owner._build.clas, owner._build.gas)
            track_free(owner._build.args, owner._templates.output, owner._templates.handles)
            count = 12
        elif kind == "batch":
            owner = wo.ClusterAccelBatch(points, grids, offsets, (1,))
            builds = (owner._clas, owner._gas)
            track_free(owner._args, owner._gas_args, owner._templates.output, owner._templates.handles)
            count = 10
        else:
            indices = wp.array([0, 1, 3], dtype=wp.int32)
            clusters = wp.array([[0, 4, 0, 1]], dtype=wp.vec4i)
            owner = wo.TriangleClusterAccel(points, indices, clusters)
            builds = (owner._clas,)
            track_free(
                owner._args,
                owner._gas_args,
                owner._gas_handle,
                owner._gas_output,
                owner._gas_scratch,
                owner._gas_destination,
            )
            count = 9
        for build in builds:
            track_free(build.output, build.scratch, build.handles)
        del builds, build
        wp.synchronize_device()
        pending = gate.block(stream)
        if kind == "triangle":
            owner.rebuild(stream)
        else:
            owner.rebuild_async(validate=False, stream=stream)
    if explicit_close:
        owner.close()
    del owner
    check(pending, count)


def test_cluster_replacing_inputs_retains_pending_build_inputs(gate, pending_cluster_build):
    stream, track_free, check = pending_cluster_build
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0], dtype=wp.int32)
    with wo.ClusterAccel(points, grids, offsets, max_edge_segments=2) as accel:
        replacement = wp.clone(points)
        track_free(points)
        del points
        wp.synchronize_device()
        pending = gate.block(stream)
        accel.rebuild_async(validate=False, stream=stream)
        accel.rebuild_async(points=replacement, validate=False, stream=stream)
        check(pending, 1)
        assert accel.id == int(accel.device_handle.numpy()[0])


def test_cluster_accel_requires_clustered_pipeline_and_unambiguous_templates(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    grids = wp.array([[2, 2]], dtype=wp.vec2i)
    offsets = wp.array([0], dtype=int)
    with wo.ClusterAccel(points, grids, offsets, max_edge_segments=2) as accel:
        with wo.Pipeline(query_grids) as pipeline:
            result = wp.empty(2, dtype=wp.vec3)
            with pytest.raises(ValueError, match="clustered_geometry"):
                wo.launch(pipeline, dim=2, inputs=[accel, result])
            with pytest.raises(ValueError, match="clustered_geometry"):
                wo.launch(pipeline, dim=2, inputs=[accel, result], record_cmd=True)
    with wo.ClusterTemplates(2) as templates:
        with pytest.raises(ValueError, match="ClusterTemplates"):
            wo.ClusterAccel(points, grids, offsets, templates=templates, max_edge_segments=2)


def test_recorded_launch_keeps_replaced_cluster_memory_alive(supported):
    points = wp.array(fine_points(), dtype=wp.vec3)
    accel = wo.ClusterAccel(
        points,
        wp.array([[8, 8], [8, 8]], dtype=wp.vec2i),
        wp.array([0, 81], dtype=int),
        cluster_ids=wp.array([42, 99], dtype=wp.uint32),
    )
    with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
        result = wp.empty(2, dtype=wp.vec3)
        recorded = wo.launch(pipeline, dim=2, inputs=[accel, result], record_cmd=True)
        retained = [buffer.size for buffer in accel._memory]
        # Growing reallocates; the recorded launch still holds the previous handle.
        accel.rebuild(
            grid_sizes=wp.array([[8, 8], [8, 8], [8, 8]], dtype=wp.vec2i),
            vertex_offsets=wp.array([0, 81, 0], dtype=int),
            cluster_ids=wp.array([1, 2, 3], dtype=wp.uint32),
        )
        accel.close()
        junk = overwrite_freed_memory(retained)
        recorded.launch()
        np.testing.assert_allclose(result.numpy()[:, :2], [[4, 42], [4, 99]])
        del junk


def test_cluster_capabilities_have_a_stable_shape():
    keys = {"flags", "max_vertices", "max_triangles", "max_grid_resolution", "max_sbt_index", "max_clusters_per_gas"}
    assert set(wo.cluster_capabilities()) == keys


def test_instanced_clusters_require_clustered_pipeline(supported):
    points = wp.array(grid_points(), dtype=wp.vec3)
    with wo.ClusterAccel(points, wp.array([[2, 2]], dtype=wp.vec2i), wp.array([0], dtype=int)) as accel:
        records = wp.empty(1, dtype=wo.Instance)
        wp.launch(place_grid, 1, inputs=[wp.uint64(accel.id), records])
        with wo.Accel([wo.instances(records, [accel])]) as scene:
            outer = wp.empty(1, dtype=wo.Instance)
            wp.launch(place_grid, 1, inputs=[wp.uint64(scene.id), outer])
            with wo.Accel([wo.instances(outer, [scene])]) as nested:
                result = wp.empty(2, dtype=wp.vec3)
                with wo.Pipeline(query_grids) as pipeline:
                    for structure in (scene, nested):
                        with pytest.raises(ValueError, match="clustered_geometry"):
                            wo.launch(pipeline, dim=2, inputs=[structure, result])
                with wo.Pipeline(query_grids, clustered_geometry=True) as pipeline:
                    wo.launch(pipeline, dim=2, inputs=[scene, result])
                    np.testing.assert_allclose(result.numpy()[0], [4, 0, 7])


def test_cluster_geometry_flags_map_by_name(supported):
    from warp_optix._src.clusters import _primitive_flags

    from warp_optix import native as n

    assert _primitive_flags(wo.GeometryFlags.DISABLE_ANYHIT) == int(n.CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT)
    assert _primitive_flags(wo.GeometryFlags.REQUIRE_SINGLE_ANYHIT_CALL) == int(
        n.CLUSTER_ACCEL_PRIMITIVE_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
    )
    assert _primitive_flags(n.GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING) == int(
        n.CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_TRIANGLE_FACE_CULLING
    )
    with pytest.raises(ValueError, match="geometry flags"):
        wo.ClusterTemplates(2, flags=8)


# ---------------------------------------------------------------------------
# TriangleClusterAccel


@wp.kernel(module="unique")
def query_triangle_clusters(scene: wp.uint64, output: wp.array[wp.vec3]):
    i = wp.tid()
    # Each ray hits the second triangle of a quad.
    q = wo.accel_query_ray(scene, wp.vec3(4.0 * float(i) - 0.25, 0.25, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
    output[i] = wp.vec3(q.t, float(q.cluster_id), float(q.primitive_index)) if q.result else wp.vec3(-1.0)


@wp.kernel(module="unique")
def place_triangle_clusters(child: wp.uint64, records: wp.array[wo.Instance]):
    records[0] = wo.make_instance(wp.transform_identity(), child, 0)


@pytest.mark.parametrize("build_flags", [wo.BuildFlags.PREFER_FAST_BUILD, wo.BuildFlags.PREFER_FAST_TRACE])
def test_triangle_clusters_rebuild_under_a_refit_instance_structure(supported, build_flags):
    # Two unit quads 4 m apart, one cluster each, with cluster-local indices.
    quad = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
    points = wp.array(np.concatenate([quad, quad + [4.0, 0.0, 0.0]]), dtype=wp.vec3, device="cuda:0")
    indices = wp.array(np.tile([0, 1, 2, 0, 2, 3], 2), dtype=wp.int32, device="cuda:0")
    clusters = wp.array([[0, 4, 0, 2], [4, 4, 2, 2]], dtype=wp.vec4i, device="cuda:0")
    ids = wp.array([10, 20], dtype=wp.uint32, device="cuda:0")
    accel = wo.TriangleClusterAccel(points, indices, clusters, cluster_ids=ids, build_flags=build_flags)
    records = wp.empty(1, dtype=wo.Instance, device="cuda:0")
    wp.launch(place_triangle_clusters, 1, inputs=[accel.id, records])
    scene = wo.Accel([wo.instances(records, [accel])], flags=wo.BuildFlags.ALLOW_UPDATE)
    pipeline = wo.Pipeline(query_triangle_clusters, clustered_geometry=True)
    output = wp.empty(2, dtype=wp.vec3, device="cuda:0")
    # Lift past the built bounds: a rebuild keeps the handle, so refitting the IAS suffices.
    points.assign(points.numpy() + [0.0, 0.0, 2.5])
    accel.rebuild()
    scene.refit()
    wo.launch(pipeline, dim=2, inputs=[scene, output])
    np.testing.assert_allclose(output.numpy(), [[1.5, 10.0, 1.0], [1.5, 20.0, 1.0]])
    assert int(accel._gas_handle.numpy()[0]) == accel.id


def test_recorded_launch_retains_triangle_cluster_buffers_after_close(supported):
    points = wp.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=wp.vec3, device="cuda:0")
    indices = wp.array([0, 1, 2], dtype=wp.int32, device="cuda:0")
    clusters = wp.array([[0, 3, 0, 1]], dtype=wp.vec4i, device="cuda:0")
    with wo.TriangleClusterAccel(points, indices, clusters) as accel:
        with wo.Pipeline(query_grids, clustered_geometry=True, device=points.device) as pipeline:
            result = wp.empty(1, dtype=wp.vec3, device=points.device)
            recorded = wo.launch(pipeline, 1, inputs=[accel, result], record_cmd=True)
            buffers = [weakref.ref(buffer) for buffer in (accel._clas.output, accel._gas_output)]
            accel.close()
            assert all(buffer() is not None for buffer in buffers)
            recorded.launch()
            np.testing.assert_allclose(result.numpy()[0, :2], [4, 0])


@pytest.mark.parametrize("name", ["points", "indices"])
@pytest.mark.parametrize("validate", [True, False])
def test_triangle_clusters_reject_strided_buffers(name, validate):
    arrays = {
        "points": wp.zeros(6, dtype=wp.vec3, device="cuda:0"),
        "indices": wp.zeros(6, dtype=wp.int32, device="cuda:0"),
        "clusters": wp.array([[0, 3, 0, 1]], dtype=wp.vec4i, device="cuda:0"),
    }
    arrays[name] = arrays[name][::2]
    with pytest.raises(ValueError, match=f"{name} must be contiguous"):
        wo.TriangleClusterAccel(**arrays, validate=validate)


@pytest.mark.parametrize(
    ("first_triangle", "indices", "cluster_id", "message"),
    [
        (-1, [0, 1, 2], 0, "triangles lie outside"),
        (1, [0, 1, 2], 0, "triangles lie outside"),
        (0, [0, 1, -1], 0, "Cluster-local indices"),
        (0, [0, 1, 3], 0, "Cluster-local indices"),
        (0, [0, 1, 2], 0xFFFFFFFF, "reserved"),
    ],
)
def test_triangle_clusters_reject_invalid_topology(supported, first_triangle, indices, cluster_id, message):
    points = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
    triangles = wp.array(indices, dtype=wp.int32, device="cuda:0")
    clusters = wp.array([[0, 3, first_triangle, 1]], dtype=wp.vec4i, device="cuda:0")
    ids = wp.array([cluster_id], dtype=wp.uint32, device="cuda:0")
    with pytest.raises(ValueError, match=message):
        wo.TriangleClusterAccel(points, triangles, clusters, cluster_ids=ids)
