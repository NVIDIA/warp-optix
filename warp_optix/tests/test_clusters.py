# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public cluster API: real traversal, rebuilds, validation and ownership."""

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
def query_loop(scene: wp.uint64, output: wp.array[wp.vec2ui]):
    # Hit-object state must be read anew after each traversal, even inside a loop.
    for i in range(2):
        q = wo.accel_query_ray(scene, wp.vec3(4.0 * float(i), 0.0, 4.0), wp.vec3(0.0, 0.0, -1.0), 100.0)
        output[i] = wp.vec2ui(q.cluster_id, wp.optix_hit_object_get_cluster_id())


@wp.kernel(module="unique")
def place_grid(child: wp.uint64, records: wp.array[wo.Instance]):
    records[0] = wo.make_instance(wp.transform_identity(), child, 7)


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
    with wo.ClusterAccel(points, grids, offsets, max_edge_segments=2, geometry_flags=flags) as accel:
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
        # Larger grids with the same count rebuild in place and keep the handle.
        handle = accel.id
        accel.rebuild(grid_sizes=wp.array([[8, 8], [8, 8]], dtype=wp.vec2i))
        assert accel.id == handle
        scene.rebuild()
        wo.launch(pipeline, dim=2, inputs=[scene, result])
        np.testing.assert_allclose(result.numpy(), [[4, 0, 7], [4, 1, 7]])
        # More clusters reallocate: rewrite the records and rebuild the instance structure.
        accel.rebuild(
            grid_sizes=wp.array([[8, 8], [8, 8], [1, 1]], dtype=wp.vec2i),
            vertex_offsets=wp.array([0, 81, 0], dtype=int),
            cluster_ids=wp.array([5, 6, 7], dtype=wp.uint32),
        )
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
    with pytest.raises(ValueError, match="geometry_flags"):
        wo.ClusterTemplates(2, geometry_flags=8)
