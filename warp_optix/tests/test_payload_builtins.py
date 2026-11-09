# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed payload builtins against their deprecated numbered forms."""

import re

import numpy as np
import pytest
import warp as wp

import warp_optix as wo

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")]


@wp.struct
class Payload3:
    a: wp.uint32
    b: wp.uint32
    c: wp.uint32


@wp.kernel(module="unique")
def payload_raygen(scene: wp.uint64, output: wp.array2d[wp.uint32]):
    i = wp.tid()
    payload = Payload3()
    payload.a = wp.uint32(10)
    payload.b = wp.uint32(20)
    payload.c = wp.uint32(30)
    # Ray 0 hits the triangle, ray 1 misses it.
    wp.optix_trace(scene, wp.vec3(0.25 + 5.0 * float(i), 0.25, 1.0), wp.vec3(0.0, 0.0, -1.0), 10.0, payload)
    output[i, 0] = payload.a
    output[i, 1] = payload.b
    output[i, 2] = payload.c


@wp.kernel(module="unique")
def indexed_hit():
    wp.optix_set_payload(0, wp.optix_get_payload(0) + wp.uint32(1))
    wp.optix_set_payload(1, wp.optix_get_payload(2) + wp.uint32(2))
    wp.optix_set_payload(2, wp.optix_get_payload(1) * wp.uint32(3))


@wp.kernel(module="unique")
def numbered_hit():
    wp.optix_set_payload_0(wp.optix_get_payload_0() + wp.uint32(1))
    wp.optix_set_payload_1(wp.optix_get_payload_2() + wp.uint32(2))
    wp.optix_set_payload_2(wp.optix_get_payload_1() * wp.uint32(3))


@wp.kernel(module="unique")
def indexed_miss():
    wp.optix_set_payload(2, wp.optix_get_payload(0) + wp.optix_get_payload(1))


@wp.kernel(module="unique")
def numbered_miss():
    wp.optix_set_payload_2(wp.optix_get_payload_0() + wp.optix_get_payload_1())


def triangle_gas():
    vertices = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cuda")
    indices = wp.array([0, 1, 2], dtype=wp.int32, device="cuda")
    return wo.Accel([wo.triangles(vertices, indices)])


def test_indexed_payload_builtins_match_numbered_forms():
    gas = triangle_gas()
    results = []
    for hit, miss in ((indexed_hit, indexed_miss), (numbered_hit, numbered_miss)):
        sbt = wo.SbtLayout(miss=miss, hit_groups={"triangles": wo.HitGroup(closest_hit=hit)})
        pipeline = wo.Pipeline(payload_raygen, sbt=sbt, payload_words=3)
        output = wp.zeros((2, 3), dtype=wp.uint32, device="cuda")
        wo.launch(pipeline, 2, [gas, output])
        results.append(output.numpy())
    # Hit: a = 10 + 1, b = 30 + 2, then c = b * 3 = 96 (registers update in program order).
    # Miss: c = 10 + 20.
    np.testing.assert_array_equal(results[0], [[11, 32, 96], [10, 20, 30]])
    np.testing.assert_array_equal(results[0], results[1])


@wp.kernel(module="unique")
def ptx_raygen(scene: wp.uint64, output: wp.array[wp.uint32]):
    payload = Payload3()
    wp.optix_trace(scene, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, -1.0), 10.0, payload)
    output[0] = payload.a


@wp.kernel(module="unique")
def ptx_indexed_miss(output: wp.array[wp.uint32]):
    output[1] = wp.optix_get_payload(2)
    wp.optix_set_payload(1, wp.uint32(7))


@wp.kernel(module="unique")
def ptx_numbered_miss(output: wp.array[wp.uint32]):
    output[1] = wp.optix_get_payload_2()
    wp.optix_set_payload_1(wp.uint32(7))


_CALL = re.compile(r"\bcall(?:\.uni)?\s+(?:\([^)]*\)\s*,\s*)?([A-Za-z_$][\w$]*)")


def _entry(ptx: str, prefix: str) -> str:
    """Return the body of the entry point whose name starts with ``prefix``."""
    match = re.search(r"\.entry\s+(" + re.escape(prefix) + r"\w*)\s*\(", ptx)
    assert match, f"no entry {prefix} in PTX"
    start = ptx.index("{", match.end())
    depth = 0
    for position in range(start, len(ptx)):
        depth += {"{": 1, "}": -1}.get(ptx[position], 0)
        if depth == 0:
            return ptx[start : position + 1]
    raise AssertionError("unterminated entry")


def test_constant_payload_index_compiles_to_the_numbered_intrinsic(monkeypatch):
    compiled = []
    original = wp.compile_aot_module

    def capture(*args, **kwargs):
        paths = original(*args, **kwargs)
        compiled.append(paths[0].read_text())
        return paths

    monkeypatch.setattr(wp, "compile_aot_module", capture)
    for miss in (ptx_indexed_miss, ptx_numbered_miss):
        pipeline = wo.Pipeline(ptx_raygen, sbt=wo.SbtLayout(miss=miss), payload_words=3)
        pipeline.close()
    assert len(compiled) == 2
    indexed, numbered = (_entry(ptx, "__miss__") for ptx in compiled)

    indexed_calls = _CALL.findall(indexed)
    numbered_calls = _CALL.findall(numbered)
    assert "_optix_get_payload" in indexed_calls and "_optix_set_payload" in indexed_calls
    # The same intrinsics, the same number of times: the switch over the index folded away.
    assert sorted(indexed_calls) == sorted(numbered_calls)
    for body in (indexed, numbered):
        assert "brx.idx" not in body  # no jump table
    assert indexed.count("bra") == numbered.count("bra")
    assert indexed.count("setp") == numbered.count("setp")
    # In fact the instruction streams are identical, debug line markers aside.
    assert _instructions(indexed) == _instructions(numbered)


def _instructions(body: str) -> list[str]:
    lines = (line.split("//")[0].strip() for line in body.splitlines())
    return [line for line in lines if line and not line.startswith((".loc", ".file"))]
