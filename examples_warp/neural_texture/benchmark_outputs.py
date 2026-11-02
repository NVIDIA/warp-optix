# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare 4/8/16-output OptiX decoders without changing the public asset format.

Usage:
    uv run python examples_warp/neural_texture/benchmark_outputs.py image.wnt

The final matrix is truncated, preserving trained outputs exactly. Each case
gets an independent OptiX module to avoid cross-entry-point compiler outlining.
All cases in a comparison write the same channels and number of bytes. Upload,
compilation and correctness checks are outside the GPU-event timing interval.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import warp as wp
from neural_texture import load_asset, upload_asset
from neural_texture.device import (
    NeuralTextureView,
    Vec32h,
    _hgelu32,
    neural_texture_input_8,
    neural_texture_sample_texel,
)

import warp_optix as wo


def make_raygen(width: int, consumed: int, production: bool, name: str):
    vector_type = wp.types.vector(length=width, dtype=wp.float16)

    @wp.func
    def infer(texture: NeuralTextureView, inputs: Vec32h):
        h0 = Vec32h()
        h1 = Vec32h()
        result = vector_type()
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            inputs,
            texture.matrices,
            texture.weight_offsets[0],
            texture.biases,
            texture.bias_offsets[0],
            wp.uint32(0),
            h0,
        )
        h0 = _hgelu32(h0)
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            h0,
            texture.matrices,
            texture.weight_offsets[1],
            texture.biases,
            texture.bias_offsets[1],
            wp.uint32(0),
            h1,
        )
        h1 = _hgelu32(h1)
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            h1,
            texture.matrices,
            texture.weight_offsets[2],
            texture.biases,
            texture.bias_offsets[2],
            wp.uint32(0),
            result,
        )
        return result

    # A distinct name gives every variant its own OptiX module.
    @wp.kernel(enable_backward=False, module="unique", name=name)
    def raygen(texture: NeuralTextureView, output: wp.array2d(dtype=wp.float32), random_access: int):
        y, x = wp.tid()
        dest = y * texture.width + x
        if random_access != 0:
            state = wp.rand_init(123, dest)
            x = wp.randi(state, 0, texture.width)
            y = wp.randi(state, 0, texture.height)
        if wp.static(production):
            values = neural_texture_sample_texel(texture, x, y)
        else:
            inputs = neural_texture_input_8(
                texture.latents,
                texture.mip_offsets,
                texture.mip_widths,
                texture.mip_heights,
                x,
                y,
                texture.width,
                texture.height,
                0,
                texture.latent_level_count,
            )
            values = infer(texture, inputs)
        for channel in range(wp.static(consumed)):
            output[dest, channel] = wp.float32(values[channel])

    return raygen


def benchmark(args, context, asset, device):
    stream = wp.get_stream(device)
    results = {}
    for consumed in (3, 7):
        output = wp.empty((asset.width * asset.height, consumed), dtype=wp.float32, device=device)
        cases = []
        widths = (4, 8, 16) if consumed == 3 else (8, 16)
        for width, production in [*((n, False) for n in widths), (16, True)]:
            label = "production16" if production else str(width)
            raygen = make_raygen(width, consumed, production, f"outputs_{consumed}_{label}")
            # Experimental upload-only view: do not relax the public format's
            # fixed-16 validation merely to measure hypothetical alternatives.
            last = asset.layers[-1]
            layers = (
                *asset.layers[:-1],
                SimpleNamespace(weights=last.weights[:width].copy(), bias=last.bias[:width].copy()),
            )
            experimental = SimpleNamespace(**{**vars(asset), "layers": layers})
            runtime = upload_asset(experimental, context, device=device)
            pipeline = wo.create_pipeline(context, raygen, device=device)
            cases.append((label, runtime, pipeline))

        for random_access in (0, 1):

            def launch(case):
                wo.launch(
                    case[2],
                    dim=(asset.height, asset.width),
                    inputs=[case[1].device_view(), output, random_access],
                    stream=stream,
                )

            baseline = None
            max_error = 0.0
            for case in cases:
                launch(case)
                actual = output.numpy()
                if baseline is None:
                    baseline = actual
                else:
                    max_error = max(max_error, float(np.max(np.abs(actual - baseline))))
                    np.testing.assert_allclose(actual, baseline, atol=0.002, rtol=0.002)
            # Warm all cases before collecting interleaved, randomized rounds.
            for _ in range(100):
                for case in cases:
                    launch(case)
            wp.synchronize_device(device)
            samples = {case[0]: [] for case in cases}
            for round_index in range(args.rounds):
                for index in np.random.default_rng(round_index).permutation(len(cases)):
                    case = cases[index]
                    start = wp.Event(device, enable_timing=True)
                    end = wp.Event(device, enable_timing=True)
                    stream.record_event(start)
                    for _ in range(args.iterations):
                        launch(case)
                    stream.record_event(end)
                    samples[case[0]].append(wp.get_event_elapsed_time(start, end) / args.iterations)
            key = f"{consumed}ch_" + ("random" if random_access else "coherent")
            results[key] = {
                "max_absolute_error": max_error,
                "variants": {
                    case[0]: {
                        "median_ms": float(np.median(samples[case[0]])),
                        "samples_ms": samples[case[0]],
                        "matrix_bytes": case[1].matrices.size,
                        "last_matrix_bytes": case[1].matrix_sizes[-1],
                        "bias_bytes": case[1].biases.size,
                    }
                    for case in cases
                },
            }
            print(key, {label: round(float(np.median(times)), 4) for label, times in samples.items()}, flush=True)
        # Context owns API objects; release these completed cases before the next
        # consumer count instead of retaining their compilation state indefinitely.
        wp.synchronize_device(device)
        for case in cases:
            pipeline = case[2]
            pipeline.pipeline.destroy()
            for group in pipeline.groups:
                group.destroy()
            for module in pipeline.modules:
                module.destroy()
        del cases
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("iterations and rounds must be positive")
    device = "cuda:0"
    wp.init()
    with wp.ScopedDevice(device):
        context = wo.create_context(device)
        try:
            asset = load_asset(args.asset)
            results = benchmark(args, context, asset, device)
            print(
                json.dumps(
                    {
                        "gpu": wp.get_device(device).name,
                        "asset": args.asset,
                        "resolution": [asset.width, asset.height],
                        "source_channels": asset.channel_count,
                        "iterations": args.iterations,
                        "rounds": args.rounds,
                        "results": results,
                    },
                    indent=2,
                )
            )
        finally:
            wp.synchronize_device(device)
            context.destroy()


if __name__ == "__main__":
    main()
