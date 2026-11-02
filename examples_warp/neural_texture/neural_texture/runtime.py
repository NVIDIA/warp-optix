# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upload neural-texture assets for OptiX inference.

This module depends on Warp and PyOptiX, but deliberately not on warp-nn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_optix import native

from .device import NeuralTextureView
from .format import ALIGNMENT, NeuralTextureAsset


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class NeuralTextureRuntime:
    """GPU allocations and byte offsets consumed by OptiX inference kernels."""

    asset: NeuralTextureAsset
    latents: wp.array
    mip_offsets: wp.array
    mip_widths: wp.array
    mip_heights: wp.array
    matrices: wp.array
    biases: wp.array
    weight_offsets: tuple[int, ...]
    bias_offsets: tuple[int, ...]
    matrix_sizes: tuple[int, ...]
    matrix_element_type: str
    device: str

    @property
    def latent_ptr(self) -> int:
        return int(self.latents.ptr)

    @property
    def matrix_ptr(self) -> int:
        return int(self.matrices.ptr)

    @property
    def bias_ptr(self) -> int:
        return int(self.biases.ptr)

    def device_view(self) -> NeuralTextureView:
        """Return the small Warp struct passed to :func:`neural_texture_sample`."""
        view = NeuralTextureView()
        view.latents = self.latents
        view.mip_offsets = self.mip_offsets
        view.mip_widths = self.mip_widths
        view.mip_heights = self.mip_heights
        view.matrices = wp.uint64(self.matrix_ptr)
        view.biases = wp.uint64(self.bias_ptr)
        view.weight_offsets = wp.vec3ui(*self.weight_offsets)
        view.bias_offsets = wp.vec3ui(*self.bias_offsets)
        view.width = self.asset.width
        view.height = self.asset.height
        view.latent_level_count = len(self.asset.latent_mips)
        return view


def _description(n, k, element_type, layout, offset, size):
    desc = native.CoopVecMatrixDescription()
    desc.N = n
    desc.K = k
    desc.offsetInBytes = offset
    desc.elementType = element_type
    desc.layout = layout
    desc.rowColumnStrideInBytes = 0
    desc.sizeInBytes = size
    return desc


def upload_asset(
    asset: NeuralTextureAsset,
    context,
    *,
    device: str = "cuda:0",
    stream: int = 0,
    matrix_element_type: str = "fp8_e4m3",
) -> NeuralTextureRuntime:
    """Upload an asset and synchronously prepare it for OptiX FP8 inference.

    The conversion API itself is asynchronous. This convenience loader performs
    a device synchronization before returning so the runtime is immediately safe
    to launch from any stream and the temporary FP16 device copy can be released.
    """
    if matrix_element_type != "fp8_e4m3":
        raise ValueError("the v1 decoder requires matrix_element_type='fp8_e4m3'")
    output_element_type = native.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3
    flags = context.getProperty(native.DEVICE_PROPERTY_COOP_VEC)
    if not flags & int(native.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD):
        raise RuntimeError("the selected OptiX device does not support cooperative vectors")

    latent_offsets = []
    latent_words = []
    offset = 0
    for mip in asset.latent_mips:
        latent_offsets.append(offset)
        flat = np.asarray(mip, dtype=np.uint16).reshape(-1)
        latent_words.append(flat)
        offset += flat.size
    all_latents = np.concatenate(latent_words)

    source_offsets = []
    source_sizes = []
    source_total = 0
    for layer in asset.layers:
        source_total = _align(source_total, ALIGNMENT)
        source_offsets.append(source_total)
        source_sizes.append(layer.weights.nbytes)
        source_total += layer.weights.nbytes
    source_blob = np.zeros(source_total, dtype=np.uint8)
    for layer, layer_offset in zip(asset.layers, source_offsets):
        source_blob[layer_offset : layer_offset + layer.weights.nbytes] = (
            np.asarray(layer.weights).view(np.uint8).reshape(-1)
        )

    weight_offsets = []
    matrix_sizes = []
    matrix_total = 0
    for layer in asset.layers:
        matrix_total = _align(matrix_total, ALIGNMENT)
        weight_offsets.append(matrix_total)
        size = context.coopVecMatrixComputeSize(
            layer.weights.shape[0],
            layer.weights.shape[1],
            output_element_type,
            native.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        )
        matrix_sizes.append(size)
        matrix_total += size

    bias_offsets = []
    bias_total = 0
    for layer in asset.layers:
        bias_total = _align(bias_total, 16)
        bias_offsets.append(bias_total)
        bias_total += layer.bias.nbytes
    bias_blob = np.zeros(bias_total, dtype=np.uint8)
    for layer, layer_offset in zip(asset.layers, bias_offsets):
        bias_blob[layer_offset : layer_offset + layer.bias.nbytes] = np.asarray(layer.bias).view(np.uint8).reshape(-1)

    with wp.ScopedDevice(device):
        d_latents = wp.array(all_latents, dtype=wp.uint16, device=device)
        d_mip_offsets = wp.array(latent_offsets, dtype=wp.int32, device=device)
        d_mip_widths = wp.array([m.shape[1] for m in asset.latent_mips], dtype=wp.int32, device=device)
        d_mip_heights = wp.array([m.shape[0] for m in asset.latent_mips], dtype=wp.int32, device=device)
        d_source = wp.array(source_blob, dtype=wp.uint8, device=device)
        d_matrices = wp.empty(matrix_total, dtype=wp.uint8, device=device)
        d_biases = wp.array(bias_blob, dtype=wp.uint8, device=device)
        # Array uploads use Warp's current stream, which may differ from the raw
        # OptiX stream supplied below. Complete uploads before conversion so the
        # source and bias buffers are valid on every stream.
        wp.synchronize_device(device)

        input_descs = []
        output_descs = []
        for layer, src_offset, src_size, dst_offset, dst_size in zip(
            asset.layers, source_offsets, source_sizes, weight_offsets, matrix_sizes
        ):
            n, k = layer.weights.shape
            input_descs.append(
                _description(
                    n,
                    k,
                    native.COOP_VEC_ELEM_TYPE_FLOAT16,
                    native.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
                    src_offset,
                    src_size,
                )
            )
            output_descs.append(
                _description(
                    n,
                    k,
                    output_element_type,
                    native.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                    dst_offset,
                    dst_size,
                )
            )
        context.coopVecMatrixConvert(stream, input_descs, d_source.ptr, output_descs, d_matrices.ptr)

        wp.synchronize_device(device)
    return NeuralTextureRuntime(
        asset=asset,
        latents=d_latents,
        mip_offsets=d_mip_offsets,
        mip_widths=d_mip_widths,
        mip_heights=d_mip_heights,
        matrices=d_matrices,
        biases=d_biases,
        weight_offsets=tuple(weight_offsets),
        bias_offsets=tuple(bias_offsets),
        matrix_sizes=tuple(matrix_sizes),
        matrix_element_type=matrix_element_type,
        device=device,
    )
