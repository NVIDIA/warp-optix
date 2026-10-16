# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline quality scoring using the same OptiX sampler as rendering."""

import numpy as np
import warp as wp

import warp_optix as wo
from warp_optix import native

from .device import NeuralTextureView, neural_texture_sample_texel
from .runtime import upload_asset


@wp.kernel(enable_backward=False, module="unique")
def _score_texels(
    texture: NeuralTextureView,
    target: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    errors: wp.array(dtype=wp.float32),
    stride: int,
    channels: int,
):
    i = wp.tid()
    pixel = indices[i]
    value = neural_texture_sample_texel(texture, pixel % texture.width, pixel // texture.width)
    for channel in range(channels):
        difference = wp.float32(value[channel]) - target[pixel, channel]
        errors[channel * stride + i] = difference * difference


@wp.kernel(enable_backward=False, module="unique")
def _sample_indices(indices: wp.array(dtype=wp.int32), pixels: int):
    i = wp.tid()
    # One deterministic random pixel per disjoint interval; no duplicate samples.
    begin = int(wp.int64(i) * wp.int64(pixels) // wp.int64(indices.shape[0]))
    end = int(wp.int64(i + 1) * wp.int64(pixels) // wp.int64(indices.shape[0]))
    state = wp.rand_init(0, i)
    indices[i] = wp.randi(state, begin, end)


@wp.kernel(enable_backward=False, module="unique")
def _sum_errors(errors: wp.array(dtype=wp.float32), stride: int, sums: wp.array(dtype=wp.float32)):
    block, channel = wp.tid()
    values = wp.tile_load(errors, shape=(256,), offset=(channel * stride + block * 256,))
    wp.tile_atomic_add(sums, wp.tile_sum(values), offset=(channel,))


class _QualityEvaluator:
    """Reuse one source upload and scoring pipeline across compression candidates."""

    def __init__(self, image: np.ndarray, *, sampled: bool, device: str):
        self.device = device
        height, width, self.channels = image.shape
        self.count = min(height * width, 65536) if sampled else height * width
        self.stride = ((self.count + 255) // 256) * 256
        self.context = None
        try:
            self._initialize(image, height * width)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.context is not None:
            # OptiX destroys child pipelines, groups and modules too, including
            # objects created during an incomplete initialization.
            with wp.ScopedDevice(self.device):
                try:
                    wp.synchronize_device(self.device)
                finally:
                    self.context.destroy()
                    self.context = None

    def _initialize(self, image, pixels):
        device = self.device
        with wp.ScopedDevice(device):
            self.context = wo.create_context(device)
            flags = self.context.getProperty(native.DEVICE_PROPERTY_COOP_VEC)
            if not flags & int(native.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD):
                raise RuntimeError("the selected OptiX device does not support cooperative vectors")
            self.target = wp.array(image.reshape(-1, self.channels), dtype=wp.float32, device=device)
            self.indices = wp.empty(self.count, dtype=wp.int32, device=device)
            wp.launch(_sample_indices, dim=self.count, inputs=[self.indices, pixels], device=device)
            self.errors = wp.zeros(self.stride * self.channels, dtype=wp.float32, device=device)
            self.sums = wp.zeros(self.channels, dtype=wp.float32, device=device)
            self.pipeline = wo.create_pipeline(self.context, _score_texels, device=device)

    def mse(self, asset):
        with wp.ScopedDevice(self.device):
            runtime = upload_asset(asset, self.context, device=self.device)
            self.sums.zero_()
            wo.launch(
                self.pipeline,
                dim=(self.count,),
                inputs=[runtime.device_view(), self.target, self.indices, self.errors, self.stride, self.channels],
            )
            wp.launch_tiled(
                _sum_errors,
                dim=(self.stride // 256, self.channels),
                inputs=[self.errors, self.stride],
                outputs=[self.sums],
                block_dim=128,
                device=self.device,
            )
            # Synchronizes before the temporary runtime allocations go out of scope.
            return self.sums.numpy().astype(np.float64) / self.count
