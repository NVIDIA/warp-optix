# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline neural-texture compression implemented with warp-nn.

Nothing in the inference/runtime path imports this module. Training follows the
NTC design: two learned latent grids, positional features, and a tiny per-asset
MLP. A final refinement phase trains through 4-bit latent quantization so the
decoder and latents adapt to the representation used at inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import warp as wp

try:
    from warp_nn import nn, optimizers
except ImportError as error:  # pragma: no cover - exercised without training extra
    raise ImportError("neural-texture compression requires the optional 'warp-nn' package") from error

from .format import (
    CompressionResult,
    NetworkLayer,
    NeuralTextureAsset,
    TextureChannel,
    pack_latents,
)
from .reference import _decode_pixels, _quantize_fp8_e4m3


@wp.func
def _quantize_latent(value: float):
    return wp.rint((wp.clamp(value, -1.0, 1.0) + 1.0) * 7.5) * (2.0 / 15.0) - 1.0


@wp.func_grad(_quantize_latent)
def _adj_quantize_latent(value: float, adj_ret: float):
    wp.adjoint[value] += adj_ret


@wp.kernel
def _build_inputs(
    latent0: wp.array3d(dtype=wp.float32),
    latent1: wp.array3d(dtype=wp.float32),
    pixel_indices: wp.array(dtype=wp.int32),
    image_width: int,
    image_height: int,
    quantize: bool,
    output: wp.array2d(dtype=wp.float32),
):
    sample = wp.tid()
    pixel = pixel_indices[sample]
    x = pixel % image_width
    y = pixel // image_width
    u = (wp.float32(x) + 0.5) / wp.float32(image_width)
    v = (wp.float32(y) + 0.5) / wp.float32(image_height)
    for feature in range(8):
        for grid_index in range(2):
            width = latent0.shape[1] if grid_index == 0 else latent1.shape[1]
            height = latent0.shape[0] if grid_index == 0 else latent1.shape[0]
            fx = u * wp.float32(width) - 0.5
            fy = v * wp.float32(height) - 0.5
            x0 = int(wp.floor(fx))
            y0 = int(wp.floor(fy))
            tx = fx - wp.float32(x0)
            ty = fy - wp.float32(y0)
            ix0 = ((x0 % width) + width) % width
            ix1 = (((x0 + 1) % width) + width) % width
            iy0 = ((y0 % height) + height) % height
            iy1 = (((y0 + 1) % height) + height) % height
            if grid_index == 0:
                s00 = latent0[iy0, ix0, feature]
                s10 = latent0[iy0, ix1, feature]
                s01 = latent0[iy1, ix0, feature]
                s11 = latent0[iy1, ix1, feature]
            else:
                s00 = latent1[iy0, ix0, feature]
                s10 = latent1[iy0, ix1, feature]
                s01 = latent1[iy1, ix0, feature]
                s11 = latent1[iy1, ix1, feature]
            if quantize:
                s00 = _quantize_latent(s00)
                s10 = _quantize_latent(s10)
                s01 = _quantize_latent(s01)
                s11 = _quantize_latent(s11)
            output[sample, grid_index * 8 + feature] = (
                s00 * (1.0 - tx) * (1.0 - ty) + s10 * tx * (1.0 - ty) + s01 * (1.0 - tx) * ty + s11 * tx * ty
            )
    px = wp.float32(x) / wp.float32(image_width)
    py = wp.float32(y) / wp.float32(image_height)
    for wave in range(3):
        index = 16 + wave * 4
        output[sample, index] = (px - wp.floor(px)) * 2.0 - 1.0
        output[sample, index + 1] = (py - wp.floor(py)) * 2.0 - 1.0
        output[sample, index + 2] = (px + 0.25 - wp.floor(px + 0.25)) * 2.0 - 1.0
        output[sample, index + 3] = (py + 0.25 - wp.floor(py + 0.25)) * 2.0 - 1.0
        px *= 2.0
        py *= 2.0
    output[sample, 28] = 0.0
    output[sample, 29] = 0.0
    output[sample, 30] = 0.0
    output[sample, 31] = 0.0


@wp.kernel
def _hgelu(input: wp.array2d(dtype=wp.float32), output: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    value = input[i, j]
    output[i, j] = wp.min(value, 3.0) * wp.clamp(value / 3.0 + 0.5, 0.0, 1.0)


@wp.kernel
def _mse(
    prediction: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float16),
    pixel_indices: wp.array(dtype=wp.int32),
    channel_weights: wp.array(dtype=wp.float32),
    normalization: float,
    errors: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    difference = prediction[i, j] - wp.float32(target[pixel_indices[i], j])
    errors[i, j] = difference * difference * channel_weights[j] * normalization


@wp.kernel
def _sum_training_loss(errors: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    block = wp.tid()
    values = wp.tile_load(errors, shape=(256,), offset=(block * 256,))
    wp.tile_atomic_add(loss, wp.tile_sum(values))


@wp.func
def _fake_quantize_fp8_e4m3(value: float):
    magnitude = wp.min(wp.abs(value), 448.0)
    step = 0.001953125
    if magnitude >= 0.015625:
        exponent = wp.floor(wp.log2(magnitude))
        step = wp.pow(2.0, exponent - 3.0)
    scaled = magnitude / step
    lower = wp.floor(scaled)
    fraction = scaled - lower
    rounded = lower
    if fraction > 0.5 or (fraction == 0.5 and int(lower) % 2 == 1):
        rounded += 1.0
    result = wp.min(rounded * step, 448.0)
    if value < 0.0:
        result = -result
    return result


@wp.func_grad(_fake_quantize_fp8_e4m3)
def _adj_fake_quantize_fp8_e4m3(value: float, adj_ret: float):
    wp.adjoint[value] += adj_ret


@wp.kernel
def _fake_quantize_fp8_e4m3_kernel(
    input: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    output[i, j] = _fake_quantize_fp8_e4m3(input[i, j])


@wp.kernel(enable_backward=False)
def _sample_pixels(
    pixel_indices: wp.array(dtype=wp.int32),
    step: wp.array(dtype=wp.int32),
    seed: int,
    pixel_count: int,
):
    sample = wp.tid()
    state = wp.rand_init(seed, sample + step[0] * pixel_indices.shape[0])
    pixel_indices[sample] = wp.randi(state, 0, pixel_count)


@wp.kernel(enable_backward=False)
def _sequential_pixels(pixel_indices: wp.array(dtype=wp.int32)):
    pixel_indices[wp.tid()] = wp.tid()


@wp.kernel(enable_backward=False)
def _project_fp8_in_place(values: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    values[i, j] = _fake_quantize_fp8_e4m3(values[i, j])


@wp.kernel(enable_backward=False)
def _record_loss(
    loss: wp.array(dtype=wp.float32),
    history: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.int32),
):
    history[step[0]] = loss[0]
    step[0] += 1


class _FP8Quantizer:
    def __init__(self):
        self._cache = {}

    def __call__(self, input):
        key = (tuple(input.shape), input.dtype)
        if key not in self._cache:
            self._cache[key] = wp.empty(input.shape, dtype=wp.float32, device=input.device, requires_grad=True)
        output = self._cache[key]
        wp.launch(
            _fake_quantize_fp8_e4m3_kernel,
            dim=input.shape,
            inputs=[input],
            outputs=[output],
            device=input.device,
        )
        return output


class _HGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def __call__(self, input):
        key = (tuple(input.shape), input.dtype)
        if key not in self._cache:
            self._cache[key] = wp.empty(input.shape, dtype=wp.float32, device=input.device, requires_grad=True)
        output = self._cache[key]
        wp.launch(
            _hgelu,
            dim=input.shape,
            inputs=[input],
            outputs=[output],
            device=input.device,
        )
        return output


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(32, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.activation0 = _HGELU()
        self.activation1 = _HGELU()
        self.quantizer0 = _FP8Quantizer()
        self.quantizer1 = _FP8Quantizer()
        self.quantizer2 = _FP8Quantizer()
        self.fp8_inputs = False
        super().__post_init__()

    def __call__(self, inputs):
        if self.fp8_inputs:
            inputs = self.quantizer0(inputs)
        hidden0 = self.activation0(self.fc0(inputs))
        if self.fp8_inputs:
            hidden0 = self.quantizer1(hidden0)
        hidden1 = self.activation1(self.fc1(hidden0))
        if self.fp8_inputs:
            hidden1 = self.quantizer2(hidden1)
        return self.fc2(hidden1)


def _seeded_decoder(seed: int) -> _Decoder:
    # warp-nn's initializers currently use NumPy's legacy global RNG. Preserve
    # its state so this public seed is deterministic without surprising callers.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return _Decoder()
    finally:
        np.random.set_state(state)


def _project_model_weights(model: _Decoder) -> None:
    for linear in (model.fc0, model.fc1, model.fc2):
        linear.weight.data.assign(_quantize_fp8_e4m3(linear.weight.data.numpy()))


def _compress_channels(
    image: np.ndarray,
    texture_channels: tuple[TextureChannel, ...],
    channel_weights: np.ndarray,
    *,
    steps: int = 1000,
    refinement_steps: int | None = None,
    learning_rate: float = 1e-2,
    latent_scale: int | None = None,
    batch_size: int = 65536,
    device: str = "cuda:0",
    seed: int = 42,
    _evaluation_indices: np.ndarray | None = None,
    _evaluator=None,
) -> CompressionResult:
    if steps < 0 or (refinement_steps is not None and refinement_steps < 0):
        raise ValueError("training step counts must be non-negative")
    if latent_scale is not None and latent_scale <= 0:
        raise ValueError("latent_scale must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 0x7FFFFFFF:
        raise ValueError("seed must be an integer in [0, 2**31 - 1]")
    if refinement_steps is None:
        refinement_steps = max(steps // 4, 1) if steps else 0

    height, width, channels = image.shape
    if _evaluator is None and wp.get_device(device).is_cuda:
        from .evaluation import _QualityEvaluator

        with _QualityEvaluator(image, sampled=latent_scale is None, device=device) as evaluator:
            return _compress_channels(
                image,
                texture_channels,
                channel_weights,
                steps=steps,
                refinement_steps=refinement_steps,
                learning_rate=learning_rate,
                latent_scale=latent_scale,
                batch_size=batch_size,
                device=device,
                seed=seed,
                _evaluation_indices=_evaluation_indices,
                _evaluator=evaluator,
            )
    if latent_scale is None:
        # A bounded search: choose the smaller asset only if every named map
        # remains within 0.5 dB of scale 4. Both see the same evaluation pixels.
        count = height * width
        indices = (
            np.random.default_rng(0).choice(count, 65536, replace=False)
            if _evaluator is None and count > 65536
            else None
        )
        candidates = []
        for scale in (4, 5):
            candidates.append(
                _compress_channels(
                    image,
                    texture_channels,
                    channel_weights,
                    steps=steps,
                    refinement_steps=refinement_steps,
                    learning_rate=learning_rate,
                    latent_scale=scale,
                    batch_size=batch_size,
                    device=device,
                    seed=seed,
                    _evaluation_indices=indices,
                    _evaluator=_evaluator,
                )
            )
        reference, compact = candidates
        acceptable = all(
            compact.psnr_by_texture[name] >= value - 0.5 for name, value in reference.psnr_by_texture.items()
        )
        if acceptable and compact.asset.storage_bytes < reference.asset.storage_bytes:
            selected = compact
            reason = "Chose the smaller asset: every texture stayed within 0.5 dB of the larger candidate."
        else:
            selected = reference
            reason = "Kept the larger asset to preserve texture detail or because the smaller grid saved no space."
        return replace(selected, selection_reason=reason)

    rng = np.random.default_rng(seed)
    latent0_shape = (
        max((height + latent_scale - 1) // latent_scale, 1),
        max((width + latent_scale - 1) // latent_scale, 1),
        8,
    )
    latent1_shape = (
        max((latent0_shape[0] + 1) // 2, 1),
        max((latent0_shape[1] + 1) // 2, 1),
        8,
    )
    target_host = image.reshape(-1, channels)

    batch_count = min(batch_size, height * width)
    with wp.ScopedDevice(device):
        latent0 = wp.array(
            rng.uniform(-0.1, 0.1, latent0_shape).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        latent1 = wp.array(
            rng.uniform(-0.1, 0.1, latent1_shape).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        target = wp.array(target_host, dtype=wp.float16, device=device)
        weights = wp.array(channel_weights, dtype=wp.float32, device=device)
        pixel_indices = wp.empty(batch_count, dtype=wp.int32, device=device)
        inputs = wp.empty((batch_count, 32), dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        loss_terms = wp.empty((batch_count, channels), dtype=wp.float32, device=device, requires_grad=True)
        flat_loss_terms = loss_terms.flatten()
        model = _seeded_decoder(int(seed))
        model.to(device)
        optimizer = optimizers.Adam(
            [latent0, latent1, *model.parameters()],
            lr=learning_rate,
            device=device,
            disable_graph=True,
        )
        total_steps = steps + refinement_steps
        loss_history = wp.empty(max(total_steps, 1), dtype=wp.float32, device=device)
        step = wp.zeros(1, dtype=wp.int32, device=device)
        full_batch = batch_count == height * width
        if total_steps and full_batch:
            wp.launch(
                _sequential_pixels,
                dim=batch_count,
                inputs=[pixel_indices],
                device=device,
            )

        def train_step(active_latent0, active_latent1, active_optimizer, project_weights):
            loss.zero_()
            if not full_batch:
                wp.launch(
                    _sample_pixels,
                    dim=batch_count,
                    inputs=[pixel_indices, step, seed, height * width],
                    device=device,
                )
            if project_weights:
                for linear in (model.fc0, model.fc1, model.fc2):
                    wp.launch(
                        _project_fp8_in_place,
                        dim=linear.weight.data.shape,
                        inputs=[linear.weight.data],
                        device=device,
                    )

            with wp.Tape() as tape:
                wp.launch(
                    _build_inputs,
                    dim=batch_count,
                    inputs=[
                        active_latent0,
                        active_latent1,
                        pixel_indices,
                        width,
                        height,
                        project_weights,
                    ],
                    outputs=[inputs],
                    device=device,
                )
                prediction = model(inputs)
                wp.launch(
                    _mse,
                    dim=(batch_count, channels),
                    inputs=[
                        prediction,
                        target,
                        pixel_indices,
                        weights,
                        loss_normalization,
                    ],
                    outputs=[loss_terms],
                    device=device,
                )
                wp.launch_tiled(
                    _sum_training_loss,
                    dim=(flat_loss_terms.size + 255) // 256,
                    inputs=[flat_loss_terms],
                    outputs=[loss],
                    block_dim=128,
                    device=device,
                )
            tape.backward(loss)
            active_optimizer.step()
            wp.launch(
                _record_loss,
                dim=1,
                inputs=[loss, loss_history, step],
                device=device,
            )
            tape.zero()

        loss_normalization = 1.0 / (batch_count * float(np.sum(channel_weights)))

        def train_phase(count, active_latent0, active_latent1, active_optimizer, project_weights):
            if count == 0:
                return
            # One real step allocates lazy layer/Tape buffers and compiles every
            # kernel before capture. The remaining identical steps replay one graph.
            train_step(active_latent0, active_latent1, active_optimizer, project_weights)
            if count == 1:
                return
            if wp.get_device(device).is_cuda:
                wp.capture_begin(device=device)
                try:
                    train_step(
                        active_latent0,
                        active_latent1,
                        active_optimizer,
                        project_weights,
                    )
                except Exception:
                    wp.capture_end(device=device)
                    raise
                graph = wp.capture_end(device=device)
                for _ in range(count - 2):
                    wp.capture_launch(graph)
            else:
                for _ in range(count - 1):
                    train_step(
                        active_latent0,
                        active_latent1,
                        active_optimizer,
                        project_weights,
                    )

        train_phase(steps, latent0, latent1, optimizer, False)

        refine_optimizer = optimizers.Adam(
            [latent0, latent1, *model.parameters()],
            lr=learning_rate * 0.25,
            device=device,
            disable_graph=True,
        )
        model.fp8_inputs = True
        train_phase(refinement_steps, latent0, latent1, refine_optimizer, True)
        packed0 = pack_latents(latent0.numpy())
        packed1 = pack_latents(latent1.numpy())
        losses = tuple(float(value) for value in loss_history.numpy()[:total_steps])

        _project_model_weights(model)
        layers = []
        for index, linear in enumerate((model.fc0, model.fc1, model.fc2)):
            layers.append(
                NetworkLayer(
                    linear.weight.data.numpy().astype(np.float16),
                    linear.bias.data.numpy().reshape(-1).astype(np.float16),
                    "none" if index == 2 else "hgelu",
                )
            )
        asset = NeuralTextureAsset(
            width=width,
            height=height,
            channels=texture_channels,
            latent_features=8,
            latent_mips=(packed0, packed1),
            layers=tuple(layers),
            metadata={
                "compressor": "warp-nn",
                "training_steps": steps,
                "refinement_steps": refinement_steps,
                "latent_scale": latent_scale,
                "batch_size": batch_count,
                "weight_type": "fp8_e4m3",
            },
        )
    if _evaluator is not None:
        channel_mse = _evaluator.mse(asset)
        evaluation_pixels = _evaluator.count
    else:
        decoded = _decode_pixels(asset, batch_size=batch_size, pixel_indices=_evaluation_indices)
        target_values = image.reshape(-1, channels)
        if _evaluation_indices is not None:
            target_values = target_values[_evaluation_indices]
        channel_mse = np.mean((decoded - target_values) ** 2, axis=0, dtype=np.float64)
        evaluation_pixels = len(decoded)
    mse = float(np.mean(channel_mse))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))
    psnr_by_texture = {}
    for texture in texture_channels:
        texture_mse = float(np.mean(channel_mse[texture.first_channel : texture.first_channel + texture.channel_count]))
        psnr_by_texture[texture.name] = float("inf") if texture_mse == 0.0 else float(-10.0 * np.log10(texture_mse))
    return CompressionResult(
        asset=asset,
        losses=tuple(losses),
        psnr=psnr,
        psnr_by_texture=psnr_by_texture,
        evaluation_pixels=evaluation_pixels,
    )


def _normalized_image(image: np.ndarray, name: str, *, max_channels: int) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or not 1 <= image.shape[2] <= max_channels:
        raise ValueError(f"texture '{name}' must have shape (height, width, 1..{max_channels})")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("texture dimensions must be positive")
    if not np.all(np.isfinite(image)) or np.min(image) < 0.0 or np.max(image) > 1.0:
        raise ValueError(f"texture '{name}' values must be finite and normalized to [0, 1]")
    return image


def compress_texture_set(
    textures: Mapping[str, np.ndarray],
    *,
    color_spaces: Mapping[str, str] | None = None,
    channel_weights: Mapping[str, float | Sequence[float]] | None = None,
    steps: int = 1000,
    refinement_steps: int | None = None,
    learning_rate: float = 1e-2,
    latent_scale: int | None = None,
    batch_size: int = 65536,
    device: str = "cuda:0",
    seed: int = 42,
) -> CompressionResult:
    """Compress a named, same-resolution texture set into one neural asset.

    Values are HWC (or HW for scalar maps), normalized to ``[0, 1]``, and may
    have one to four channels. Up to 16 channels are decoded together, allowing
    correlated material maps such as albedo, normal, and roughness to share the
    same latents and network. By default the compressor selects a size using
    sampled quality checks. Set ``latent_scale`` explicitly to train one size
    and compute full-image PSNR instead.
    """
    if not isinstance(textures, Mapping) or not textures:
        raise ValueError("textures must be a non-empty mapping")
    color_spaces = {} if color_spaces is None else color_spaces
    channel_weights = {} if channel_weights is None else channel_weights
    if not isinstance(color_spaces, Mapping):
        raise TypeError("color_spaces must be a mapping")
    if not isinstance(channel_weights, Mapping):
        raise TypeError("channel_weights must be a mapping")
    unknown = (set(color_spaces) | set(channel_weights)) - set(textures)
    if unknown:
        raise ValueError(f"options provided for unknown textures: {sorted(unknown)}")

    images = []
    descriptors = []
    weights = []
    shape = None
    first_channel = 0
    for name, value in textures.items():
        if not isinstance(name, str) or not name:
            raise ValueError("texture names must be non-empty strings")
        image = _normalized_image(value, name, max_channels=4)
        if shape is None:
            shape = image.shape[:2]
        elif image.shape[:2] != shape:
            raise ValueError("all textures must have the same height and width")
        if first_channel + image.shape[2] > 16:
            raise ValueError("a neural texture set supports at most 16 channels")

        color_space = color_spaces.get(name, "linear")
        descriptors.append(TextureChannel(name, first_channel, image.shape[2], color_space))
        texture_weight = np.asarray(channel_weights.get(name, 1.0), dtype=np.float32)
        if texture_weight.ndim == 0:
            texture_weight = np.full(image.shape[2], texture_weight, dtype=np.float32)
        if texture_weight.shape != (image.shape[2],):
            raise ValueError(f"channel_weights['{name}'] must be scalar or have one value per channel")
        if not np.all(np.isfinite(texture_weight)) or np.any(texture_weight <= 0.0):
            raise ValueError("channel weights must be finite and positive")
        images.append(image)
        weights.append(texture_weight)
        first_channel += image.shape[2]

    return _compress_channels(
        np.concatenate(images, axis=2),
        tuple(descriptors),
        np.concatenate(weights),
        steps=steps,
        refinement_steps=refinement_steps,
        learning_rate=learning_rate,
        latent_scale=latent_scale,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


def compress_texture(
    image: np.ndarray,
    *,
    steps: int = 1000,
    refinement_steps: int | None = None,
    learning_rate: float = 1e-2,
    latent_scale: int | None = None,
    batch_size: int = 65536,
    device: str = "cuda:0",
    seed: int = 42,
    channel_name: str = "texture",
    color_space: str = "linear",
) -> CompressionResult:
    """Compress one normalized HWC texture.

    This compatibility helper accepts up to 16 channels. Prefer
    :func:`compress_texture_set` for named material maps.
    """
    image = _normalized_image(image, channel_name, max_channels=16)
    descriptors = tuple(
        TextureChannel(
            channel_name if first == 0 else f"{channel_name}_{first // 4}",
            first,
            min(4, image.shape[2] - first),
            color_space,
        )
        for first in range(0, image.shape[2], 4)
    )
    return _compress_channels(
        image,
        descriptors,
        np.ones(image.shape[2], dtype=np.float32),
        steps=steps,
        refinement_steps=refinement_steps,
        learning_rate=learning_rate,
        latent_scale=latent_scale,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
