# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neural texture assets and OptiX inference support.

Runtime modules intentionally do not import :mod:`warp_nn`; it is only used by
the optional offline training tools.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .device import NeuralTextureView as NeuralTextureView
from .device import neural_texture_sample as neural_texture_sample
from .device import neural_texture_sample_texel as neural_texture_sample_texel
from .format import CompressionResult as CompressionResult
from .format import NetworkLayer as NetworkLayer
from .format import NeuralTextureAsset as NeuralTextureAsset
from .format import TextureChannel as TextureChannel
from .format import load_asset as load_asset
from .format import pack_latents as pack_latents
from .format import save_asset as save_asset
from .format import unpack_latents as unpack_latents
from .reference import decode_image as decode_image
from .reference import decode_texel as decode_texel
from .reference import network_input as network_input
from .runtime import NeuralTextureRuntime as NeuralTextureRuntime
from .runtime import upload_asset as upload_asset


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
    """Compress a texture, automatically balancing size and sampled quality.

    The import is deliberately lazy so loading or evaluating a neural texture
    never requires the offline-training dependency.
    """
    from .training import compress_texture as _compress_texture

    return _compress_texture(
        image,
        steps=steps,
        refinement_steps=refinement_steps,
        learning_rate=learning_rate,
        latent_scale=latent_scale,
        batch_size=batch_size,
        device=device,
        seed=seed,
        channel_name=channel_name,
        color_space=color_space,
    )


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
    """Compress named material textures, automatically balancing size and quality.

    Importing this module remains independent of the optional training package;
    warp-nn is loaded only when this function is called.
    """
    from .training import compress_texture_set as _compress_texture_set

    return _compress_texture_set(
        textures,
        color_spaces=color_spaces,
        channel_weights=channel_weights,
        steps=steps,
        refinement_steps=refinement_steps,
        learning_rate=learning_rate,
        latent_scale=latent_scale,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


__all__ = [
    "CompressionResult",
    "NetworkLayer",
    "NeuralTextureAsset",
    "NeuralTextureRuntime",
    "NeuralTextureView",
    "TextureChannel",
    "compress_texture",
    "compress_texture_set",
    "decode_image",
    "decode_texel",
    "load_asset",
    "network_input",
    "neural_texture_sample",
    "neural_texture_sample_texel",
    "pack_latents",
    "save_asset",
    "unpack_latents",
    "upload_asset",
]
