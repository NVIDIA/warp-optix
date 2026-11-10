# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable, memory-mappable neural-texture asset format.

The format intentionally stores portable row-major FP16 network parameters.
OptiX's device-specific inference-optimal layout is produced once at load time.
Latent features, which dominate asset size, are packed at four bits per value.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"WNTX"
VERSION_MAJOR = 1
VERSION_MINOR = 0
ALIGNMENT = 64
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
STANDARD_ARCHITECTURE = "ntc-8x32x32x16-v1"
STANDARD_LAYER_SHAPES = ((32, 32), (32, 32), (16, 32))
STANDARD_ACTIVATIONS = ("hgelu", "hgelu", "none")
_HEADER = struct.Struct("<4sHHQQ")


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_latents(values: np.ndarray) -> np.ndarray:
    """Quantize ``[-1, 1]`` features and pack four nibbles per uint16 texel word."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("latent values must have shape (height, width, features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("latent values must be finite")
    height, width, features = values.shape
    words = (features + 3) // 4
    quantized = np.rint((np.clip(values, -1.0, 1.0) + 1.0) * 7.5).astype(np.uint16)
    padded = np.zeros((height, width, words * 4), dtype=np.uint16)
    padded[..., :features] = quantized
    packed = padded[..., 0::4] | (padded[..., 1::4] << 4) | (padded[..., 2::4] << 8) | (padded[..., 3::4] << 12)
    return np.ascontiguousarray(packed)


def unpack_latents(packed: np.ndarray, features: int) -> np.ndarray:
    """Unpack uint16 latent words to float32 features in ``[-1, 1]``."""
    packed = np.asarray(packed, dtype=np.uint16)
    if packed.ndim != 3 or features <= 0 or packed.shape[2] != (features + 3) // 4:
        raise ValueError("packed latent shape does not match feature count")
    result = np.empty((*packed.shape[:2], packed.shape[2] * 4), dtype=np.float32)
    for lane in range(4):
        result[..., lane::4] = ((packed >> (4 * lane)) & 0xF).astype(np.float32)
    return result[..., :features] * (2.0 / 15.0) - 1.0


@dataclass(frozen=True)
class TextureChannel:
    name: str
    first_channel: int
    channel_count: int
    color_space: str = "linear"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("texture channel name must not be empty")
        if (
            type(self.first_channel) is not int
            or type(self.channel_count) is not int
            or self.first_channel < 0
            or not 1 <= self.channel_count <= 4
        ):
            raise ValueError("invalid texture channel range")
        if not isinstance(self.color_space, str) or self.color_space not in {
            "linear",
            "srgb",
        }:
            raise ValueError("color_space must be 'linear' or 'srgb'")


@dataclass(frozen=True)
class NetworkLayer:
    weights: np.ndarray
    bias: np.ndarray
    activation: str = "hgelu"

    def __post_init__(self):
        weights = self.weights if isinstance(self.weights, np.ndarray) else np.asarray(self.weights)
        bias = self.bias if isinstance(self.bias, np.ndarray) else np.asarray(self.bias)
        if weights.dtype != np.float16 or not weights.flags.c_contiguous:
            weights = np.ascontiguousarray(weights, dtype=np.float16)
        if bias.dtype != np.float16 or not bias.flags.c_contiguous:
            bias = np.ascontiguousarray(bias, dtype=np.float16)
        if weights.ndim != 2 or bias.ndim != 1 or weights.shape[0] != bias.shape[0]:
            raise ValueError("layer weights must be (out, in) and bias must be (out,)")
        if not weights.shape[0] or not weights.shape[1] or weights.shape[0] % 16 or weights.shape[1] % 16:
            raise ValueError("OptiX cooperative-vector layer dimensions must be positive multiples of 16")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("network parameters must be finite")
        if self.activation not in {"hgelu", "none"}:
            raise ValueError("activation must be 'hgelu' or 'none'")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)


@dataclass(frozen=True)
class NeuralTextureAsset:
    width: int
    height: int
    channels: tuple[TextureChannel, ...]
    latent_features: int
    latent_mips: tuple[np.ndarray, ...]
    layers: tuple[NetworkLayer, ...]
    metadata: dict = field(default_factory=dict)

    architecture: str = STANDARD_ARCHITECTURE

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("texture dimensions must be positive")
        if self.architecture != STANDARD_ARCHITECTURE or self.latent_features != 8:
            raise ValueError(f"version 1 supports only {STANDARD_ARCHITECTURE} with 8 latent features")
        if not self.latent_mips:
            raise ValueError("at least one latent mip is required")
        words = (self.latent_features + 3) // 4
        mips = []
        for level, mip in enumerate(self.latent_mips):
            if mip.dtype != np.uint16 or not mip.flags.c_contiguous:
                mip = np.ascontiguousarray(mip, dtype=np.uint16)
            mip = mip if isinstance(mip, np.ndarray) else np.asarray(mip)
            if mip.ndim != 3 or not mip.shape[0] or not mip.shape[1] or mip.shape[2] != words:
                raise ValueError(f"latent mip {level} must have shape (height, width, {words})")
            mips.append(mip)

        shapes = tuple(layer.weights.shape for layer in self.layers)
        activations = tuple(layer.activation for layer in self.layers)
        if shapes != STANDARD_LAYER_SHAPES or activations != STANDARD_ACTIVATIONS:
            raise ValueError(
                f"{STANDARD_ARCHITECTURE} requires layer shapes {STANDARD_LAYER_SHAPES} "
                f"and activations {STANDARD_ACTIVATIONS}"
            )
        if not self.channels:
            raise ValueError("at least one texture channel mapping is required")
        used_channels = set()
        for channel in self.channels:
            indices = set(range(channel.first_channel, channel.first_channel + channel.channel_count))
            if max(indices) >= 16:
                raise ValueError("texture channel mapping exceeds the 16 decoder outputs")
            if used_channels & indices:
                raise ValueError("texture channel mappings must not overlap")
            used_channels.update(indices)
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "latent_mips", tuple(mips))
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def channel_count(self) -> int:
        return max((c.first_channel + c.channel_count for c in self.channels), default=0)

    @property
    def storage_bytes(self) -> int:
        return sum(x.nbytes for x in self.latent_mips) + sum(
            layer.weights.nbytes + layer.bias.nbytes for layer in self.layers
        )


@dataclass(frozen=True)
class CompressionResult:
    """Result of offline compression, kept dependency-free for public typing."""

    asset: NeuralTextureAsset
    losses: tuple[float, ...]
    psnr: float
    psnr_by_texture: dict[str, float] = field(default_factory=dict)
    evaluation_pixels: int = 0
    selection_reason: str = ""


def _array_view(array: np.ndarray, offset: int) -> dict:
    return {
        "offset": offset,
        "nbytes": array.nbytes,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def save_asset(path: str | os.PathLike, asset: NeuralTextureAsset) -> None:
    """Atomically write an asset. No pickle or executable metadata is used."""
    arrays: list[tuple[str, np.ndarray]] = []
    arrays.extend((f"latent_{i}", mip) for i, mip in enumerate(asset.latent_mips))
    for index, layer in enumerate(asset.layers):
        arrays.append((f"weight_{index}", layer.weights))
        arrays.append((f"bias_{index}", layer.bias))

    offset = 0
    views = {}
    for name, array in arrays:
        offset = _align(offset)
        views[name] = _array_view(array, offset)
        offset += array.nbytes

    manifest = {
        "schema": "warp-optix-neural-texture",
        "version": [VERSION_MAJOR, VERSION_MINOR],
        "architecture": asset.architecture,
        "texture": {"width": asset.width, "height": asset.height},
        "channels": [vars(channel) for channel in asset.channels],
        "latents": {
            "encoding": "snorm4x4-u16",
            "features": asset.latent_features,
            "views": [views[f"latent_{i}"] for i in range(len(asset.latent_mips))],
        },
        "network": {
            "weight_storage": "float16-row-major",
            "inference_weight": "float8-e4m3-optimal",
            "layers": [
                {
                    "weights": views[f"weight_{i}"],
                    "bias": views[f"bias_{i}"],
                    "activation": layer.activation,
                }
                for i, layer in enumerate(asset.layers)
            ],
        },
        "metadata": asset.metadata,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data_offset = _align(_HEADER.size + len(manifest_bytes))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(temp_fd, "wb") as stream:
            stream.write(
                _HEADER.pack(
                    MAGIC,
                    VERSION_MAJOR,
                    VERSION_MINOR,
                    len(manifest_bytes),
                    data_offset,
                )
            )
            stream.write(manifest_bytes)
            stream.write(b"\0" * (data_offset - stream.tell()))
            for name, array in arrays:
                absolute = data_offset + views[name]["offset"]
                stream.write(b"\0" * (absolute - stream.tell()))
                stream.write(memoryview(array).cast("B"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_asset(path: str | os.PathLike, *, mmap: bool = True, verify: bool = True) -> NeuralTextureAsset:
    """Load and validate an asset, optionally keeping payload arrays memory-mapped."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("truncated neural-texture header")
        magic, major, minor, manifest_size, data_offset = _HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError("not a Warp-OptiX neural-texture asset")
        if major != VERSION_MAJOR or minor > VERSION_MINOR:
            raise ValueError(f"unsupported neural-texture version {major}.{minor}")
        if (
            manifest_size > MAX_MANIFEST_BYTES
            or manifest_size > file_size
            or data_offset < _HEADER.size + manifest_size
            or data_offset > file_size
            or data_offset % ALIGNMENT
        ):
            raise ValueError("invalid neural-texture manifest bounds")
        try:
            manifest = json.loads(stream.read(manifest_size))
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("invalid neural-texture manifest JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("neural-texture manifest must be an object")
    if manifest.get("schema") != "warp-optix-neural-texture":
        raise ValueError("invalid neural-texture schema")

    def require_object(container: dict, key: str) -> dict:
        value = container.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"neural-texture manifest field '{key}' must be an object")
        return value

    texture = require_object(manifest, "texture")
    width = texture.get("width")
    height = texture.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("invalid neural-texture dimensions")

    architecture = manifest.get("architecture")
    if architecture != STANDARD_ARCHITECTURE:
        raise ValueError("unsupported neural-texture architecture")

    latent_info = require_object(manifest, "latents")
    latent_views = latent_info.get("views")
    if (
        latent_info.get("encoding") != "snorm4x4-u16"
        or type(latent_info.get("features")) is not int
        or not isinstance(latent_views, list)
        or not 1 <= len(latent_views) <= 32
        or not all(isinstance(view, dict) for view in latent_views)
    ):
        raise ValueError("invalid neural-texture latent manifest")

    network = require_object(manifest, "network")
    layer_infos = network.get("layers")
    if (
        network.get("weight_storage") != "float16-row-major"
        or not isinstance(layer_infos, list)
        or len(layer_infos) != len(STANDARD_LAYER_SHAPES)
        or not all(isinstance(layer, dict) for layer in layer_infos)
    ):
        raise ValueError("invalid neural-texture network manifest")
    for layer in layer_infos:
        if (
            not isinstance(layer.get("weights"), dict)
            or not isinstance(layer.get("bias"), dict)
            or not isinstance(layer.get("activation"), str)
        ):
            raise ValueError("invalid neural-texture layer manifest")

    channel_infos = manifest.get("channels")
    if (
        not isinstance(channel_infos, list)
        or not 1 <= len(channel_infos) <= 16
        or not all(isinstance(channel, dict) for channel in channel_infos)
    ):
        raise ValueError("invalid neural-texture channel manifest")
    try:
        channels = tuple(
            TextureChannel(
                channel["name"],
                channel["first_channel"],
                channel["channel_count"],
                channel.get("color_space", "linear"),
            )
            for channel in channel_infos
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid neural-texture channel manifest") from error

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("neural-texture metadata must be an object")

    def read_view(view: dict, expected_dtype, rank: int) -> np.ndarray:
        if not isinstance(view, dict) or view.get("dtype") != np.dtype(expected_dtype).str:
            raise ValueError("invalid neural-texture data-view dtype")
        raw_shape = view.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != rank
            or not all(type(x) is int and x > 0 for x in raw_shape)
        ):
            raise ValueError("invalid neural-texture data-view shape")
        shape = tuple(raw_shape)
        dtype = np.dtype(expected_dtype)
        nbytes = dtype.itemsize
        for dimension in shape:
            if nbytes > file_size // dimension:
                raise ValueError("neural-texture data view is too large")
            nbytes *= dimension
        relative_offset = view.get("offset")
        stored_nbytes = view.get("nbytes")
        checksum = view.get("sha256")
        if type(relative_offset) is not int or relative_offset < 0 or relative_offset % ALIGNMENT:
            raise ValueError("invalid neural-texture data-view offset")
        if type(stored_nbytes) is not int or not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("invalid neural-texture data-view metadata")
        offset = data_offset + relative_offset
        if nbytes != stored_nbytes or offset < data_offset or offset + nbytes > file_size:
            raise ValueError("invalid neural-texture data view")
        if mmap:
            result = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
        else:
            with path.open("rb") as stream:
                stream.seek(offset)
                result = np.frombuffer(stream.read(nbytes), dtype=dtype).reshape(shape).copy()
        if verify and hashlib.sha256(memoryview(np.asarray(result)).cast("B")).hexdigest() != view["sha256"]:
            raise ValueError("neural-texture payload checksum mismatch")
        return result

    layers = tuple(
        NetworkLayer(
            read_view(layer["weights"], np.float16, 2),
            read_view(layer["bias"], np.float16, 1),
            layer["activation"],
        )
        for layer in layer_infos
    )
    return NeuralTextureAsset(
        width=width,
        height=height,
        channels=channels,
        latent_features=latent_info["features"],
        latent_mips=tuple(read_view(view, np.uint16, 3) for view in latent_views),
        layers=layers,
        architecture=architecture,
        metadata=dict(metadata),
    )
