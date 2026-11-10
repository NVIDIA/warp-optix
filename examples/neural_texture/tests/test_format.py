# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from neural_texture import (
    NetworkLayer,
    NeuralTextureAsset,
    TextureChannel,
    load_asset,
    pack_latents,
    save_asset,
    unpack_latents,
)
from neural_texture import format as nt_format


def make_asset():
    rng = np.random.default_rng(7)
    latent_values = rng.uniform(-1.0, 1.0, (4, 6, 8)).astype(np.float32)
    return NeuralTextureAsset(
        width=16,
        height=12,
        channels=(TextureChannel("albedo", 0, 3, "srgb"),),
        latent_features=8,
        latent_mips=(
            pack_latents(latent_values),
            pack_latents(latent_values[::2, ::2]),
        ),
        layers=(
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(
                np.ones((16, 32), dtype=np.float16),
                np.zeros(16, dtype=np.float16),
                "none",
            ),
        ),
        metadata={"source": "unit-test"},
    )


def test_latent_nibble_roundtrip_has_expected_error():
    values = np.linspace(-1.0, 1.0, 7 * 5 * 8, dtype=np.float32).reshape(7, 5, 8)
    packed = pack_latents(values)
    restored = unpack_latents(packed, 8)
    assert packed.shape == (7, 5, 2)
    assert packed.nbytes == 7 * 5 * 8 // 2
    np.testing.assert_allclose(restored, values, atol=1.0 / 15.0)


def test_asset_roundtrip_is_memory_mappable(tmp_path):
    path = tmp_path / "material.wnt"
    source = make_asset()
    save_asset(path, source)
    loaded = load_asset(path)

    assert loaded.width == source.width
    assert loaded.height == source.height
    assert loaded.channels == source.channels
    assert loaded.metadata == source.metadata
    assert isinstance(loaded.latent_mips[0], np.memmap)
    np.testing.assert_array_equal(loaded.latent_mips[0], source.latent_mips[0])
    np.testing.assert_array_equal(loaded.layers[0].weights, source.layers[0].weights)
    assert path.stat().st_size < source.storage_bytes + 2048


def test_asset_detects_payload_corruption(tmp_path):
    path = tmp_path / "corrupt.wnt"
    save_asset(path, make_asset())
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        byte = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="checksum"):
        load_asset(path)


def test_asset_rejects_manifest_controlled_object_dtype(tmp_path, monkeypatch):
    path = tmp_path / "unsafe.wnt"
    save_asset(path, make_asset())
    with path.open("rb") as stream:
        _, _, _, manifest_size, _ = nt_format._HEADER.unpack(stream.read(nt_format._HEADER.size))
        manifest = json.loads(stream.read(manifest_size))
    manifest["latents"]["views"][0]["dtype"] = "|O"
    monkeypatch.setattr(nt_format.json, "loads", lambda _: manifest)
    with pytest.raises(ValueError, match="dtype"):
        load_asset(path)


def test_asset_rejects_malformed_network():
    with pytest.raises(ValueError, match="multiples of 16"):
        NetworkLayer(np.zeros((3, 4)), np.zeros(3))


def test_asset_normalizes_missing_manifest_fields(tmp_path, monkeypatch):
    path = tmp_path / "missing-field.wnt"
    save_asset(path, make_asset())
    monkeypatch.setattr(
        nt_format.json,
        "loads",
        lambda _: {"schema": "warp-optix-neural-texture"},
    )
    with pytest.raises(ValueError, match="texture.*object"):
        load_asset(path)


def test_asset_rejects_extra_layers_before_payload_reads(tmp_path, monkeypatch):
    path = tmp_path / "extra-layers.wnt"
    save_asset(path, make_asset())
    with path.open("rb") as stream:
        _, _, _, manifest_size, _ = nt_format._HEADER.unpack(stream.read(nt_format._HEADER.size))
        manifest = json.loads(stream.read(manifest_size))
    manifest["network"]["layers"].append(manifest["network"]["layers"][0])
    monkeypatch.setattr(nt_format.json, "loads", lambda _: manifest)
    monkeypatch.setattr(
        nt_format.np,
        "memmap",
        lambda *args, **kwargs: pytest.fail("payload was read before validation"),
    )
    with pytest.raises(ValueError, match="network manifest"):
        load_asset(path)


def test_runtime_package_has_no_warp_nn_dependency():
    code = """
import importlib.abc, sys
class BlockWarpNN(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'warp_nn' or fullname.startswith('warp_nn.'):
            raise RuntimeError('inference imported warp_nn')
        return None
sys.meta_path.insert(0, BlockWarpNN())
import neural_texture
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
