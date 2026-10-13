# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from warp_optix.pathtracing.asset_loaders import (
    _build_texture_list,
    _load_gltf_images,
    _load_gltf_json_and_buffers,
)


def test_missing_gltf_image_does_not_shift_texture_sources():
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[..., 0] = 0.25
    textures = _build_texture_list(
        {"textures": [{"source": 0}, {"source": 1}]}, [None, image]
    )

    np.testing.assert_array_equal(textures[0], np.ones((1, 1, 4), dtype=np.float32))
    assert textures[1] is image


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gltf_image_normalization_uses_source_bit_depth(monkeypatch, tmp_path, dtype):
    imageio = pytest.importorskip("imageio.v3")
    maximum = np.iinfo(dtype).max
    monkeypatch.setattr(
        imageio,
        "imread",
        lambda _stream: np.full((1, 1, 3), maximum, dtype=dtype),
    )
    images = _load_gltf_images(
        tmp_path,
        {"images": [{"uri": "data:image/png;base64,AA=="}]},
        [],
    )
    np.testing.assert_array_equal(images[0], np.ones((1, 1, 4), dtype=np.float32))


def _write_gltf(path, uri):
    path.write_text(json.dumps({"buffers": [{"uri": uri, "byteLength": 4}]}))


def test_gltf_buffer_uri_resolves_inside_asset_directory(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mesh.bin").write_bytes(b"\x01\x02\x03\x04")
    scene = tmp_path / "scene.gltf"
    _write_gltf(scene, "data/../data/mesh.bin")

    _, buffers = _load_gltf_json_and_buffers(scene)

    assert buffers == [b"\x01\x02\x03\x04"]


@pytest.mark.parametrize("uri", ["../secret.bin", "data/../../secret.bin", "ABSOLUTE"])
def test_gltf_buffer_uri_cannot_escape_asset_directory(tmp_path, uri):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"secret")
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    scene = asset_dir / "scene.gltf"
    _write_gltf(scene, str(secret) if uri == "ABSOLUTE" else uri)

    with pytest.raises(ValueError, match="outside the asset directory"):
        _load_gltf_json_and_buffers(scene)


def test_gltf_image_uri_cannot_escape_asset_directory(tmp_path):
    (tmp_path / "secret.png").write_bytes(b"not an image")
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()

    with pytest.raises(ValueError, match="outside the asset directory"):
        _load_gltf_images(asset_dir / "scene.gltf", {"images": [{"uri": "../secret.png"}]}, [])
