# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest
from warp_optix_pathtracing.asset_loaders import (
    _build_texture_list,
    _load_gltf_images,
    _load_gltf_json_and_buffers,
)


def test_missing_gltf_image_does_not_shift_texture_sources():
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[..., 0] = 0.25
    textures = _build_texture_list({"textures": [{"source": 0}, {"source": 1}]}, [None, image])

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


def _link_directory(link, target):
    if os.name == "nt":
        # Junctions do not require Windows developer mode or symlink privileges.
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "New-Item -ItemType Junction -Path $env:WARP_OPTIX_TEST_LINK "
                "-Target $env:WARP_OPTIX_TEST_TARGET | Out-Null",
            ],
            env={**os.environ, "WARP_OPTIX_TEST_LINK": str(link), "WARP_OPTIX_TEST_TARGET": str(target)},
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def test_gltf_resources_resolve_inside_asset_directory(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "mesh.bin").write_bytes(b"mesh")
    _link_directory(tmp_path / "linked", data)
    scene = tmp_path / "scene.gltf"
    for uri in ("data/../data/mesh.bin", "linked/mesh.bin"):
        _write_gltf(scene, uri)
        _, buffers = _load_gltf_json_and_buffers(scene)
        assert buffers == [b"mesh"]


def test_gltf_resources_cannot_escape_asset_directory(tmp_path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"secret")
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    _link_directory(asset_dir / "linked", tmp_path)
    scene = asset_dir / "scene.gltf"
    for uri in ("../secret.bin", "data/../../secret.bin", str(secret), "linked/secret.bin"):
        _write_gltf(scene, uri)
        with pytest.raises(ValueError, match="outside the asset directory"):
            _load_gltf_json_and_buffers(scene)
    if importlib.util.find_spec("imageio") is None:
        return  # Image loading needs the pathtracing extra.
    for uri in ("../secret.bin", "linked/secret.bin"):
        with pytest.raises(ValueError, match="outside the asset directory"):
            _load_gltf_images(scene, {"images": [{"uri": uri}]}, [])
