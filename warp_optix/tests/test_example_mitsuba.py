# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from scenes import mitsuba


def test_download_publishes_complete_scene_and_rejects_escaping_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("WARP_OPTIX_CACHE", str(tmp_path))
    calls = []

    def download(url, archive):
        calls.append(url)
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("glass-of-water/scene_v3.xml", "<scene/>")
            zf.writestr("glass-of-water/mesh.obj", "v 0 0 0")

    monkeypatch.setattr(mitsuba.urllib.request, "urlretrieve", download)
    xml = mitsuba.fetch("glass-of-water")
    assert xml.read_text() == "<scene/>"
    assert (xml.parent / "mesh.obj").is_file()
    assert mitsuba.fetch("glass-of-water") == xml
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Unknown bundled scene"):
        mitsuba.fetch("../../escape")
    for name in ("../secret.obj", str(tmp_path / "secret.obj")):
        with pytest.raises(ValueError, match="escapes its directory"):
            mitsuba._asset_path(xml.parent, name)


def test_download_failure_does_not_poison_scene_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WARP_OPTIX_CACHE", str(tmp_path))

    def download(url, archive):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("dining-room/scene_v3.xml", "<scene/>")
            zf.writestr("../../escape.obj", "untrusted")

    monkeypatch.setattr(mitsuba.urllib.request, "urlretrieve", download)
    with pytest.raises(ValueError, match="escapes its directory"):
        mitsuba.fetch("dining-room")
    assert not (tmp_path / "scenes" / "dining-room").exists()
    assert list((tmp_path / "scenes").iterdir()) == []


def test_obj_resolves_relative_indices_at_each_face_and_missing_attributes(tmp_path):
    obj = tmp_path / "mesh.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 0.5\nvt 0.75 1\n"
        "vn 0 0 1\nvn 0 1 0\nf -3/1/1 -2/2/1 -1// # first triangle\n"
        "v 10 10 10\nf 1 3 2\n"
    )
    positions, uvs, normals = mitsuba._read_obj(obj)
    np.testing.assert_array_equal(positions[:3], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(uvs[:3], [[0.25, 0.5], [0.75, 1], [0, 0]])
    np.testing.assert_array_equal(normals[:3], [[0, 0, 1]] * 3)
    np.testing.assert_array_equal(normals[3:], [[0, 0, -1]] * 3)
    obj.write_text("v 0 0 0\nf 0 1 1\n")
    with pytest.raises(ValueError, match="Invalid OBJ index"):
        mitsuba._read_obj(obj)
