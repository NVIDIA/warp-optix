# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from warp_optix import native as optix
import pytest


def test_dlss_support_query():
    # Using DLSS is optional, but every supported wheel includes the bindings.
    assert optix.dlss_support_available() is True


def test_dlss_contexts_share_sdk_lifetime():
    first = optix.DlssRRContext()
    second = optix.DlssRRContext()
    try:
        try:
            first.init()
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
        second.init()
        first.deinit()
        assert isinstance(second.isDlssRRAvailable(), bool)
    finally:
        second.deinit()
        first.deinit()


def test_dlss_rejects_relative_feature_search_path():
    context = optix.DlssRRContext()
    with pytest.raises(ValueError, match="absolute"):
        context.init(featureSearchPath="relative/dir")


def test_dlss_data_directory_is_private(tmp_path, monkeypatch):
    cache_variable = "LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"
    monkeypatch.setenv(cache_variable, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    context = optix.DlssRRContext()
    try:
        try:
            context.init()
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
    finally:
        context.deinit()
    data_dir = tmp_path / "warp_optix" / "ngx"
    assert data_dir.is_dir()
    if os.name == "posix":
        assert data_dir.stat().st_mode & 0o777 == 0o700
    assert not any(p.suffix == ".log" for p in tmp_path.iterdir())
