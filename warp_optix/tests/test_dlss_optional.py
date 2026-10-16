# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from warp_optix import native

# Any GUID-formatted string is accepted as an NGX project ID for testing.
PROJECT_ID = "0e7f3a52-8c1b-4d0e-9a57-2f6c4b8d1e93"


def test_dlss_bindings_are_scoped_and_report_results():
    # Using DLSS is optional, but every supported wheel includes the bindings.
    assert native.dlss_support_available() is True
    for name in ("D", "E", "F", "DEFAULT", "BALANCED", "MAX_QUALITY", "DLAA", "RESOURCE_COLOR_IN"):
        assert name not in dir(native)
    assert native.dlssRRGetResultString(1) == "NVSDK_NGX_Result_Success"


def test_dlss_contexts_share_sdk_lifetime(tmp_path):
    first = native.DlssRRContext()
    second = native.DlssRRContext()
    try:
        try:
            first.init(PROJECT_ID, applicationDataPath=str(tmp_path))
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
        second.init(PROJECT_ID, applicationDataPath=str(tmp_path))
        first.deinit()
        assert isinstance(second.isDlssRRAvailable(), bool)
    finally:
        second.deinit()
        first.deinit()


def test_dlss_rejects_relative_feature_search_path():
    context = native.DlssRRContext()
    with pytest.raises(ValueError, match="absolute"):
        context.init(PROJECT_ID, featureSearchPath="relative/dir")


def test_dlss_data_directory_is_private(tmp_path, monkeypatch):
    cache_variable = "LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"
    monkeypatch.setenv(cache_variable, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    context = native.DlssRRContext()
    try:
        try:
            context.init(PROJECT_ID)
        except RuntimeError as error:
            pytest.skip(f"DLSS runtime is unavailable: {error}")
    finally:
        context.deinit()
    data_dir = tmp_path / "warp_optix" / "ngx"
    assert data_dir.is_dir()
    if os.name == "posix":
        assert data_dir.stat().st_mode & 0o777 == 0o700
    assert not any(p.suffix == ".log" for p in tmp_path.iterdir())
