# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration boundaries between the dependency and supplemental extension."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import optix
import pytest
from warp_optix import _ext, native


def test_optix_headers_are_available_without_separate_sdk(monkeypatch):
    monkeypatch.delenv("OPTIX_SDK_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("OPTIX_INCLUDE_DIR", raising=False)
    include = Path(native.get_optix_include_dir())
    assert (include / "optix.h").is_file()
    assert (include / "optix_device.h").is_file()
    assert (include / "OptiX_LICENSE.txt").is_file()
    from warp_optix.addon import build_options
    assert str(include) in build_options().extra_cuda_include_dirs


def test_optix_headers_editable_install_fallback(monkeypatch, tmp_path):
    backend = import_module("warp_optix._backend")
    monkeypatch.setattr(backend, "_package_dir", tmp_path / "source")
    monkeypatch.setattr(backend, "_get_package_dir", lambda: tmp_path / "installed")
    include = tmp_path / "installed" / "include"
    include.mkdir(parents=True)
    (include / "optix.h").touch()
    (include / "optix_device.h").touch()
    assert Path(native.get_optix_include_dir()) == include


def test_optix_headers_incomplete_package(monkeypatch, tmp_path):
    backend = import_module("warp_optix._backend")
    monkeypatch.setattr(backend, "_package_dir", tmp_path)
    monkeypatch.setattr(backend, "_get_package_dir", lambda: tmp_path)
    include = tmp_path / "include"
    include.mkdir()
    (include / "optix.h").touch()
    with pytest.raises(FileNotFoundError, match="missing or incomplete; reinstall"):
        native.get_optix_include_dir()


@pytest.mark.parametrize("first", ["optix", "warp_optix"])
def test_dependency_and_supplement_import_in_either_order(first, tmp_path):
    second = "warp_optix" if first == "optix" else "optix"
    subprocess.run(
        [sys.executable, "-c", f"""
import {first}
import {second}
import sys
from warp_optix import native
assert sys.modules['optix'] is optix
assert optix.__spec__.name == 'optix'
assert 'warp_optix._bindings' not in sys.modules
for name in ('DeviceContext', 'Module', 'ProgramGroup', 'Pipeline', 'Denoiser',
             'BuildInputTriangleArray', 'Image2D', 'ModuleCompileOptions'):
    assert getattr(native, name) is getattr(optix, name), name
assert native.version() == (9, 1, 0)
assert warp_optix._ext.optix_version() == optix.version()
assert hasattr(native.DenoiserParams(), 'temporalModeUsePreviousLayers')
assert hasattr(optix.DeviceContext, 'coopVecMatrixConvert')
"""],
        cwd=tmp_path,
        check=True,
    )


def test_unsupported_wrapper_abi_rejected_before_installing_methods():
    incompatible = ModuleType("incompatible_optix")
    incompatible.__dict__.update(optix.__dict__)
    incompatible.DeviceContext = type("DifferentContext", (), {})
    original = optix.DeviceContext.coopVecMatrixConvert
    with pytest.raises(ImportError, match="ABI-compatible pyoptix 9.1.0"):
        _ext.install_pyoptix_extensions(incompatible)
    assert optix.DeviceContext.coopVecMatrixConvert is original


def test_unsupported_dependency_version_rejected(tmp_path):
    subprocess.run(
        [sys.executable, "-c", """
import importlib.metadata
original = importlib.metadata.version
importlib.metadata.version = lambda name: '9.2.0' if name == 'pyoptix' else original(name)
try:
    import warp_optix
except ImportError as error:
    assert 'requires pyoptix==9.1.0' in str(error), error
else:
    raise AssertionError('Unsupported dependency was accepted')
"""],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "upstream_sdk, extension_sdk",
    [((9, 0, 0), (9, 1, 0)), ((9, 1, 0), (9, 0, 0)), ((9, 0, 0), (9, 0, 0))],
)
def test_compiled_sdk_mismatch_rejected_before_installing_methods(
    upstream_sdk, extension_sdk, tmp_path
):
    subprocess.run(
        [sys.executable, "-c", f"""
import sys
from types import ModuleType
import optix
assert not hasattr(optix.DeviceContext, 'coopVecMatrixConvert')
optix.version = lambda: {upstream_sdk!r}
extension = ModuleType('warp_optix._ext')
extension.optix_version = lambda: {extension_sdk!r}
def install(upstream):
    raise AssertionError('Bindings installed before checking SDK versions')
extension.install_pyoptix_extensions = install
sys.modules[extension.__name__] = extension
try:
    import warp_optix
except ImportError as error:
    message = str(error)
    assert 'OptiX SDK version mismatch' in message, message
    assert 'pyoptix was compiled with {'.'.join(map(str, upstream_sdk))}' in message, message
    assert 'warp-optix was compiled with {'.'.join(map(str, extension_sdk))}' in message, message
else:
    raise AssertionError('Unsupported SDK was accepted')
assert not hasattr(optix.DeviceContext, 'coopVecMatrixConvert')
"""],
        cwd=tmp_path,
        check=True,
    )


def test_native_installer_rejects_compiled_sdk_mismatch():
    incompatible = ModuleType("incompatible_optix")
    incompatible.__dict__.update(optix.__dict__)
    incompatible.version = lambda: (9, 0, 0)
    original = optix.DeviceContext.coopVecMatrixConvert
    with pytest.raises(ImportError, match="OptiX SDK version mismatch"):
        _ext.install_pyoptix_extensions(incompatible)
    assert optix.DeviceContext.coopVecMatrixConvert is original


def test_supplemental_enums_work_with_dependency_structs():
    curves = native.BuiltinISOptions(
        builtinISModuleType=native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER
    )
    assert int(curves.builtinISModuleType) == int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    image = optix.Image2D()
    image.format = native.PIXEL_FORMAT_FLOAT2
    assert int(image.format) == int(native.PIXEL_FORMAT_FLOAT2)
