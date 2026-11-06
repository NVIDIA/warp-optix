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


def test_kernels_need_no_cuda_toolkit_headers():
    # The packaged OptiX headers and warp_optix_builtins.h must be enough for
    # NVRTC, so every GPU test also covers kernels compiling without a Toolkit.
    from warp_optix.addon import _HEADER_DIR, build_options

    assert build_options().extra_cuda_include_dirs == [str(_HEADER_DIR), native.get_optix_include_dir()]


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
        [
            sys.executable,
            "-c",
            f"""
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
assert callable(native.coopVecMatrixConvert)
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_importing_leaves_pyoptix_module_unmodified(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import optix

def snapshot():
    return {
        name: (value, dict(vars(value)) if isinstance(value, type) else None)
        for name, value in vars(optix).items()
    }

before = snapshot()
import warp_optix
from warp_optix import native
assert native.coopVecMatrixConvert and native.moduleCreate
after = snapshot()
assert before.keys() == after.keys(), sorted(before.keys() ^ after.keys())
for name, (value, members) in before.items():
    assert after[name][0] is value, name
    if members is not None:
        assert members.keys() == after[name][1].keys(), (name, sorted(members.keys() ^ after[name][1].keys()))
        for member, attribute in members.items():
            assert after[name][1][member] is attribute, (name, member)
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_unsupported_wrapper_abi_rejected():
    incompatible = ModuleType("incompatible_optix")
    incompatible.__dict__.update(optix.__dict__)
    incompatible.DeviceContext = type("DifferentContext", (), {})
    with pytest.raises(ImportError, match="ABI-compatible pyoptix 9.1.1"):
        _ext.validate_pyoptix_bindings(incompatible)


def test_unsupported_dependency_version_rejected(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.metadata
original = importlib.metadata.version
importlib.metadata.version = lambda name: '9.2.0' if name == 'pyoptix' else original(name)
try:
    import warp_optix
except ImportError as error:
    assert 'requires pyoptix==9.1.1' in str(error), error
else:
    raise AssertionError('Unsupported dependency was accepted')
""",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "upstream_sdk, extension_sdk",
    [((9, 0, 0), (9, 1, 0)), ((9, 1, 0), (9, 0, 0)), ((9, 0, 0), (9, 0, 0))],
)
def test_compiled_sdk_mismatch_rejected_before_validating_bindings(upstream_sdk, extension_sdk, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
from types import ModuleType
import optix
optix.version = lambda: {upstream_sdk!r}
extension = ModuleType('warp_optix._ext')
extension.optix_version = lambda: {extension_sdk!r}
def validate(upstream):
    raise AssertionError('Bindings validated before checking SDK versions')
extension.validate_pyoptix_bindings = validate
sys.modules[extension.__name__] = extension
try:
    import warp_optix
except ImportError as error:
    message = str(error)
    assert 'OptiX SDK version mismatch' in message, message
    assert 'pyoptix was compiled with {".".join(map(str, upstream_sdk))}' in message, message
    assert 'warp-optix was compiled with {".".join(map(str, extension_sdk))}' in message, message
else:
    raise AssertionError('Unsupported SDK was accepted')
""",
        ],
        cwd=tmp_path,
        check=True,
    )


def test_native_validator_rejects_compiled_sdk_mismatch():
    incompatible = ModuleType("incompatible_optix")
    incompatible.__dict__.update(optix.__dict__)
    incompatible.version = lambda: (9, 0, 0)
    with pytest.raises(ImportError, match="OptiX SDK version mismatch"):
        _ext.validate_pyoptix_bindings(incompatible)


def test_supplemental_enums_work_with_dependency_structs():
    curves = native.BuiltinISOptions(builtinISModuleType=native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    assert int(curves.builtinISModuleType) == int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    image = optix.Image2D()
    image.format = native.PIXEL_FORMAT_FLOAT2
    assert int(image.format) == int(native.PIXEL_FORMAT_FLOAT2)
