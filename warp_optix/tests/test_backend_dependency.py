# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration boundaries between the dependency and supplemental extension."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import optix
import pytest

from warp_optix import _ext, native


def test_packaged_optix_headers_are_all_kernels_need(monkeypatch):
    monkeypatch.delenv("OPTIX_SDK_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("OPTIX_INCLUDE_DIR", raising=False)
    include = Path(native.get_optix_include_dir())
    for name in ("optix.h", "optix_device.h", "OptiX_LICENSE.txt"):
        assert (include / name).is_file(), name
    # No separate OptiX SDK and no CUDA Toolkit headers: every GPU test compiles with only these.
    from warp_optix.addon import _HEADER_DIR, build_options

    assert build_options().extra_cuda_include_dirs == [str(_HEADER_DIR), str(include)]


def test_optix_header_lookup_falls_back_to_the_installed_package(monkeypatch, tmp_path):
    backend = import_module("warp_optix._backend")
    monkeypatch.setattr(backend, "_package_dir", tmp_path / "source")
    monkeypatch.setattr(backend, "_get_package_dir", lambda: tmp_path / "installed")
    include = tmp_path / "installed" / "include"
    include.mkdir(parents=True)
    (include / "optix.h").touch()
    with pytest.raises(FileNotFoundError, match="missing or incomplete; reinstall"):
        native.get_optix_include_dir()
    (include / "optix_device.h").touch()
    assert Path(native.get_optix_include_dir()) == include


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
for name in ('DeviceContext', 'Module', 'ProgramGroup', 'Pipeline', 'Denoiser',
             'BuildInputTriangleArray', 'Image2D', 'ModuleCompileOptions'):
    assert getattr(native, name) is getattr(optix, name), name
assert warp_optix._ext.optix_version() == optix.version()
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


def test_native_validator_rejects_other_abi_and_sdk():
    other_abi = ModuleType("incompatible_optix")
    other_abi.__dict__.update(optix.__dict__)
    other_abi.DeviceContext = type("DifferentContext", (), {})
    with pytest.raises(ImportError, match="ABI-compatible pyoptix 9.1.1"):
        _ext.validate_pyoptix_bindings(other_abi)
    other_sdk = ModuleType("incompatible_optix")
    other_sdk.__dict__.update(optix.__dict__)
    other_sdk.version = lambda: (9, 0, 0)
    with pytest.raises(ImportError, match="OptiX SDK version mismatch"):
        _ext.validate_pyoptix_bindings(other_sdk)


def test_loading_rejects_other_dependency_version_and_sdk_before_validating(monkeypatch):
    backend = import_module("warp_optix._backend")
    monkeypatch.setattr(backend, "version", lambda name: "9.2.0")
    with pytest.raises(ImportError, match="requires pyoptix==9.1.1"):
        backend._load_backend()
    monkeypatch.setattr(backend, "version", lambda name: "9.1.1")

    def validate(upstream):
        raise AssertionError("Bindings validated before checking SDK versions")

    for upstream_sdk, extension_sdk in (((9, 0, 0), (9, 1, 0)), ((9, 1, 0), (9, 0, 0)), ((9, 0, 0), (9, 0, 0))):
        monkeypatch.setattr(optix, "version", lambda sdk=upstream_sdk: sdk)
        extension = SimpleNamespace(optix_version=lambda sdk=extension_sdk: sdk, validate_pyoptix_bindings=validate)
        monkeypatch.setattr(backend, "_extensions", extension)
        with pytest.raises(ImportError, match="OptiX SDK version mismatch") as error:
            backend._load_backend()
        message = str(error.value)
        assert f"pyoptix was compiled with {'.'.join(map(str, upstream_sdk))}" in message
        assert f"warp-optix was compiled with {'.'.join(map(str, extension_sdk))}" in message


def test_supplemental_enums_work_with_dependency_structs():
    curves = native.BuiltinISOptions(builtinISModuleType=native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    assert int(curves.builtinISModuleType) == int(native.PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
    image = optix.Image2D()
    image.format = native.PIXEL_FORMAT_FLOAT2
    assert int(image.format) == int(native.PIXEL_FORMAT_FLOAT2)
