# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose PyOptiX with the small extension shipped by warp-optix.

PyOptiX owns every standard OptiX object and its destruction. The extension
adds missing operations as functions; it never modifies the `optix` module or
registers a second copy of its types.
"""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

_package_dir = Path(__file__).resolve().parent


def _get_package_dir():
    # Editable installs load Python from the source tree and native/data files
    # from the environment. Resolve runtime libraries consistently.
    return Path(_extensions.__file__).resolve().parent


def _get_optix_include_dir():
    # Editable installs keep SDK data alongside the installed native extension.
    for root in (_package_dir, _get_package_dir()):
        include = root / "include"
        if (include / "optix.h").is_file() and (include / "optix_device.h").is_file():
            return str(include)
    raise FileNotFoundError("Packaged OptiX SDK headers are missing or incomplete; reinstall warp-optix.")


def _load_backend():
    if version("pyoptix") != "9.1.1":
        raise ImportError("warp-optix requires pyoptix==9.1.1; reinstall its dependencies")
    upstream = import_module("optix")
    upstream_sdk = tuple(upstream.version())
    extension_sdk = tuple(_extensions.optix_version())
    if upstream_sdk != extension_sdk or extension_sdk != (9, 1, 0):
        raise ImportError(
            "OptiX SDK version mismatch: pyoptix was compiled with "
            + ".".join(map(str, upstream_sdk))
            + ", but warp-optix was compiled with "
            + ".".join(map(str, extension_sdk))
            + ". Both must use OptiX 9.1.0; reinstall or rebuild with matching headers."
        )
    _extensions.validate_pyoptix_bindings(upstream)

    # PyOptiX 9.1.1 ignores module compile metadata. Repair it in these
    # wrappers instead of patching the shared `optix` module.
    def module_create(context, moduleCompileOptions, pipelineCompileOptions, *args):
        _extensions.sync_module_options(moduleCompileOptions)
        return context.moduleCreate(moduleCompileOptions, pipelineCompileOptions, *args)

    def builtin_is_module_get(context, moduleCompileOptions, pipelineCompileOptions, *args):
        _extensions.sync_module_options(moduleCompileOptions)
        return context.builtinISModuleGet(moduleCompileOptions, pipelineCompileOptions, *args)

    backend = ModuleType("warp_optix._backend_api")
    backend.__file__ = _extensions.__file__
    for module in (upstream, _extensions):
        for name in dir(module):
            if not name.startswith("_") and name not in (
                "validate_pyoptix_bindings",
                "sync_module_options",
                "set_internal_guide_layers",
                "set_temporal_mode_use_previous_layers",
                "denoiser_temporal_sizes",
                "optix_version",
            ):
                setattr(backend, name, getattr(module, name))
    backend.moduleCreate = module_create
    backend.builtinISModuleGet = builtin_is_module_get

    # Missing enum values must use the dependency's existing enum types so
    # they remain accepted by upstream constructors and field setters.
    for prefix, enum in (
        ("DEVICE_PROPERTY_COOP_VEC", upstream.DeviceProperty),
        ("PRIMITIVE_TYPE_FLAGS_", upstream.PmitiveTypeFlags),
        ("PRIMITIVE_TYPE_", upstream.PrimitiveType),
        ("PIXEL_FORMAT_", upstream.PixelFormat),
        ("DENOISER_MODEL_KIND_", upstream.DenoiserModelKind),
    ):
        for name in dir(_extensions):
            if prefix == "PRIMITIVE_TYPE_" and name.startswith("PRIMITIVE_TYPE_FLAGS_"):
                continue
            if name.startswith(prefix) and isinstance(getattr(_extensions, name), int):
                setattr(backend, name, enum(getattr(_extensions, name)))

    backend.get_optix_include_dir = _get_optix_include_dir
    return backend


try:
    _extensions = import_module("warp_optix._ext")
except ImportError as error:
    raise ImportError(
        "The warp-optix supplemental extension could not be loaded. "
        "Install a compatible wheel or build this repository with `pip install .`."
    ) from error

_backend = _load_backend()


def get_backend():
    return _backend
