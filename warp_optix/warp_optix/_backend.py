# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose PyOptiX with the small extension shipped by warp-optix.

PyOptiX owns every standard OptiX object and its destruction. The extension
adds missing operations to those types; it never registers a second copy.
"""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import os
import sys
import sysconfig
from types import ModuleType


_package_dir = Path(__file__).resolve().parent
_dll_directory_handles = []
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for path in (
        _package_dir,
        Path(sysconfig.get_path("platlib")) / "warp_optix",
        Path(os.environ.get("CUDA_BIN_DIR", "/__missing_cuda_bin__")),
        Path(os.environ.get("CUDA_PATH", "/__missing_cuda_path__")) / "bin",
    ):
        if path.is_dir():
            _dll_directory_handles.append(os.add_dll_directory(str(path)))


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
    raise FileNotFoundError(
        "Packaged OptiX SDK headers are missing or incomplete; reinstall warp-optix."
    )


def _dlss_support_available():
    return hasattr(_extensions, "DlssRRContext")


def _dlss_rr_available(project_id):
    if not _dlss_support_available():
        return False
    context = _extensions.DlssRRContext()
    try:
        context.init(project_id, featureSearchPath=str(_get_package_dir()))
        return context.isDlssRRAvailable()
    except RuntimeError:
        return False
    finally:
        context.deinit()


def _load_backend():
    if version("pyoptix") != "9.1.0":
        raise ImportError("warp-optix requires pyoptix==9.1.0; reinstall its dependencies")
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
    _extensions.install_pyoptix_extensions(upstream)

    # These two upstream defects can be repaired while retaining upstream's
    # real objects, compiler, program groups, and ownership conventions.
    if not getattr(upstream, "_warp_optix_extensions_installed", False):
        for name in ("moduleCreate", "builtinISModuleGet"):
            original = getattr(upstream.DeviceContext, name)

            def compile_module(self, moduleCompileOptions, pipelineCompileOptions, *args,
                               _original=original, **kwargs):
                _extensions.sync_module_options(moduleCompileOptions)
                return _original(self, moduleCompileOptions, pipelineCompileOptions, *args, **kwargs)

            setattr(upstream.DeviceContext, name, compile_module)

        original_get_property = upstream.DeviceContext.getProperty

        def get_property(self, property):
            if int(property) == int(_extensions.DEVICE_PROPERTY_COOP_VEC):
                return _extensions.coop_vec_property(self)
            return original_get_property(self, property)

        upstream.DeviceContext.getProperty = get_property
        original_descriptor_init = upstream.ProgramGroupDesc.__init__

        def init_descriptor(self, *args, **kwargs):
            original_descriptor_init(self, *args, **kwargs)
            continuation = kwargs.get("callablesModuleCC", args[10] if len(args) > 10 else None)
            if continuation is not None:
                self.callablesModuleCC = continuation

        upstream.ProgramGroupDesc.__init__ = init_descriptor
        upstream._warp_optix_extensions_installed = True

    backend = ModuleType("warp_optix._backend_api")
    # The renderer uses __file__ to find the packaged DLSS runtime.
    backend.__file__ = _extensions.__file__
    for module in (upstream, _extensions):
        for name in dir(module):
            if not name.startswith("_") and name not in (
                "install_pyoptix_extensions", "sync_module_options", "coop_vec_property", "optix_version"
            ):
                setattr(backend, name, getattr(module, name))

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
                value = enum(getattr(_extensions, name))
                setattr(backend, name, value)
                if not hasattr(enum, name):
                    setattr(enum, name, value)

    if not _dlss_support_available():
        stub = import_module("warp_optix._dlss_stub")
        for name in stub.__all__:
            setattr(backend, name, getattr(stub, name))
        for enum in (stub.DlssPerfQuality, stub.RayReconstructionHintRenderPreset, stub.DlssRRResource):
            for member in enum:
                setattr(backend, member.name, member)
    backend.get_optix_include_dir = _get_optix_include_dir
    backend.dlss_support_available = _dlss_support_available
    backend.dlss_rr_available = _dlss_rr_available
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
