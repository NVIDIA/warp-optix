# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Warp releases without addon hooks."""

from __future__ import annotations

import ctypes
import inspect
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

_CODEGEN_LOCK = threading.RLock()
_CPP_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_MODULE_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def validate_module_tag(module_tag: str) -> str:
    """Reject module tags that could escape the OptiX kernel cache directory."""
    if not isinstance(module_tag, str) or not _MODULE_TAG_RE.match(module_tag):
        raise ValueError(
            "module_tag must be 1-128 characters of letters, digits, '_', '.' "
            f"or '-' and must not start with '.' or '-', got {module_tag!r}"
        )
    return module_tag


def _find_cuda_include_dir() -> Path | None:
    """Find the CUDA Toolkit headers required by some OptiX headers."""
    roots = []
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        if value := os.environ.get(variable):
            roots.append(Path(value))

    if nvcc := shutil.which("nvcc"):
        roots.append(Path(nvcc).resolve().parent.parent)

    roots.append(Path("/usr/local/cuda"))
    candidates = []
    for root in roots:
        candidates.append(root / "include")
        candidates.extend(sorted(root.glob("targets/*/include")))

    return next(
        (path for path in candidates if (path / "cuda_fp16.h").is_file()),
        None,
    )


def has_public_addon_hooks(wp) -> bool:
    """Return whether Warp exposes the addon hooks used by warp_optix."""
    kernel_parameters = inspect.signature(wp.kernel).parameters
    build_api = _get_build_api(wp)
    return (
        build_api is not None
        and hasattr(build_api, "add_builtin")
        and hasattr(wp, "ModuleBuildOptions")
        and hasattr(wp, "compile_aot_module")
        and "name" in kernel_parameters
        and "entry_point_abi" in kernel_parameters
    )


def _get_build_api(wp):
    """Return Warp's supported builtin-registration namespace, when present."""
    return getattr(wp, "build_experimental", None) or getattr(wp, "build", None)


def get_add_builtin(wp):
    """Return Warp's addon builtin registrar or a private-API adapter."""
    build_api = _get_build_api(wp)
    if build_api is not None and hasattr(build_api, "add_builtin"):
        return build_api.add_builtin

    from warp._src.context import add_builtin as private_add_builtin

    def add_builtin(
        name: str,
        input_types=None,
        value_type=None,
        *,
        native_name: str | None = None,
        doc: str = "",
    ):
        native_name = native_name or f"wp::{name}"
        namespace, separator, native_func = native_name.rpartition("::")
        if not separator:
            namespace = ""
            native_func = native_name
        elif namespace:
            namespace += "::"
        return private_add_builtin(
            name,
            input_types=dict(input_types or {}),
            value_type=value_type,
            doc=doc,
            namespace=namespace,
            native_func=native_func,
            export=False,
            hidden=True,
            is_differentiable=False,
        )

    return add_builtin


def create_external_kernel(
    wp,
    fn: Callable[..., Any],
    *,
    name: str,
    entry_point_abi: str,
    kernel_kwargs: dict[str, Any],
):
    """Create a named external-entry kernel using vanilla Warp internals."""
    if not _CPP_IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"OptiX kernel name must be a C++ identifier, got {name!r}")
    if entry_point_abi != "external_constant_params":
        raise ValueError(f"Unsupported external entry-point ABI: {entry_point_abi!r}")

    kwargs = dict(kernel_kwargs)
    module = kwargs.pop("module", None)
    module_options = kwargs.pop("module_options", None)

    if kwargs.get("enable_backward") is not False:
        raise ValueError("external_constant_params requires enable_backward=False")
    if module_options is not None and module != "unique":
        raise ValueError('module_options requires module="unique"')

    is_unique = module == "unique"
    if is_unique:
        from warp._src.context import Module

        module = Module(fn.__name__, None)
        if module_options:
            unknown = sorted(set(module_options) - set(module.options))
            if unknown:
                raise ValueError(
                    f"unknown module_options: {', '.join(repr(key) for key in unknown)}"
                )
            module.options.update(module_options)

    kernel = wp.kernel(module=module, **kwargs)(fn)
    old_key = kernel.key
    kernel.key = name
    kernel.options["entry_point_abi"] = entry_point_abi
    kernel.options["enable_backward"] = False

    if kernel.module.kernels.get(old_key) is kernel:
        del kernel.module.kernels[old_key]
    kernel.module.kernels[kernel.key] = kernel
    kernel.module.mark_modified()

    if is_unique:
        kernel.is_unique_module = True
        module_hash = kernel.module.get_module_hash()
        kernel.module.name = f"{kernel.key}_{module_hash.hex()[:8]}"

    return kernel


def _external_codegen_kernel(kernel, device, options):
    from warp._src import codegen

    if kernel.options.get("entry_point_abi") != "external_constant_params":
        return _external_codegen_kernel.original(kernel, device, options)
    options = options | kernel.options
    if device != "cuda":
        raise RuntimeError("external_constant_params kernels are CUDA-only")
    if options.get("enable_backward", False):
        raise RuntimeError(
            "external_constant_params kernels do not support backward code generation"
        )
    if len(kernel.adj.args) != 1 or not isinstance(
        kernel.adj.args[0].type, codegen.Struct
    ):
        raise RuntimeError(
            f"OptiX kernel '{kernel.key}' must take exactly one Warp struct launch-params argument"
        )
    if kernel.adj.get_total_required_shared():
        raise RuntimeError(
            f"OptiX kernel '{kernel.key}' cannot use shared-memory tiles"
        )

    arg = kernel.adj.args[0]
    body = codegen.codegen_func_forward(kernel.adj, func_type="function", device="cuda")
    return (
        f'\nextern "C" __global__ void {kernel.get_mangled_name()}()\n{{\n'
        f"    {arg.ctype()} var_{arg.label} = "
        f"*reinterpret_cast<const {arg.ctype()}*>(params);\n"
        f"{body}}}\n"
    )


_external_codegen_kernel.original = None


def _generate_cuda_source(wp, module, launch_preamble: str):
    from warp._src import codegen
    from warp._src.context import ModuleBuilder

    options = module.resolve_options(wp.config)
    original_codegen_kernel = codegen.codegen_kernel
    _external_codegen_kernel.original = original_codegen_kernel
    try:
        codegen.codegen_kernel = _external_codegen_kernel
        builder = ModuleBuilder(module, options)
        source = builder.codegen("cuda")
    finally:
        codegen.codegen_kernel = original_codegen_kernel
        _external_codegen_kernel.original = None

    external_kernels = [
        kernel
        for kernel in builder.kernels
        if kernel.options.get("entry_point_abi") == "external_constant_params"
    ]
    if not external_kernels:
        raise RuntimeError(f"Warp module '{module.name}' contains no OptiX kernels")

    params_types = {kernel.adj.args[0].ctype() for kernel in external_kernels}
    if len(params_types) != 1:
        raise RuntimeError(
            "All OptiX kernels in one Warp module must use the same launch-params struct type"
        )
    params_ctype = params_types.pop()

    entry_marker = 'extern "C" __global__ void '
    entry_offset = min(
        source.index(entry_marker + kernel.get_mangled_name())
        for kernel in external_kernels
    )
    params_decl = (
        'extern "C" {\n'
        f"__constant__ __align__(alignof({params_ctype})) unsigned char params[sizeof({params_ctype})];\n"
        "}\n\n"
    )
    preamble = (
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "#define WP_OPTIX_PROGRAM 1\n"
    )
    if launch_preamble:
        preamble += launch_preamble
        if not preamble.endswith("\n"):
            preamble += "\n"
    preamble += '#include "warp_optix_builtins.h"\n'
    source = preamble + source[:entry_offset] + params_decl + source[entry_offset:]
    return source, builder


def _compile_cuda_source_to_ptx(
    wp, source: str, builder, output_path: Path, device
) -> None:
    from warp._src import context

    if wp.config.llvm_cuda:
        raise RuntimeError(
            "Vanilla Warp OptiX compatibility requires Warp's NVRTC backend"
        )

    include_dirs = [
        str(Path(__file__).resolve().parent / "_native" / "include"),
    ]
    try:
        from warp_optix._backend import get_backend
        optix = get_backend()

        include_dirs.append(optix.get_optix_include_dir())
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "The installed optix package does not expose its OptiX include directory"
        ) from error

    if cuda_include_dir := _find_cuda_include_dir():
        include_dirs.append(str(cuda_include_dir))

    encoded_include_dirs = [os.fsencode(path) for path in include_dirs]
    include_array = (ctypes.c_char_p * len(encoded_include_dirs))(*encoded_include_dirs)
    ltoirs = list(builder.ltoirs.values())
    fatbins = list(builder.fatbins.values())
    link_data = ltoirs + fatbins
    link_array = (ctypes.c_char_p * len(link_data))(*link_data)
    link_sizes = (ctypes.c_size_t * len(link_data))(*[len(item) for item in link_data])
    link_types = (ctypes.c_int * len(link_data))(
        *([3] * len(ltoirs) + [4] * len(fatbins))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arch = device.arch
    arch_suffix = device._get_cuda_arch_suffix(arch)
    warp_native = os.fsencode(Path(wp.__file__).resolve().parent / "native")
    compile_program = context.runtime.core.wp_cuda_compile_program
    if len(compile_program.argtypes or ()) != 22:
        raise RuntimeError(
            f"Unsupported vanilla Warp {wp.__version__}: private CUDA compiler signature changed"
        )

    mode = builder.options.get("mode") or wp.config.mode
    optimization_level = builder.options.get("optimization_level")
    if optimization_level is None:
        optimization_level = 0 if mode == "debug" else 3

    error = compile_program(
        source.encode("utf-8"),
        f"warp_optix_{output_path.stem}.cu".encode(),
        arch,
        arch_suffix.encode(),
        warp_native,
        len(encoded_include_dirs),
        include_array,
        mode == "debug",
        optimization_level,
        bool(wp.config.verbose),
        builder.options.get("verify_fp", False),
        builder.options["fast_math"],
        builder.options["fuse_fp"],
        builder.options["lineinfo"],
        builder.options["compile_time_trace"],
        False,
        os.fsencode(output_path),
        None,
        len(link_data),
        link_array,
        link_sizes,
        link_types,
    )
    if error:
        raise RuntimeError(
            f"Vanilla Warp NVRTC compilation failed with error code {error}"
        )


def compile_module_to_ptx(
    wp, module, launch_preamble: str, module_tag: str, device: str
) -> bytes:
    """Compile an OptiX module through vanilla Warp's private codegen/compiler APIs."""
    validate_module_tag(module_tag)
    with _CODEGEN_LOCK:
        cuda_device = wp.get_device(device)
        if not cuda_device.is_cuda:
            raise RuntimeError(
                f"OptiX compilation requires a CUDA device, got {device!r}"
            )
        source, builder = _generate_cuda_source(wp, module, launch_preamble)
        module_dir = Path(wp.config.kernel_cache_dir) / "optix" / module_tag
        output_path = module_dir / f"{module_tag}.sm{cuda_device.arch}.ptx"
        _compile_cuda_source_to_ptx(wp, source, builder, output_path, cuda_device)
        return output_path.read_bytes()
