# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn ordinary Warp kernels into OptiX programs.

OptiX programs take no arguments: they read a single launch-parameter block
from constant memory. Warp's ``external_constant_params`` entry ABI supports
exactly that shape, so a kernel ``k(a: A, b: B)`` is compiled as

    def k(_wo_params: Params):
        a = _wo_params.a
        b = _wo_params.b
        <body of k>

where ``Params`` is an explicit Warp struct or is derived from the raygen signature.
``wp.tid()`` is mapped onto the OptiX launch index, so the same kernel source
runs under ``wp.launch`` and under :func:`warp_optix.launch` with equal ``dim``.
Programs bind fields of the shared parameter schema by name.
"""

from __future__ import annotations

import ast
import enum
import inspect
import types

import warp as wp

PARAMS_ARG = "_wo_params"
_INDEX_VAR = "_wo_launch_index"


class ProgramKind(enum.Enum):
    """OptiX program kinds, valued by their mandatory entry-name prefix."""

    RAYGEN = "__raygen__"
    MISS = "__miss__"
    CLOSEST_HIT = "__closesthit__"
    ANY_HIT = "__anyhit__"
    INTERSECTION = "__intersection__"
    EXCEPTION = "__exception__"
    DIRECT_CALLABLE = "__direct_callable__"
    CONTINUATION_CALLABLE = "__continuation_callable__"


def kernel(func=None, *, kind: ProgramKind, **options):
    """Declare a Warp kernel as an OptiX program of ``kind``, like ``@wp.kernel``.

    Use as ``@wo.kernel(kind=wo.ProgramKind.MISS)``; ``options`` are forwarded to
    ``wp.kernel``. The result is an ordinary Warp kernel. Pipeline construction
    checks that it occupies a slot of the declared kind.
    """
    if not isinstance(kind, ProgramKind):
        raise TypeError("kind must be a ProgramKind")

    def decorate(func):
        if isinstance(func, wp.Kernel) and options:
            raise TypeError("Set Warp kernel options on @wp.kernel when decorating an existing kernel")
        result = func if isinstance(func, wp.Kernel) else wp.kernel(func, **options)
        result._optix_kind = kind
        return result

    return decorate if func is None else decorate(func)


def kernel_signature(kernel: wp.Kernel) -> dict[str, type]:
    """Return a kernel's argument names and Warp types in declaration order."""
    return {arg.label: arg.type for arg in kernel.adj.args}


def params_struct(kernel: wp.Kernel):
    """Return a Warp struct whose fields are ``kernel``'s arguments, in order."""
    signature = kernel_signature(kernel)
    name = f"{_identifier(kernel.key)}_params"
    cls = type(name, (), {"__annotations__": signature, "__module__": __name__, "__qualname__": name})
    return wp.struct(cls)


def as_program(kernel: wp.Kernel, kind: ProgramKind, params, module: str) -> wp.Kernel:
    """Return a copy of ``kernel`` compiled as an OptiX program of ``kind``.

    ``params`` is the launch-parameter struct shared by the pipeline. Every
    argument of ``kernel`` must be a field of ``params`` with the same type;
    generic arguments take the type of their field.
    """
    declared = getattr(kernel, "_optix_kind", None)
    if declared is not None and declared is not kind:
        raise TypeError(f"Kernel {kernel.key!r} declares {declared.name}, but occupies {kind.name}")
    fields = params.vars
    names = list(kernel_signature(kernel))
    for name in names:
        if name not in fields:
            raise TypeError(f"{kind.name.lower()} kernel '{kernel.key}' argument '{name}' is not a launch parameter")
    if kernel.is_generic:
        kernel = wp.overload(kernel, {name: fields[name].type for name in names})
    for name, arg_type in kernel_signature(kernel).items():
        if not wp.types.types_equal(fields[name].type, arg_type):
            raise TypeError(
                f"{kind.name.lower()} kernel '{kernel.key}' argument '{name}' type differs from launch parameters"
            )

    # Warp compiles the kernel's own source with its own globals and closure, so
    # error messages and line info point at the user's file. Only the signature
    # differs: one params struct, unpacked by name at the top of the body.
    func = kernel.func
    program = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    program.__module__ = func.__module__
    program.__qualname__ = func.__qualname__
    program.__annotations__ = {PARAMS_ARG: params}
    program.__signature__ = inspect.Signature(
        [inspect.Parameter(PARAMS_ARG, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=params)]
    )

    options = dict(kernel.options) | {"entry_point_abi": "external_constant_params", "enable_backward": False}
    return wp.Kernel(
        program,
        key=f"{kind.value}{_identifier(kernel.key)}",
        module=wp.get_module(module),
        options=options,
        code_transformers=[_ProgramTransformer(names, kind)],
    )


def _identifier(key: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in key)


def _is_tid_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    target = node.func
    return (isinstance(target, ast.Attribute) and target.attr == "tid") or (
        isinstance(target, ast.Name) and target.id == "tid"
    )


def _statement(source: str, location: ast.AST) -> ast.stmt:
    """Parse one statement and attribute it to the source line of ``location``."""
    statement = ast.parse(source).body[0]
    for node in ast.walk(statement):
        ast.copy_location(node, location)
    return statement


class _ProgramTransformer(ast.NodeTransformer):
    """Unpack the params struct into the kernel's argument names and map ``wp.tid()``."""

    def __init__(self, names: list[str], kind: ProgramKind):
        self.names = names
        self.kind = kind

    def visit_FunctionDef(self, node: ast.FunctionDef):
        _rewrite_tid(node, self.kind)
        node.body = [_statement(f"{name} = {PARAMS_ARG}.{name}", node) for name in self.names] + node.body
        return node


def _rewrite_tid(fn: ast.FunctionDef, kind: ProgramKind) -> None:
    """Replace ``a, b = wp.tid()`` with reads of the OptiX launch index.

    Warp's ``dim=(d0, d1, d2)`` maps to an OptiX launch of ``(d2, d1, d0)`` so
    that the fastest-varying Warp index is OptiX's ``x``.
    """

    class Rewriter(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            if not _is_tid_call(node.value):
                return self.generic_visit(node)
            if kind is not ProgramKind.RAYGEN:
                raise TypeError("wp.tid() is only available in raygen programs")
            if len(node.targets) != 1:
                raise TypeError("wp.tid() must be assigned to a single target")
            target = node.targets[0]
            names = target.elts if isinstance(target, ast.Tuple) else [target]
            if len(names) > 3 or not all(isinstance(name, ast.Name) for name in names):
                raise TypeError("OptiX launches support at most three wp.tid() dimensions")
            if not isinstance(node.value.func, ast.Attribute):
                raise TypeError("Call wp.tid() through the warp module in OptiX programs")
            # The module that provides tid() in the kernel's scope also provides the OptiX builtins.
            warp_module = ast.unparse(node.value.func.value)
            ndim = len(names)
            index = _statement(f"{_INDEX_VAR} = {warp_module}.optix_get_launch_index()", node)
            return [index] + [
                _statement(f"{name.id} = int({_INDEX_VAR}[{ndim - 1 - i}])", node) for i, name in enumerate(names)
            ]

        def visit_Call(self, node: ast.Call):
            if _is_tid_call(node):
                raise TypeError("wp.tid() must appear as the right-hand side of an assignment")
            return self.generic_visit(node)

    Rewriter().visit(fn)
