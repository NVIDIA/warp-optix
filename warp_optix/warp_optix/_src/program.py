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
import weakref
from typing import TypeGuard, cast

import warp as wp

PARAMS_ARG = "_wo_params"
_INDEX_VAR = "_wo_launch_index"
# Warp never frees a struct type, so derive one per kernel rather than per pipeline.
_derived_params: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
    ``wp.kernel``, with ``module="unique"`` unless another module is given, so
    that OptiX builtins do not break ``wp.launch`` of other kernels in the same
    Python module. The result is an ordinary Warp kernel. Pipeline construction
    checks that it occupies a slot of the declared kind. An existing kernel keeps
    its module, which therefore must not be its Python module's default one.
    """
    if not isinstance(kind, ProgramKind):
        raise TypeError("kind must be a ProgramKind")

    def decorate(func):
        if isinstance(func, wp.Kernel):
            if options:
                raise TypeError("Set Warp kernel options on @wp.kernel when decorating an existing kernel")
            if func.module.name == func.func.__module__:
                # Its OptiX builtins would break wp.launch of every other kernel in that module.
                raise TypeError(
                    f"{func.key} is in its Python module's default Warp module; use @wo.kernel without "
                    "@wp.kernel, or @wp.kernel(module='unique')"
                )
        result = func if isinstance(func, wp.Kernel) else wp.kernel(func, **({"module": "unique"} | options))
        result._optix_kind = kind
        return result

    return decorate if func is None else decorate(func)


def kernel_signature(kernel: wp.Kernel) -> dict[str, type]:
    """Return a kernel's argument names and Warp types in declaration order."""
    return {arg.label: arg.type for arg in kernel.adj.args}


def params_struct(kernel: wp.Kernel):
    """Return a Warp struct whose fields are ``kernel``'s arguments, in order."""
    struct = _derived_params.get(kernel)
    if struct is None:
        signature = kernel_signature(kernel)
        name = f"{_identifier(kernel.key)}_params"
        cls = type(name, (), {"__annotations__": signature, "__module__": __name__, "__qualname__": name})
        struct = _derived_params[kernel] = wp.struct(cls)
    return struct


def as_program(kernel: wp.Kernel, kind: ProgramKind, params, module: wp.Module) -> wp.Kernel:
    """Return a copy of ``kernel`` compiled as an OptiX program of ``kind``.

    ``params`` is the launch-parameter struct shared by the pipeline. Every
    argument of ``kernel`` must be a field of ``params`` with the same type;
    generic arguments take the type of their field.
    """
    declared = getattr(kernel, "_optix_kind", None)
    if declared is not None and declared is not kind:
        raise TypeError(f"Kernel {kernel.key!r} declares {declared.name}, but occupies {kind.name}")
    fields = params.vars
    source_adj = kernel.adj
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
    tid_ndim = []
    # Warp applies code transformers by calling their visit(tree).
    transformer = types.SimpleNamespace(visit=lambda tree: _program_tree(tree, names, kind, tid_ndim))
    result = wp.Kernel(
        program,
        key=f"{kind.value}{_identifier(kernel.key)}",
        module=module,
        options=options,
        source=source_adj.source,
        code_transformers=[*source_adj.transformers, transformer],
    )
    # source= also supports generated kernels whose Python function has no
    # inspectable source. Warp starts explicit source at line zero; restore the
    # original location for ordinary kernels and their generic overloads.
    result.adj.filename = source_adj.filename
    result.adj.fun_lineno = source_adj.fun_lineno
    result._wo_tid_ndim = tid_ndim[-1]
    return result


def _identifier(key: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in key)


def _tid_module(node: ast.AST) -> str | None:
    """Return ``m`` if ``node`` is a ``m.tid()`` call; Warp builtins are attributes of the warp module."""
    if (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tid"
    ):
        return ast.unparse(node.func.value)
    return None


def _statement(source: str, location: ast.AST) -> ast.stmt:
    """Parse one statement and attribute it to the source line of ``location``."""
    statement = ast.parse(source).body[0]
    for node in ast.walk(statement):
        ast.copy_location(node, location)
    return statement


def _program_tree(tree: ast.Module, names: list[str], kind: ProgramKind, tid_ndim: list[int]) -> ast.Module:
    """Unpack the params struct into the kernel's argument names and map ``wp.tid()``.

    ``a, b = wp.tid()`` becomes reads of the OptiX launch index. Warp's ``dim=(d0, d1, d2)``
    maps to an OptiX launch of ``(d2, d1, d0)``, so the fastest-varying Warp index is OptiX's
    ``x``; this matches ``wp.tid()`` when the launch has as many dimensions as indices are
    unpacked. The number of unpacked indices is appended to ``tid_ndim``.
    """
    fn = cast(ast.FunctionDef, tree.body[0])
    rewritten = set()
    ndims = set()
    # ast.walk queues a node's children before yielding it, so replacing statement lists is safe.
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if isinstance(statements, list) and any(_is_tid_assignment(stmt) for stmt in statements):
                expanded = []
                for stmt in statements:
                    if _is_tid_assignment(stmt):
                        rewritten.add(stmt.value)
                        expanded += _launch_index(stmt, kind, ndims)
                    else:
                        expanded.append(stmt)
                setattr(node, field, expanded)
    if len(ndims) > 1:
        raise TypeError("every wp.tid() call in a raygen program must unpack the same number of indices")
    if any(_tid_module(node) is not None and node not in rewritten for node in ast.walk(fn)):
        raise TypeError("wp.tid() must appear as the right-hand side of an assignment")
    fn.body = [_statement(f"{name} = {PARAMS_ARG}.{name}", fn) for name in names] + fn.body
    tid_ndim.append(ndims.pop() if ndims else 0)
    return tree


def _is_tid_assignment(stmt: ast.stmt) -> TypeGuard[ast.Assign]:
    return isinstance(stmt, ast.Assign) and _tid_module(stmt.value) is not None


def _launch_index(node: ast.Assign, kind: ProgramKind, ndims: set) -> list[ast.stmt]:
    """Statements reading the OptiX launch index into the targets of ``... = wp.tid()``."""
    if kind is not ProgramKind.RAYGEN:
        raise TypeError("wp.tid() is only available in raygen programs")
    if len(node.targets) != 1:
        raise TypeError("wp.tid() must be assigned to a single target")
    target = node.targets[0]
    names = target.elts if isinstance(target, ast.Tuple) else [target]
    ids = [name.id for name in names if isinstance(name, ast.Name)]
    if len(names) > 3 or len(ids) != len(names):
        raise TypeError("OptiX launches support at most three wp.tid() dimensions")
    # The module that provides tid() in the kernel's scope also provides the OptiX builtins.
    warp_module = _tid_module(node.value)
    ndims.add(len(ids))
    index = _statement(f"{_INDEX_VAR} = {warp_module}.optix_get_launch_index()", node)
    return [index] + [_statement(f"{name} = int({_INDEX_VAR}[{len(ids) - 1 - i}])", node) for i, name in enumerate(ids)]
