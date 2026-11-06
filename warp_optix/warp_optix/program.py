# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn ordinary Warp kernels into OptiX programs.

OptiX programs take no arguments: they read a single launch-parameter block
from constant memory. Warp's ``external_constant_params`` entry ABI supports
exactly that shape, so a kernel ``k(a: A, b: B)`` is rewritten into

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
import itertools

import warp as wp

PARAMS_ARG = "_wo_params"
_INDEX_VAR = "_wo_launch_index"
_PARAMS_IDS = itertools.count()


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
    name = f"{_identifier(kernel.key)}_params_{next(_PARAMS_IDS)}"
    cls = type(name, (), {"__annotations__": signature, "__module__": __name__, "__qualname__": name})
    return wp.struct(cls)


def as_program(kernel: wp.Kernel, kind: ProgramKind, params, module: str) -> wp.Kernel:
    """Return a copy of ``kernel`` compiled as an OptiX program of ``kind``.

    ``params`` is the launch-parameter struct shared by the pipeline. Every
    argument of ``kernel`` must be a field of ``params`` with the same type.
    """
    declared = getattr(kernel, "_optix_kind", None)
    if declared is not None and declared is not kind:
        raise TypeError(f"Kernel {kernel.key!r} declares {declared.name}, but occupies {kind.name}")
    fields = params.vars
    for name, arg_type in kernel_signature(kernel).items():
        if name not in fields:
            raise TypeError(f"{kind.name.lower()} kernel '{kernel.key}' argument '{name}' is not a launch parameter")
        if not wp.types.types_equal(fields[name].type, arg_type):
            raise TypeError(
                f"{kind.name.lower()} kernel '{kernel.key}' argument '{name}' type differs from launch parameters"
            )

    tree = ast.parse(kernel.adj.source)
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise TypeError(f"kernel '{kernel.key}' source is not a function definition")
    tid_ndim = _rewrite_tid(fn, kind)
    bindings = [ast.parse(f"{name} = {PARAMS_ARG}.{name}").body[0] for name in kernel_signature(kernel)]
    index = [ast.parse(f"{_INDEX_VAR} = wp.optix_get_launch_index()").body[0]] if tid_ndim else []
    fn.body = bindings + index + fn.body
    fn.decorator_list = []
    fn.args = ast.arguments(posonlyargs=[], args=[ast.arg(PARAMS_ARG)], kwonlyargs=[], kw_defaults=[], defaults=[])
    fn.returns = None
    source = ast.unparse(ast.fix_missing_locations(tree))

    # The rewritten function resolves names exactly like the original kernel:
    # its globals are the original module globals plus the closure's cells.
    func = kernel.func
    scope = dict(func.__globals__)
    scope |= {n: c.cell_contents for n, c in zip(func.__code__.co_freevars, func.__closure__ or ())}
    scope.setdefault("wp", wp)
    namespace: dict = {}
    exec(compile(source, f"<warp_optix {kernel.key}>", "exec"), scope, namespace)  # noqa: S102
    program = namespace[fn.name]
    program.__annotations__ = {PARAMS_ARG: params}
    program.__module__ = func.__module__
    program.__qualname__ = fn.name

    options = dict(kernel.options) | {"entry_point_abi": "external_constant_params", "enable_backward": False}
    result = wp.Kernel(
        program,
        key=f"{kind.value}{_identifier(kernel.key)}",
        module=wp.get_module(module),
        options=options,
        source=source,
    )
    result._wo_tid_ndim = tid_ndim
    return result


def _identifier(key: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in key)


def _is_tid_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    target = node.func
    return (isinstance(target, ast.Attribute) and target.attr == "tid") or (
        isinstance(target, ast.Name) and target.id == "tid"
    )


def _rewrite_tid(fn: ast.FunctionDef, kind: ProgramKind) -> int:
    """Replace ``a, b = wp.tid()`` with reads of the OptiX launch index; return the index count (0 if unused).

    Warp's ``dim=(d0, d1, d2)`` maps to an OptiX launch of ``(d2, d1, d0)`` so
    that the fastest-varying Warp index is OptiX's ``x``. This matches
    ``wp.tid()`` only when the launch has as many dimensions as indices are unpacked.
    """
    tid_ndim = 0

    class Rewriter(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            nonlocal tid_ndim
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
            ndim = len(names)
            if tid_ndim and tid_ndim != ndim:
                raise TypeError("every wp.tid() call in a raygen program must unpack the same number of indices")
            tid_ndim = ndim
            return [ast.parse(f"{name.id} = int({_INDEX_VAR}[{ndim - 1 - i}])").body[0] for i, name in enumerate(names)]

        def visit_Call(self, node: ast.Call):
            if _is_tid_call(node):
                raise TypeError("wp.tid() must appear as the right-hand side of an assignment")
            return self.generic_visit(node)

    Rewriter().visit(fn)
    return tid_ndim
