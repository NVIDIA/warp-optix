# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The path tracer is a client of the public warp_optix API."""

import ast
from pathlib import Path

import warp_optix as wo
import warp_optix_pathtracing


def _core_uses(tree):
    """Yield (line, name) for every warp_optix module or attribute the tree reaches."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "warp_optix" or alias.name.startswith("warp_optix."):
                    yield node.lineno, alias.name
                    if alias.name == "warp_optix":
                        aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "warp_optix":
                for alias in node.names:
                    yield node.lineno, f"warp_optix.{alias.name}"
            elif node.module.startswith("warp_optix."):
                yield node.lineno, node.module
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            yield node.lineno, f"warp_optix.{node.attr}"


def test_path_tracer_uses_only_public_warp_optix_names():
    public = {"warp_optix"} | {f"warp_optix.{name}" for name in wo.__all__}
    root = Path(warp_optix_pathtracing.__file__).parent
    violations = []
    for path in sorted(root.rglob("*.py")):
        for line, name in _core_uses(ast.parse(path.read_text(encoding="utf-8"))):
            if name not in public:
                violations.append(f"{path.relative_to(root)}:{line}: {name}")
    assert violations == []


def test_boundary_check_flags_private_access():
    source = """
import warp_optix as wo
import warp_optix.context
from warp_optix import native, Accel
from warp_optix.context import resolve
wo._backend
wo.Pipeline
"""
    public = {"warp_optix"} | {f"warp_optix.{name}" for name in wo.__all__}
    flagged = sorted({name for _, name in _core_uses(ast.parse(source)) if name not in public})
    assert flagged == ["warp_optix._backend", "warp_optix.context", "warp_optix.native"]
