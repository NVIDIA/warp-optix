# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named, backend-independent descriptions of shader binding table layouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .pipeline import HitGroup


def _validate_entry(entry):
    if entry is not None and not isinstance(entry, wp.Kernel):
        raise TypeError("Expected an ordinary Warp kernel or None")


def _entry_name(entry):
    return "(none)" if entry is None else entry.key


def _ray_type_index(ray_types: tuple[str, ...], name: str) -> int:
    try:
        return ray_types.index(name)
    except ValueError:
        raise KeyError(f"Unknown ray type {name!r}; expected one of {ray_types}") from None


@dataclass(frozen=True)
class HitGroupHandle:
    """A group in one builder's layout; resolve its offset with ``SbtLayout.get_sbt_offset()``.

    Offsets are in records, not bytes. Handles remain stable as groups are appended.
    """

    _owner: object = field(repr=False)
    name: str
    index: int
    sbt_offset: int


@dataclass(frozen=True)
class SbtLayout:
    """Immutable pipeline layout in group-major, ray-type-minor record order.

    ``ray_type_index(name)`` supplies both the trace SBT offset and miss index.
    ``trace_stride`` supplies the trace SBT stride. These are record counts,
    distinct from the native table's byte strides.
    """

    ray_types: tuple[str, ...]
    hit_groups: tuple[HitGroupHandle, ...]
    _miss_entries: tuple[wp.Kernel | None, ...] = field(repr=False)
    _hit_kernels: tuple[tuple[HitGroup, ...], ...] = field(repr=False)

    @property
    def trace_stride(self) -> int:
        return len(self.ray_types)

    def ray_type_index(self, name: str) -> int:
        return _ray_type_index(self.ray_types, name)

    def get_sbt_offset(self, handle: HitGroupHandle) -> int:
        if not isinstance(handle, HitGroupHandle) or not any(handle is h for h in self.hit_groups):
            raise ValueError("Hit-group handle does not belong to this SBT layout")
        return handle.sbt_offset

    def describe(self) -> str:
        """Return a printable table of miss and hit records and their shaders."""
        rows = [("section", "record", "group", "ray type", "programs")]
        for index, (ray, entry) in enumerate(zip(self.ray_types, self._miss_entries)):
            rows.append(("miss", str(index), "-", ray, _entry_name(entry)))
        for handle, kernels in zip(self.hit_groups, self._hit_kernels):
            for index, (ray, kernel) in enumerate(zip(self.ray_types, kernels)):
                programs = [
                    f"{label}={getattr(entry, 'key', entry)}"
                    for label, entry in (
                        ("CH", kernel.closest_hit),
                        ("AH", kernel.any_hit),
                        ("IS", kernel.intersection),
                        (
                            "builtin",
                            getattr(
                                kernel.builtin,
                                "name",
                                kernel.builtin,
                            ),
                        ),
                    )
                    if entry is not None
                ]
                rows.append(
                    (
                        "hit",
                        str(handle.sbt_offset + index),
                        handle.name,
                        ray,
                        ", ".join(programs),
                    )
                )
        widths = [max(len(row[i]) for row in rows) for i in range(5)]
        table = "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)
        return f"Trace stride: {self.trace_stride} records (ray index = trace offset = miss index)\n{table}"


class SbtBuilder:
    """Build an SBT description without allocating native or device resources.

    Create with ``SbtBuilder(ray_types=("radiance", "shadow"))``. Assign each
    miss slot explicitly (a kernel or None) and one HitGroup per ray type in
    each named group. Empty HitGroups support hit-object traversal. Pipeline creation takes a
    snapshot; subsequent edits only affect pipelines created afterwards.
    """

    def __init__(self, ray_types: Sequence[str] = ("radiance",)):
        if isinstance(ray_types, str):
            raise TypeError("ray_types must be a sequence of names, not a string")
        names = tuple(ray_types)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("ray_types must contain nonempty names")
        if len(set(names)) != len(names):
            raise ValueError("ray_types must be unique")
        # OptiX trace SBT offset and stride are four-bit values.
        if len(names) > 15:
            raise ValueError("At most 15 ray types are supported by the trace SBT stride")
        self._owner = object()
        self._ray_types = names
        self._miss = {}
        self._groups = []
        self._kernels = []

    @property
    def ray_types(self) -> tuple[str, ...]:
        return self._ray_types

    @property
    def trace_stride(self) -> int:
        return len(self.ray_types)

    def ray_type_index(self, name: str) -> int:
        return _ray_type_index(self.ray_types, name)

    def set_miss(self, ray_type: str, kernel: wp.Kernel | None) -> None:
        self.ray_type_index(ray_type)
        try:
            _validate_entry(kernel)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Miss program for ray type {ray_type!r}: {error}") from error
        self._miss[ray_type] = kernel

    def add_hit_group(self, name: str, /, **ray_kernels: HitGroup) -> HitGroupHandle:
        """Append one named material/geometry group, with a kernel per ray type."""
        from .pipeline import HitGroup

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Hit-group name must be nonempty")
        if any(handle.name == name for handle in self._groups):
            raise ValueError(f"Duplicate hit-group name {name!r}")
        missing = set(self.ray_types) - ray_kernels.keys()
        extra = ray_kernels.keys() - set(self.ray_types)
        if missing or extra:
            raise ValueError(
                f"Hit group {name!r}: missing ray types {sorted(missing)}, unknown ray types {sorted(extra)}"
            )
        kernels = []
        for ray in self.ray_types:
            kernel = ray_kernels[ray]
            label = f"Hit group {name!r}, ray type {ray!r}"
            if not isinstance(kernel, HitGroup):
                raise TypeError(f"{label}: expected a warp_optix.HitGroup")
            try:
                for entry in (kernel.closest_hit, kernel.any_hit, kernel.intersection):
                    _validate_entry(entry)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label}: {error}") from error
            kernels.append(kernel)
        geometry_types = {(k.intersection is not None, k.builtin) for k in kernels}
        if len(geometry_types) != 1:
            raise ValueError(
                f"Hit group {name!r}: all ray types must use the same geometry kind (triangle, custom, or built-in primitive type)"
            )
        if len(self._groups) * self.trace_stride > 0xFFFFFF:
            raise ValueError("Hit-group offset exceeds OptiX's 24-bit instance SBT offset")
        handle = HitGroupHandle(
            self._owner,
            name,
            len(self._groups),
            len(self._groups) * self.trace_stride,
        )
        self._groups.append(handle)
        self._kernels.append(tuple(kernels))
        return handle

    def snapshot(self) -> SbtLayout:
        """Validate and freeze the layout; later edits do not change this snapshot."""
        missing = [ray for ray in self.ray_types if ray not in self._miss]
        if missing:
            raise ValueError(f"Missing miss programs for ray types {missing}; call set_miss() for each ray type")
        return SbtLayout(
            self.ray_types,
            tuple(self._groups),
            tuple(self._miss[ray] for ray in self.ray_types),
            tuple(self._kernels),
        )

    def describe(self) -> str:
        """Validate and preview the layout that the next pipeline will use."""
        return self.snapshot().describe()
