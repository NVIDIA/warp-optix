# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent description of a shader binding table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import warp as wp

from warp_optix.enums import BuildFlags, PrimitiveType

_CURVE_TYPES = frozenset(int(kind) for kind in PrimitiveType if kind.name.startswith(("ROUND_", "FLAT_")))


@dataclass(frozen=True)
class HitGroup:
    """Programs run for one geometry type and ray type.

    ``builtin`` selects an OptiX built-in intersection (a :class:`PrimitiveType`)
    for curves. ``build_flags`` must match the flags used to build those curves.
    A hit group without programs still occupies its SBT record, which
    is what pipelines that only use ``optix_traverse`` hit objects need.
    ``data`` is a Warp struct instance stored in the record, read on the device
    with ``wp.optix_load_sbt_data()``.
    """

    closest_hit: wp.Kernel | None = None
    any_hit: wp.Kernel | None = None
    intersection: wp.Kernel | None = None
    builtin: int | None = None
    build_flags: int = BuildFlags.PREFER_FAST_TRACE
    data: object | None = None

    def __post_init__(self):
        if self.builtin is None:
            return
        if self.intersection is not None:
            raise ValueError("HitGroup intersection and builtin cannot be combined")
        if int(self.builtin) not in _CURVE_TYPES:
            raise ValueError(f"HitGroup.builtin must be a curve PrimitiveType, got {self.builtin!r}")


def _ray_type_index(ray_types: Sequence[str], name: str) -> int:
    try:
        return ray_types.index(name)
    except ValueError:
        raise KeyError(f"Unknown ray type {name!r}; expected one of {ray_types}") from None


@dataclass(frozen=True)
class Record:
    """A miss or callable program with the Warp struct instance stored in its SBT record."""

    program: wp.Kernel
    data: object | None = None

    def __post_init__(self):
        if not isinstance(self.program, wp.Kernel):
            raise TypeError("Record program must be a Warp kernel")


@dataclass(frozen=True)
class SbtLayout:
    """Shader binding table of a pipeline: named ray types, miss records, and hit groups.

    ``miss`` maps every ray type to a miss program, a :class:`Record`, or None
    (a record without a program). ``hit_groups`` maps each group name to a
    :class:`HitGroup` per ray type. A bare miss entry or ``HitGroup`` applies to
    every ray type. The defaults are miss records without programs and one
    empty hit group named ``"default"``.

    Records are group-major and ray-type-minor: ``get_sbt_offset(name)`` is a
    group's instance SBT offset, ``ray_type_index(name)`` is both the trace SBT
    offset and the miss index, and ``trace_stride`` is the trace SBT stride.
    These count records, not bytes. After construction, ``miss`` and
    ``hit_groups`` hold the validated, read-only mappings.
    """

    ray_types: Sequence[str] = ("radiance",)
    miss: Mapping[str, wp.Kernel | Record | None] | wp.Kernel | Record | None = None
    hit_groups: Mapping[str, Mapping[str, HitGroup] | HitGroup] = field(default_factory=lambda: {"default": HitGroup()})

    def __post_init__(self):
        if isinstance(self.ray_types, str):
            raise TypeError("ray_types must be a sequence of names, not a string")
        rays = tuple(self.ray_types)
        if not rays or any(not isinstance(name, str) or not name.strip() for name in rays):
            raise ValueError("ray_types must contain nonempty names")
        if len(set(rays)) != len(rays):
            raise ValueError("ray_types must be unique")
        # OptiX trace SBT offset and stride are four-bit values.
        if len(rays) > 15:
            raise ValueError("At most 15 ray types are supported by the trace SBT stride")
        object.__setattr__(self, "ray_types", rays)

        miss = _per_ray(self.miss, rays, "Miss records", lambda entry: not isinstance(entry, Mapping))
        for ray, entry in miss.items():
            program = entry.program if isinstance(entry, Record) else entry
            if program is not None and not isinstance(program, wp.Kernel):
                raise TypeError(f"Miss program for ray type {ray!r}: expected a Warp kernel, Record, or None")
        object.__setattr__(self, "miss", miss)

        if not isinstance(self.hit_groups, Mapping):
            raise TypeError("hit_groups must map group names to hit groups")
        groups = {}
        for name, entry in self.hit_groups.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Hit-group names must be nonempty strings")
            label = f"Hit group {name!r}"
            per_ray = _per_ray(entry, rays, label, lambda entry: isinstance(entry, HitGroup))
            for ray, group in per_ray.items():
                _validate_hit_group(group, f"{label}, ray type {ray!r}")
            if len({(group.intersection is not None, group.builtin) for group in per_ray.values()}) != 1:
                raise ValueError(
                    f"{label}: all ray types must use the same geometry kind (triangle, custom, or built-in primitive type)"
                )
            groups[name] = per_ray
        if (len(groups) - 1) * len(rays) > 0xFFFFFF:
            raise ValueError("Hit-group offset exceeds OptiX's 24-bit instance SBT offset")
        object.__setattr__(self, "hit_groups", MappingProxyType(groups))

    @property
    def trace_stride(self) -> int:
        return len(self.ray_types)

    def ray_type_index(self, name: str) -> int:
        return _ray_type_index(self.ray_types, name)

    def get_sbt_offset(self, name: str) -> int:
        """Return the instance SBT offset, in records, of hit group ``name``."""
        try:
            return list(self.hit_groups).index(name) * self.trace_stride
        except ValueError:
            raise KeyError(f"Unknown hit group {name!r}; expected one of {tuple(self.hit_groups)}") from None

    def describe(self) -> str:
        """Return a printable table of miss and hit records and their shaders."""
        rows = [("section", "record", "group", "ray type", "programs")]
        # __post_init__ normalizes the constructor's shorthand to mappings.
        miss = cast(Mapping[str, wp.Kernel | Record | None], self.miss)
        hit_groups = cast(Mapping[str, Mapping[str, HitGroup]], self.hit_groups)
        for index, (ray, entry) in enumerate(miss.items()):
            program = entry.program if isinstance(entry, Record) else entry
            rows.append(("miss", str(index), "-", ray, "(none)" if program is None else program.key))
        for group_index, (name, per_ray) in enumerate(hit_groups.items()):
            for index, (ray, group) in enumerate(per_ray.items()):
                programs = [
                    f"{label}={getattr(entry, 'key', entry)}"
                    for label, entry in (
                        ("CH", group.closest_hit),
                        ("AH", group.any_hit),
                        ("IS", group.intersection),
                        ("builtin", getattr(group.builtin, "name", group.builtin)),
                    )
                    if entry is not None
                ]
                record = group_index * self.trace_stride + index
                rows.append(("hit", str(record), name, ray, ", ".join(programs)))
        widths = [max(len(row[i]) for row in rows) for i in range(5)]
        table = "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)
        return f"Trace stride: {self.trace_stride} records (ray index = trace offset = miss index)\n{table}"


def _per_ray(entry, rays: tuple[str, ...], label: str, is_bare) -> MappingProxyType:
    """Expand a bare entry to every ray type, or check that a mapping names each ray type."""
    if is_bare(entry):
        return MappingProxyType(dict.fromkeys(rays, entry))
    if not isinstance(entry, Mapping):
        raise TypeError(f"{label}: expected a HitGroup or a mapping from ray type to HitGroup")
    missing = [ray for ray in rays if ray not in entry]
    extra = [ray for ray in entry if ray not in rays]
    if missing or extra:
        raise ValueError(f"{label}: missing ray types {missing}, unknown ray types {extra}")
    return MappingProxyType({ray: entry[ray] for ray in rays})


def _validate_hit_group(group, label: str) -> None:
    if not isinstance(group, HitGroup):
        raise TypeError(f"{label}: expected a warp_optix.HitGroup")
    if group.intersection is not None and group.builtin is not None:
        raise ValueError(f"{label}: custom and built-in intersection programs cannot be combined")
    for program in (group.closest_hit, group.any_hit, group.intersection):
        if program is not None and not isinstance(program, wp.Kernel):
            raise TypeError(f"{label}: programs must be Warp kernels or None")
