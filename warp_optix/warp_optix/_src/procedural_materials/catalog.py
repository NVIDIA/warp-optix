# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Searchable metadata and material ID resolution for the texture bakers."""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import TypedDict

import numpy as np
import warp as wp

from .bake import bake, bake_gltf, reference_ids, reference_maps
from .configurable_metals import _FINISHES as _METAL_FINISHES
from .configurable_metals import MetalSurface, metal_names
from .plastic_surfaces import _FINISHES as _PLASTIC_FINISHES
from .plastic_surfaces import PLASTIC_PRESETS, PlasticSurface

Material = str | MetalSurface | PlasticSurface


class _MaterialSummary(TypedDict):
    id: str
    label: str
    family: str


class _MaterialInfo(_MaterialSummary):
    maps: list[str]
    parameters: dict


_FAMILIES = {
    "wood": ("Bamboo", "Bark", "Chipboard", "Cork", "PaintedWood", "Planks", "Wicker", "Wood"),
    "flooring": ("Carpet", "WoodFloor"),
    "masonry": ("Bricks", "CharcoalBrick", "Concrete", "BoardFormedConcrete", "GraphiteConcrete", "PavingStones"),
    "stone": ("EngineeredQuartz", "Marble", "Onyx", "Rock", "Terrazzo", "Travertine"),
    "ceramic": ("BlueCeramicTile", "Clay", "SageCeramicTile", "SandstoneFloorTile", "Tiles", "WhiteSubwayTile"),
    "wall": ("LimewashWall", "PaintedDrywall", "PaintedPlaster", "Plaster", "Wallpaper"),
    "roofing": ("AsphaltShingles", "RoofingTiles"),
    "pavement": ("Asphalt", "Road", "TactilePaving"),
    "metal": (
        "CorrugatedSteel",
        "DiamondPlate",
        "Fence",
        "Foil",
        "GalvanizedZinc",
        "Metal",
        "MetalPlates",
        "MetalWalkway",
        "PaintedMetal",
        "Rust",
        "SheetMetal",
    ),
    "plastic": ("Plastic", "Rubber", "AcousticFoam", "Sponge"),
    "textile": ("Fabric", "Rope"),
    "leather": ("Leather",),
    "ground": ("Grass", "Gravel", "Ground", "Ice", "Moss", "ScatteredLeaves", "Snow"),
    "paper": ("Cardboard", "Paper"),
}
_FAMILY_BY_STEM = {stem: family for family, stems in _FAMILIES.items() for stem in stems}

_PLASTIC_LABELS = {
    "gloss": "Gloss molded plastic (ABS)",
    "satin": "Satin molded plastic",
    "pebbled": "Pebbled automotive trim",
    "grained": "Grained automotive dashboard",
    "scuffed": "Scuffed plastic housing",
    "recycled": "Recycled plastic with pigment chips",
    "vinyl": "Commercial vinyl flooring",
    "rubber": "Matte rubber",
}

_UNIT_INTERVAL = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_RGB = {"type": "array", "items": _UNIT_INTERVAL, "minItems": 3, "maxItems": 3}
_METAL_PARAMETERS = {
    "finish": {"type": "string", "enum": list(_METAL_FINISHES), "default": "brushed"},
    "roughness": {**_UNIT_INTERVAL, "description": "Perceptual roughness. Defaults to the selected finish."},
    "scratches": {**_UNIT_INTERVAL, "description": "Scratch amount. Defaults to the selected finish."},
    "scratch_density": {"type": "integer", "minimum": 1, "maximum": 128, "default": 10},
    "direction": {"type": "number", "default": 0.0, "description": "Degrees clockwise from U toward increasing V."},
    "reflectance": {**_RGB, "description": "Linear RGB reflectance. Defaults to the selected metal."},
}
_PLASTIC_PARAMETERS = {
    "color": {**_RGB, "description": "Mold color in sRGB. Defaults to the selected preset."},
    "roughness": {**_UNIT_INTERVAL, "description": "Perceptual roughness. Defaults to the selected preset."},
}


def _summary(material_id: str) -> _MaterialSummary:
    family, _, name = material_id.partition("/")
    if name:
        label = name.replace("_", " ").title()
        if family == "plastic":
            label = _PLASTIC_LABELS.get(name, label)
    else:
        stem = re.split(r"\d", material_id, maxsplit=1)[0]
        family = _FAMILY_BY_STEM.get(stem, "other")
        preset = PLASTIC_PRESETS.get(material_id)
        if preset is not None:
            family = "plastic"
            label = _PLASTIC_LABELS.get(preset.finish, material_id)
        else:
            label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
            if stem != material_id:
                label += f" {material_id[len(stem) :]}"
    return {"id": material_id, "label": label, "family": family}


def list_materials(*, family: str | None = None, query: str | None = None) -> list[_MaterialSummary]:
    """Find material IDs installed locally, including configurable metal and plastic presets.

    Match a family exactly or search for words in the ID, label, and family.
    Results are JSON-serializable and require neither a bake nor a CUDA device.
    """
    ids = (
        *reference_ids(),
        *(f"metal/{name}" for name in metal_names()),
        *(f"plastic/{name}" for name in _PLASTIC_FINISHES),
    )
    family = family.casefold() if family else None
    words = query.casefold().split() if query else ()
    result = []
    for material_id in ids:
        entry = _summary(material_id)
        if family is not None and entry["family"] != family:
            continue
        searchable = f"{material_id} {entry['label']} {entry['family']}".casefold()
        if all(word in searchable for word in words):
            result.append(entry)
    return result


def material_families() -> tuple[str, ...]:
    """List families represented by at least one installed material."""
    return tuple(sorted({entry["family"] for entry in list_materials()}))


def _resolve(material: Material, parameters: Mapping[str, object] | None = None) -> MetalSurface | PlasticSurface | str:
    overrides = dict(parameters or {})
    if isinstance(material, (MetalSurface, PlasticSurface)):
        if overrides:
            raise ValueError("Pass overrides on the configured surface or use a material ID with parameters.")
        return material
    family, _, name = material.partition("/")
    if family == "metal" and name in metal_names():
        surface = MetalSurface(metal=name)
        allowed = _METAL_PARAMETERS
    elif family == "plastic" and name in _PLASTIC_FINISHES:
        surface = PlasticSurface(finish=name)
        allowed = _PLASTIC_PARAMETERS
    elif material in PLASTIC_PRESETS:
        surface = PLASTIC_PRESETS[material]
        allowed = _PLASTIC_PARAMETERS
    elif material in reference_ids():
        surface = material
        allowed = {}
    else:
        raise KeyError(f"Unknown material {material!r}. Use list_materials() to find installed IDs.")
    invalid = overrides.keys() - allowed.keys()
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ValueError(f"Unsupported parameters for {material!r}: {names}. See describe_material().")
    if overrides and isinstance(surface, (MetalSurface, PlasticSurface)):
        return replace(surface, **overrides)
    return surface


def describe_material(material_id: str) -> _MaterialInfo:
    """Return a material's available maps and JSON Schema for parameter overrides."""
    surface = _resolve(material_id)
    if isinstance(surface, MetalSurface):
        properties = _METAL_PARAMETERS
    elif isinstance(surface, PlasticSurface):
        properties = _PLASTIC_PARAMETERS
    else:
        properties = {}
    return {
        **_summary(material_id),
        "maps": list(reference_maps(surface)),
        "parameters": {"type": "object", "properties": deepcopy(properties), "additionalProperties": False},
    }


def generate(
    material: Material,
    shape: tuple[int, int],
    *,
    maps: str | Sequence[str] | None = None,
    parameters: Mapping[str, object] | None = None,
    seed: int | None = None,
    device: wp.DeviceLike = None,
) -> dict[str, wp.array]:
    """Generate named Warp maps at (height, width). By default return all available maps."""
    surface = _resolve(material, parameters)
    available = reference_maps(surface)
    requested = (maps,) if isinstance(maps, str) else (available if maps is None else maps)
    for name in requested:
        if name not in available:
            raise KeyError(f"Unknown map {name!r} for {material!r}. Available maps: {', '.join(available)}.")
    return {name: bake(surface, shape, map_type=name, seed=seed, device=device) for name in dict.fromkeys(requested)}


def generate_gltf(
    material: Material,
    shape: tuple[int, int],
    *,
    parameters: Mapping[str, object] | None = None,
    seed: int | None = None,
    device: wp.DeviceLike = None,
    first_texture_index: int = 0,
) -> tuple[list[np.ndarray], dict]:
    """Generate packed RGBA8 maps and material kwargs for the path tracer in one bake."""
    surface = _resolve(material, parameters)
    return bake_gltf(surface, shape, seed=seed, device=device, first_texture_index=first_texture_index)
