# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load scenes from Benedikt Bitterli's rendering resources (Mitsuba 3 format).

Scenes are downloaded on first use into ``~/.cache/warp-optix/scenes`` (override with
``WARP_OPTIX_CACHE``). Only the subset of Mitsuba needed by the bundled scenes is
supported: OBJ and rectangle shapes, matrix/rotate/scale/translate transforms, and
diffuse, plastic, conductor and dielectric BSDFs.
"""

from __future__ import annotations

import math
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

URL = "https://benedikt-bitterli.me/resources/mitsuba/{name}.zip"

# Name -> (title, author, license) as listed on https://benedikt-bitterli.me/resources/.
CREDITS = {
    "glass-of-water": ("Glass of Water", "aXel", "CC0"),
    "dining-room": ("The Breakfast Room", "Wig42", "CC BY 3.0"),
}


def fetch(name: str) -> Path:
    """Return the scene XML of ``name``, downloading the scene if needed."""
    if name not in CREDITS:
        raise ValueError(f"Unknown bundled scene: {name!r}")
    root = Path(os.environ.get("WARP_OPTIX_CACHE", Path.home() / ".cache" / "warp-optix")) / "scenes"
    xml = root / name / name / "scene_v3.xml"
    if not xml.exists():
        title, author, license = CREDITS[name]
        print(f"[optix] downloading '{title}' by {author} ({license}) from benedikt-bitterli.me")
        root.mkdir(parents=True, exist_ok=True)
        # Publish only a complete scene; failed or concurrent downloads must not
        # leave a cached XML whose meshes and textures have not arrived yet.
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
            staging = Path(temporary)
            archive = staging / "scene.zip"
            urllib.request.urlretrieve(URL.format(name=name), archive)
            extracted = staging / "extracted"
            with zipfile.ZipFile(archive) as zf:
                for entry in zf.infolist():
                    _asset_path(extracted, entry.filename)
                zf.extractall(extracted)
            if not (extracted / name / "scene_v3.xml").is_file():
                raise ValueError(f"Scene archive lacks {name}/scene_v3.xml")
            try:
                extracted.rename(root / name)
            except OSError:
                if not xml.is_file():
                    raise
    return xml


def _asset_path(directory: Path, filename: str) -> Path:
    """Keep archive entries and XML asset references inside the scene directory."""
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Scene asset escapes its directory: {filename!r}")
    return path


def _values(node, tag, name):
    for child in node:
        if child.tag == tag and child.get("name") == name:
            return [float(v) for v in child.get("value").replace(",", " ").split()]
    return None


def _rgb(node, name, default):
    values = _values(node, "rgb", name) or _values(node, "float", name)
    if values is None:
        return default
    return tuple(values * 3) if len(values) == 1 else tuple(values)


def _float(node, name, default):
    values = _values(node, "float", name)
    return default if values is None else values[0]


def _texture_file(node, name):
    for child in node:
        if child.tag == "texture" and child.get("name") == name:
            for prop in child:
                if prop.get("name") == "filename":
                    return prop.get("value")
    return None


def _transform(node) -> np.ndarray:
    matrix = np.eye(4)
    transform = node.find("transform")
    for op in [] if transform is None else transform:
        step = np.eye(4)
        if op.tag == "matrix":
            step = np.array(op.get("value").split(), dtype=float).reshape(4, 4)
        elif op.tag == "translate":
            step[:3, 3] = [float(op.get(k, 0.0)) for k in "xyz"]
        elif op.tag == "scale":
            value = op.get("value")
            step[:3, :3] = np.diag([float(value)] * 3 if value else [float(op.get(k, 1.0)) for k in "xyz"])
        elif op.tag == "rotate":
            axis = np.array([float(op.get(k, 0.0)) for k in "xyz"])
            x, y, z = axis / np.linalg.norm(axis)
            a = math.radians(float(op.get("angle")))
            c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
            step[:3, :3] = [
                [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
            ]
        matrix = step @ matrix
    return matrix


def _read_obj(path: Path):
    """Return unwelded triangle corners: positions, UVs and normals."""
    positions, uvs, normals = [], [], []
    corners = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                positions.append(line.split()[1:4])
            elif line.startswith("vt "):
                uvs.append(line.split()[1:3])
            elif line.startswith("vn "):
                normals.append(line.split()[1:4])
            elif line.startswith("f "):
                face = []
                for corner in line.split("#", 1)[0].split()[1:]:
                    values = (corner.split("/") + ["", ""])[:3]
                    indices = []
                    for value, count in zip(values, (len(positions), len(uvs), len(normals))):
                        if not value:
                            indices.append(None)
                            continue
                        index = int(value)
                        index = index - 1 if index > 0 else count + index
                        if not 0 <= index < count or int(value) == 0:
                            raise ValueError(f"Invalid OBJ index in {path}: {corner!r}")
                        indices.append(index)
                    if indices[0] is None:
                        raise ValueError(f"Missing OBJ position index in {path}")
                    face.append(indices)
                for k in range(1, len(face) - 1):
                    corners += [face[0], face[k], face[k + 1]]
    positions = np.array(positions, dtype=np.float32)
    p = positions[np.array([c[0] for c in corners], dtype=np.intp)].reshape(-1, 3)
    uv = np.array([uvs[c[1]] if c[1] is not None else (0.0, 0.0) for c in corners], dtype=np.float32).reshape(-1, 2)
    tri = p.reshape(-1, 3, 3)
    face_normals = np.repeat(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), 3, axis=0)
    n = np.array(
        [normals[c[2]] if c[2] is not None else face_normals[i] for i, c in enumerate(corners)], dtype=np.float32
    ).reshape(-1, 3)
    return p, uv, n


_RECTANGLE = (
    np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
    np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32),
    np.tile(np.array([0, 0, 1], dtype=np.float32), (6, 1)),
)


class MitsubaScene:
    """Adds the shapes of a Mitsuba scene to a :class:`PathTracer` and exposes its camera."""

    def __init__(self, xml_path: Path, *, emitters: bool = True):
        self.directory = Path(xml_path).parent
        self.root = ET.parse(xml_path).getroot()
        self.emitters = emitters
        self._textures: list[np.ndarray] = []
        self._texture_ids: dict[str, int] = {}

    def _texture(self, filename: str) -> int:
        if filename not in self._texture_ids:
            from PIL import Image

            image = np.asarray(Image.open(_asset_path(self.directory, filename)).convert("RGBA"))[::-1]
            self._texture_ids[filename] = len(self._textures)
            self._textures.append(np.ascontiguousarray(image))
        return self._texture_ids[filename]

    def _material(self, api, bsdf, emitter) -> int:
        materials = api.scene.materials
        if emitter is not None:
            return api.create_pbr_material((0.0, 0.0, 0.0), 1.0, 0.0, emissive=_rgb(emitter, "radiance", (1, 1, 1)))
        while bsdf is not None and bsdf.get("type") in ("twosided", "mask", "bumpmap"):
            bsdf = bsdf.find("bsdf")
        if bsdf is None:
            return materials.add_gltf_material(base_color=(0.7, 0.7, 0.7, 1.0), metallic=0.0, roughness=1.0)
        kind = bsdf.get("type")
        alpha = _float(bsdf, "alpha", 0.0)
        roughness = math.sqrt(alpha)
        if kind.endswith("dielectric"):
            ior = _float(bsdf, "int_ior", 1.5) / _float(bsdf, "ext_ior", 1.0)
            return materials.add_gltf_material(
                base_color=(1.0, 1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=roughness,
                transmission=1.0,
                ior=max(ior, 1.0 / ior),
                thickness=0.0 if kind == "thindielectric" else 1.0,
                double_sided=True,
            )
        if kind.endswith("conductor"):
            # Normal-incidence reflectance from the complex index of refraction.
            eta = _rgb(bsdf, "eta", (0.2, 0.92, 1.1))
            k = _rgb(bsdf, "k", (3.9, 2.45, 2.14))
            tint = _rgb(bsdf, "specular_reflectance", (1.0, 1.0, 1.0))
            f0 = [s * ((n - 1) ** 2 + kk**2) / ((n + 1) ** 2 + kk**2) for s, n, kk in zip(tint, eta, k)]
            return materials.add_gltf_material(base_color=(*f0, 1.0), metallic=1.0, roughness=max(roughness, 0.02))
        name = "reflectance" if kind == "diffuse" else "diffuse_reflectance"
        options = {
            "metallic": 0.0,
            "roughness": 1.0 if kind == "diffuse" else max(roughness, 0.05),
            "specular": 0.0 if kind == "diffuse" else 1.0,
        }
        texture = _texture_file(bsdf, name)
        if texture is not None:
            return materials.add_gltf_material(
                base_color=(1.0, 1.0, 1.0, 1.0), base_color_texture={"index": self._texture(texture)}, **options
            )
        return materials.add_gltf_material(base_color=(*_rgb(bsdf, name, (0.5, 0.5, 0.5)), 1.0), **options)

    def build(self, api) -> None:
        bsdfs = {b.get("id"): b for b in self.root.findall("bsdf")}
        materials: dict = {}
        for shape in self.root.findall("shape"):
            emitter = shape.find("emitter")
            if emitter is not None and not self.emitters:
                continue
            ref = shape.find("ref")
            bsdf = bsdfs[ref.get("id")] if ref is not None else shape.find("bsdf")
            key = id(shape) if emitter is not None or ref is None else ref.get("id")
            if key not in materials:
                materials[key] = self._material(api, bsdf, emitter)
            if shape.get("type") == "obj":
                filename = next(c.get("value") for c in shape if c.get("name") == "filename")
                p, uv, n = _read_obj(_asset_path(self.directory, filename))
            elif shape.get("type") == "rectangle":
                p, uv, n = _RECTANGLE
            else:
                continue
            m = _transform(shape)
            p = p @ m[:3, :3].T + m[:3, 3]
            n = n @ np.linalg.inv(m[:3, :3])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-12)
            indices = np.arange(len(p), dtype=np.uint32).reshape(-1, 3)
            mesh = api.create_mesh(
                p.astype(np.float32), indices, normals=n.astype(np.float32), uvs=uv, material_id=materials[key]
            )
            api.create_instance(mesh)
        if self._textures:
            api.scene.set_gltf_textures(self._textures, srgb_texture_indices=set(range(len(self._textures))))

    def camera(self, aspect: float = 16.0 / 9.0):
        """Return the sensor's position, view direction and vertical field of view in degrees."""
        sensor = self.root.find("sensor")
        m = _transform(sensor)
        fov = _float(sensor, "fov", 40.0)
        vertical = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov) / 2.0) / aspect))
        return tuple(m[:3, 3]), m[:3, 2], vertical
