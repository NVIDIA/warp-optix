# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset loaders for path tracing scenes.

This module follows the reference sample loader flow:
- glTF first
- OBJ fallback
- procedural fallback in caller
"""

from __future__ import annotations

import base64
import io
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scene import Scene

logger = logging.getLogger(__name__)


_GLTF_COMPONENT_DTYPES = {
    5120: np.int8,  # BYTE
    5121: np.uint8,  # UNSIGNED_BYTE
    5122: np.int16,  # SHORT
    5123: np.uint16,  # UNSIGNED_SHORT
    5125: np.uint32,  # UNSIGNED_INT
    5126: np.float32,  # FLOAT
}

_GLTF_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _decode_data_uri(uri: str) -> bytes:
    marker = "base64,"
    at = uri.find(marker)
    if at < 0:
        raise ValueError("Unsupported data URI format")
    return base64.b64decode(uri[at + len(marker) :])


def _read_external_uri(path: Path, uri: str) -> bytes:
    """Read a glTF external resource, confined to the directory of the glTF file.

    glTF files are untrusted input, so a ``uri`` must not reach files elsewhere
    on disk through absolute paths, ``..`` components, or filesystem links.
    """
    base = path.parent.resolve()
    target = (base / uri).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"glTF resource {uri!r} resolves outside the asset directory {base}")
    return target.read_bytes()


def _load_gltf_json_and_buffers(path: Path) -> tuple[dict, list[bytes]]:
    if path.suffix.lower() == ".glb":
        return _load_glb(path)

    root = json.loads(path.read_text(encoding="utf-8"))
    buffers: list[bytes] = []
    for b in root.get("buffers", []):
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            raw = _decode_data_uri(uri)
        else:
            raw = _read_external_uri(path, uri)
        buffers.append(raw)
    return root, buffers


def _load_glb(path: Path) -> tuple[dict, list[bytes]]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("Invalid GLB file")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or length > len(data):
        raise ValueError("Invalid GLB header")

    off = 12
    json_chunk = None
    bin_chunks: list[bytes] = []
    while off + 8 <= length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off : off + chunk_len]
        off += chunk_len
        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = chunk
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunks.append(chunk)
    if json_chunk is None:
        raise ValueError("GLB missing JSON chunk")
    root = json.loads(json_chunk.decode("utf-8"))
    return root, bin_chunks


def _node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4).T
    t = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    s = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    q = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)  # x,y,z,w
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    r = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = s[0], s[1], s[2]
    m = r @ m
    m[:3, 3] = t
    return m


def _read_accessor(root: dict, buffers: list[bytes], accessor_index: int) -> np.ndarray:
    accessors = root.get("accessors", [])
    buffer_views = root.get("bufferViews", [])
    accessor = accessors[accessor_index]
    view = buffer_views[accessor["bufferView"]]

    component_dtype = _GLTF_COMPONENT_DTYPES[accessor["componentType"]]
    num_components = _GLTF_NUM_COMPONENTS[accessor["type"]]
    count = int(accessor["count"])
    view_offset = int(view.get("byteOffset", 0))
    accessor_offset = int(accessor.get("byteOffset", 0))
    byte_stride = int(view.get("byteStride", 0))
    buffer_data = buffers[int(view["buffer"])]
    item_size = np.dtype(component_dtype).itemsize * num_components
    offset = view_offset + accessor_offset

    if byte_stride and byte_stride != item_size:
        out = np.empty((count, num_components), dtype=component_dtype)
        scalar_dtype = np.dtype(component_dtype)
        for i in range(count):
            start = offset + i * byte_stride
            end = start + item_size
            out[i] = np.frombuffer(buffer_data[start:end], dtype=scalar_dtype, count=num_components)
        return out

    total_count = count * num_components
    arr = np.frombuffer(buffer_data, dtype=component_dtype, count=total_count, offset=offset)
    return arr.reshape(count, num_components)


def _load_gltf_images(path: Path, root: dict, buffers: list[bytes]) -> list[np.ndarray | None]:
    import imageio.v3 as iio  # noqa: PLC0415

    images = root.get("images", [])
    buffer_views = root.get("bufferViews", [])

    def _decode_one(img: dict) -> np.ndarray | None:
        if "uri" in img:
            uri = img["uri"]
            if uri.startswith("data:"):
                raw = _decode_data_uri(uri)
            else:
                raw = _read_external_uri(path, uri)
        elif "bufferView" in img:
            view = buffer_views[int(img["bufferView"])]
            bo = int(view.get("byteOffset", 0))
            bl = int(view["byteLength"])
            raw = buffers[int(view["buffer"])][bo : bo + bl]
        else:
            return None

        arr = iio.imread(io.BytesIO(raw))
        arr = np.asarray(arr)
        opaque = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, opaque)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full((*arr.shape[:2], 1), opaque, dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        elif arr.shape[-1] > 4:
            arr = arr[..., :4]

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        else:
            arr = arr.astype(np.float32)
        # No vertical flip to keep texture orientation consistent with the reference.
        # glTF defines UV (0,0) = top-left, and images are stored top-to-bottom,
        # so row 0 in memory IS the top. The shader samples with fy = v * (height-1)
        # which correctly maps V=0 → row 0 → image top.
        return np.ascontiguousarray(arr, dtype=np.float32)

    if not images:
        return []

    # Image decode is one of the largest glTF load costs; decode in parallel.
    workers = max(1, min(8, len(images)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        decoded = list(ex.map(_decode_one, images))
    return decoded


def _build_texture_list(root: dict, images: list[np.ndarray | None]) -> list[np.ndarray]:
    textures = root.get("textures", [])
    out: list[np.ndarray] = []
    white = np.ones((1, 1, 4), dtype=np.float32)
    for tex in textures:
        src = int(tex.get("source", -1))
        if 0 <= src < len(images) and images[src] is not None:
            out.append(images[src])
        else:
            out.append(white)
    return out


def _texture_index(tex_info: dict | None) -> int:
    if not isinstance(tex_info, dict):
        return -1
    return int(tex_info.get("index", -1))


def _collect_srgb_texture_indices(root: dict) -> set[int]:
    """
    Collect glTF texture indices that should be sampled as sRGB color data.

    This mirrors the Vulkan path where color textures are created with
    ``R8G8B8A8_SRGB`` and scalar/vector utility textures remain linear.
    """
    srgb: set[int] = set()
    for m in root.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        ext = m.get("extensions", {})

        # Core glTF color textures.
        idx = _texture_index(pbr.get("baseColorTexture"))
        if idx >= 0:
            srgb.add(idx)
        idx = _texture_index(m.get("emissiveTexture"))
        if idx >= 0:
            srgb.add(idx)

        # KHR_materials_pbrSpecularGlossiness color textures.
        spec_gloss = ext.get("KHR_materials_pbrSpecularGlossiness", {})
        idx = _texture_index(spec_gloss.get("diffuseTexture"))
        if idx >= 0:
            srgb.add(idx)
        idx = _texture_index(spec_gloss.get("specularGlossinessTexture"))
        if idx >= 0:
            srgb.add(idx)

        # Additional color extensions.
        specular_ext = ext.get("KHR_materials_specular", {})
        idx = _texture_index(specular_ext.get("specularColorTexture"))
        if idx >= 0:
            srgb.add(idx)

        sheen_ext = ext.get("KHR_materials_sheen", {})
        idx = _texture_index(sheen_ext.get("sheenColorTexture"))
        if idx >= 0:
            srgb.add(idx)

        diff_tx_ext = ext.get("KHR_materials_diffuse_transmission", {})
        idx = _texture_index(diff_tx_ext.get("diffuseTransmissionColorTexture"))
        if idx >= 0:
            srgb.add(idx)

    return srgb


def _add_gltf_material(
    scene: Scene,
    root: dict,
    material_index: int | None,
    cache: dict[int, int],
    texture_index_offset: int = 0,
) -> int:
    if material_index is None:
        return scene.materials.add_default()
    material_index = int(material_index)
    if material_index in cache:
        return cache[material_index]

    m = root.get("materials", [])[material_index]
    pbr = m.get("pbrMetallicRoughness", {})
    ext = m.get("extensions", {})
    tex_default = {"index": -1, "texCoord": 0}

    def _tex_info(t: dict | None, *, include_scale: bool = False) -> dict:
        if not t:
            return dict(tex_default)
        tex_index = int(t.get("index", -1))
        if tex_index >= 0:
            tex_index += int(texture_index_offset)
        out = {
            "index": tex_index,
            "texCoord": int(t.get("texCoord", 0)),
        }
        ext = t.get("extensions", {})
        tt = ext.get("KHR_texture_transform", {}) if isinstance(ext, dict) else {}
        if tt:
            out["transform"] = {
                "offset": list(tt.get("offset", [0.0, 0.0])),
                "scale": list(tt.get("scale", [1.0, 1.0])),
                "rotation": float(tt.get("rotation", 0.0)),
            }
            if "texCoord" in tt:
                out["texCoord"] = int(tt["texCoord"])
        if include_scale and "scale" in t:
            out["scale"] = float(t["scale"])
        return out

    kwargs = {
        "base_color": tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])),
        "emissive_factor": tuple(m.get("emissiveFactor", [0.0, 0.0, 0.0])),
        "metallic": float(pbr.get("metallicFactor", 1.0)),
        "roughness": float(pbr.get("roughnessFactor", 1.0)),
        "alpha_mode": str(m.get("alphaMode", "OPAQUE")),
        "alpha_cutoff": float(m.get("alphaCutoff", 0.5)),
        "double_sided": bool(m.get("doubleSided", False)),
        "base_color_texture": _tex_info(pbr.get("baseColorTexture")),
        "normal_texture": _tex_info(m.get("normalTexture"), include_scale=True),
        "metallic_roughness_texture": _tex_info(pbr.get("metallicRoughnessTexture")),
        "emissive_texture": _tex_info(m.get("emissiveTexture")),
        "occlusion_texture": _tex_info(m.get("occlusionTexture")),
        "occlusion_strength": float(m.get("occlusionTexture", {}).get("strength", 1.0)),
        "transmission": float(ext.get("KHR_materials_transmission", {}).get("transmissionFactor", 0.0)),
        "ior": float(ext.get("KHR_materials_ior", {}).get("ior", 1.5)),
        "thickness": float(ext.get("KHR_materials_volume", {}).get("thicknessFactor", 0.0)),
        "attenuation_color": tuple(ext.get("KHR_materials_volume", {}).get("attenuationColor", [1.0, 1.0, 1.0])),
        "attenuation_distance": float(ext.get("KHR_materials_volume", {}).get("attenuationDistance", 1.0e10)),
    }
    mat_id = scene.materials.add_gltf_material(**kwargs)
    cache[material_index] = mat_id
    return mat_id


def load_scene_from_gltf(scene: Scene, gltf_path: str | Path, root_transform: np.ndarray | None = None) -> bool:
    path = Path(gltf_path)
    if not path.exists():
        return False

    t_all_start = time.perf_counter()

    t0 = time.perf_counter()
    root, buffers = _load_gltf_json_and_buffers(path)
    t_parse = time.perf_counter() - t0

    from .scene import Mesh  # noqa: PLC0415

    t0 = time.perf_counter()
    gltf_images = _load_gltf_images(path, root, buffers)
    t_images = time.perf_counter() - t0

    t0 = time.perf_counter()
    texture_index_offset = int(scene.texture_count)
    textures = _build_texture_list(root, gltf_images)
    srgb_texture_indices = _collect_srgb_texture_indices(root)
    scene.set_gltf_textures(
        textures,
        srgb_texture_indices=srgb_texture_indices,
        append=texture_index_offset > 0,
    )
    t_textures = time.perf_counter() - t0

    root_world = np.eye(4, dtype=np.float32)
    if root_transform is not None:
        root_world = np.asarray(root_transform, dtype=np.float32).reshape(4, 4)

    nodes = root.get("nodes", [])
    meshes = root.get("meshes", [])
    scenes = root.get("scenes", [])
    default_scene = int(root.get("scene", 0)) if scenes else -1

    material_cache: dict[int, int] = {}
    stats = {
        "nodes": 0,
        "prims": 0,
        "verts": 0,
        "tris": 0,
        "t_accessors": 0.0,
        "t_xform": 0.0,
        "t_material": 0.0,
        "t_mesh": 0.0,
    }

    def visit(node_index: int, parent_world: np.ndarray):
        stats["nodes"] += 1
        node = nodes[node_index]
        local = _node_local_matrix(node)
        world = parent_world @ local

        mesh_idx = node.get("mesh")
        if mesh_idx is not None:
            mesh_def = meshes[int(mesh_idx)]
            for prim in mesh_def.get("primitives", []):
                stats["prims"] += 1
                attrs = prim.get("attributes", {})
                if "POSITION" not in attrs:
                    continue

                t_acc = time.perf_counter()
                pos = _read_accessor(root, buffers, int(attrs["POSITION"])).astype(np.float32)
                nrm = None
                uv0 = None
                uv1 = None
                if "NORMAL" in attrs:
                    nrm = _read_accessor(root, buffers, int(attrs["NORMAL"])).astype(np.float32)
                if "TEXCOORD_0" in attrs:
                    uv0 = _read_accessor(root, buffers, int(attrs["TEXCOORD_0"])).astype(np.float32)
                if "TEXCOORD_1" in attrs:
                    uv1 = _read_accessor(root, buffers, int(attrs["TEXCOORD_1"])).astype(np.float32)

                if "indices" in prim:
                    idx = _read_accessor(root, buffers, int(prim["indices"])).reshape(-1)
                else:
                    idx = np.arange(pos.shape[0], dtype=np.uint32)
                if idx.size % 3 != 0:
                    continue
                tri = idx.reshape(-1, 3).astype(np.uint32)
                stats["t_accessors"] += time.perf_counter() - t_acc

                t_xf = time.perf_counter()
                pos4 = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=np.float32)], axis=1)
                world_pos = (pos4 @ world.T)[:, :3].astype(np.float32)

                world_nrm = None
                if nrm is not None:
                    normal_mat = np.linalg.inv(world[:3, :3]).T
                    world_nrm = (nrm @ normal_mat.T).astype(np.float32)
                    nlen = np.linalg.norm(world_nrm, axis=1, keepdims=True)
                    nlen[nlen == 0.0] = 1.0
                    world_nrm /= nlen
                stats["t_xform"] += time.perf_counter() - t_xf

                t_mat = time.perf_counter()
                mat_id = _add_gltf_material(
                    scene,
                    root,
                    prim.get("material"),
                    material_cache,
                    texture_index_offset=texture_index_offset,
                )
                stats["t_material"] += time.perf_counter() - t_mat

                t_mesh = time.perf_counter()
                mesh = Mesh(world_pos, tri, normals=world_nrm, texcoords=uv0, texcoords1=uv1, material_id=mat_id)
                m_idx = scene.add_mesh(mesh)
                scene.add_instance(m_idx)
                stats["t_mesh"] += time.perf_counter() - t_mesh
                stats["verts"] += int(world_pos.shape[0])
                stats["tris"] += int(tri.shape[0])

        for child in node.get("children", []):
            visit(int(child), world)

    if default_scene >= 0:
        for node_idx in scenes[default_scene].get("nodes", []):
            visit(int(node_idx), root_world)
    else:
        for i in range(len(nodes)):
            visit(i, root_world)

    t_total = time.perf_counter() - t_all_start
    logger.info(
        "[glTF timing] parse=%.1f ms, images=%.1f ms, set_textures=%.1f ms, "
        "accessors=%.1f ms, xform=%.1f ms, materials=%.1f ms, mesh_build=%.1f ms, "
        "total=%.1f ms, nodes=%d prims=%d verts=%d tris=%d images=%d textures=%d",
        t_parse * 1000.0,
        t_images * 1000.0,
        t_textures * 1000.0,
        stats["t_accessors"] * 1000.0,
        stats["t_xform"] * 1000.0,
        stats["t_material"] * 1000.0,
        stats["t_mesh"] * 1000.0,
        t_total * 1000.0,
        stats["nodes"],
        stats["prims"],
        stats["verts"],
        stats["tris"],
        len(gltf_images),
        len(root.get("textures", [])),
    )

    return scene.mesh_count > 0


def load_scene_from_obj(scene: Scene, obj_path: str | Path) -> bool:
    path = Path(obj_path)
    if not path.exists():
        return False
    from .scene import Mesh  # noqa: PLC0415

    verts: list[list[float]] = []
    norms: list[list[float]] = []
    uvs: list[list[float]] = []
    out_v: list[list[float]] = []
    out_n: list[list[float]] = []
    out_uv: list[list[float]] = []
    tris: list[list[int]] = []
    remap: dict[tuple[int, int, int], int] = {}

    def parse_idx(token: str, length: int) -> int:
        val = int(token)
        return (val - 1) if val > 0 else (length + val)

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "vn" and len(parts) >= 4:
            norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "vt" and len(parts) >= 3:
            uvs.append([float(parts[1]), float(parts[2])])
        elif tag == "f" and len(parts) >= 4:
            face_ix: list[int] = []
            for ref in parts[1:]:
                sub = ref.split("/")
                v_i = parse_idx(sub[0], len(verts))
                vt_i = parse_idx(sub[1], len(uvs)) if len(sub) > 1 and sub[1] else -1
                vn_i = parse_idx(sub[2], len(norms)) if len(sub) > 2 and sub[2] else -1
                key = (v_i, vt_i, vn_i)
                if key not in remap:
                    remap[key] = len(out_v)
                    out_v.append(verts[v_i])
                    out_uv.append(uvs[vt_i] if vt_i >= 0 else [0.0, 0.0])
                    out_n.append(norms[vn_i] if vn_i >= 0 else [0.0, 0.0, 0.0])
                face_ix.append(remap[key])
            for i in range(1, len(face_ix) - 1):
                tris.append([face_ix[0], face_ix[i], face_ix[i + 1]])

    if not out_v or not tris:
        return False

    v_np = np.asarray(out_v, dtype=np.float32)
    i_np = np.asarray(tris, dtype=np.uint32)
    uv_np = np.asarray(out_uv, dtype=np.float32)
    n_np = np.asarray(out_n, dtype=np.float32)
    if np.allclose(n_np, 0.0):
        n_np = None

    mat_id = scene.materials.add_default()
    mesh = Mesh(v_np, i_np, normals=n_np, texcoords=uv_np, material_id=mat_id)
    mesh_idx = scene.add_mesh(mesh)
    scene.add_instance(mesh_idx)
    return True
