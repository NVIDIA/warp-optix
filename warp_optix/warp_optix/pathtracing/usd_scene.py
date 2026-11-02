# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained USD transform hierarchy for dynamic path-traced scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .scene import Scene


@dataclass(frozen=True, slots=True)
class USDTransformHandle:
    """Stable transform identifier for one composed USD prim path."""

    index: int
    path: str
    _owner_id: int = field(repr=False)

    def __int__(self) -> int:
        return self.index


class USDScene:
    """Path-addressable USD hierarchy retained after geometry import.

    Local-to-world hierarchy composition, render-node updates, and OptiX TLAS
    updates execute on CUDA. Device batches accept caller-owned ``wp.int32``
    handle arrays and ``wp.mat44`` local-transform arrays without staging
    through NumPy.
    """

    def __init__(self, scene: Scene, stage, source_path: str, paths: Iterable[str]):
        self._scene = scene
        self._stage = stage
        self._source_path = str(source_path)
        self._handles = tuple(
            USDTransformHandle(index, str(path), id(self))
            for index, path in enumerate(paths)
        )
        self._path_to_handle = {handle.path: handle for handle in self._handles}
        self._local_transforms_device = None
        self._world_transforms_device = None

    def _attach_device_arrays(self, local_transforms, world_transforms):
        self._local_transforms_device = local_transforms
        self._world_transforms_device = world_transforms

    @property
    def transforms(self) -> tuple[USDTransformHandle, ...]:
        """All transformable composed USD paths, in stable hierarchy order."""
        return self._handles

    @property
    def stage(self):
        """The composed ``pxr.Usd.Stage`` used by the importer."""
        return self._stage

    @property
    def source_path(self) -> str:
        return self._source_path

    def get_prim(self, path: str):
        """Return any composed USD prim, including non-transformable prims."""
        return self._stage.GetPrimAtPath(str(path))

    @property
    def transform_count(self) -> int:
        return len(self._handles)

    @property
    def local_transforms_device(self) -> wp.array | None:
        """CUDA ``wp.mat44`` array, available after the renderer scene is built."""
        return self._local_transforms_device

    @property
    def world_transforms_device(self) -> wp.array | None:
        """CUDA-composed world ``wp.mat44`` array, available after build."""
        return self._world_transforms_device

    def get_transform(self, path: str) -> USDTransformHandle | None:
        """Return the handle for an exact composed USD prim path."""
        return self._path_to_handle.get(str(path))

    def require_transform(self, path: str) -> USDTransformHandle:
        """Return a handle or raise ``KeyError`` when the path is not transformable."""
        handle = self.get_transform(path)
        if handle is None:
            raise KeyError(f"USD path has no transform handle: {path}")
        return handle

    def instance_ids(self, handle: USDTransformHandle | int) -> tuple[int, ...]:
        """Return render instances attached directly to a transform node."""
        node_index = self._index(handle)
        selected = self._scene._usd_instance_ids[
            self._scene._usd_instance_node_ids == node_index
        ]
        return tuple(int(value) for value in selected)

    def set_visible(self, visible: bool):
        """Show or hide every render instance imported from this USD stage."""
        self._require_current()
        instance_ids = self._scene._usd_instance_ids
        if len(instance_ids) == 0:
            return
        self._scene.set_instances_visible_batch(instance_ids, bool(visible))

    def get_local_transform(self, handle: USDTransformHandle | int) -> np.ndarray:
        """Read one local matrix; this synchronizes when CUDA state exists."""
        index = self._index(handle)
        if self._local_transforms_device is not None:
            return self._local_transforms_device.numpy()[index].copy()
        return self._scene._usd_local_transforms[index].copy()

    def get_world_transform(self, handle: USDTransformHandle | int) -> np.ndarray:
        """Read one composed world matrix; this synchronizes CUDA work."""
        index = self._index(handle)
        if self._world_transforms_device is not None:
            return self._world_transforms_device.numpy()[index].copy()
        return self._scene._usd_world_transforms[index].copy()

    def update_local_transform(
        self,
        handle: USDTransformHandle | int,
        matrix: np.ndarray,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Update one local matrix through the batched CUDA path."""
        self.update_local_transforms(
            [handle], [matrix], stream=stream, rebuild_tlas=rebuild_tlas
        )

    def update_local_transforms(
        self,
        handles: Iterable[USDTransformHandle | int],
        matrices: np.ndarray,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Upload and apply a NumPy batch on an optional Warp CUDA stream."""
        indices = np.fromiter((self._index(handle) for handle in handles), dtype=np.int32)
        self._scene.set_usd_local_transforms(
            indices, matrices, stream=stream, rebuild_tlas=rebuild_tlas
        )

    def update_local_transforms_device(
        self,
        transform_count: wp.array,
        transform_ids: wp.array,
        local_transforms: wp.array,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply a device-counted prefix of CUDA handle and local-matrix arrays."""
        self._require_current()
        self._scene.set_usd_local_transforms_device(
            transform_count,
            transform_ids,
            local_transforms,
            stream=stream,
            rebuild_tlas=rebuild_tlas,
        )

    def update_local_transform_trs_device(
        self,
        transform_count: wp.array,
        transform_ids: wp.array,
        local_poses: wp.array,
        local_scales: wp.array,
        *,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply a device-counted prefix of CUDA pose and scale arrays."""
        self._require_current()
        self._scene.set_usd_local_transform_trs_device(
            transform_count,
            transform_ids,
            local_poses,
            local_scales,
            stream=stream,
            rebuild_tlas=rebuild_tlas,
        )

    def update_tlas(self, *, stream=None):
        """Update OptiX traversal after batches submitted with auto-update off."""
        self._require_current()
        if stream is None:
            self._scene.rebuild_tlas()
            return
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            self._scene.rebuild_tlas()

    def _index(self, handle: USDTransformHandle | int) -> int:
        self._require_current()
        index = int(handle)
        if index < 0 or index >= len(self._handles):
            raise IndexError(f"USD transform handle is out of range: {index}")
        if isinstance(handle, USDTransformHandle) and handle._owner_id != id(self):
            raise ValueError("USD transform handle belongs to a different stage")
        return index

    def _require_current(self):
        if self._scene.usd_scene is not self:
            raise RuntimeError("USD scene handle was invalidated by a later scene load or clear")
