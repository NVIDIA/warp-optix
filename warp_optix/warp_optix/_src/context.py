# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-device OptiX contexts, device/stream resolution, and resource release.

Like Warp's CUDA context, the OptiX device context is implicit: each CUDA
device gets one, created on first use with the settings in
:mod:`warp_optix.config` and destroyed at interpreter exit. Resources
(:class:`~warp_optix.Accel`, :class:`~warp_optix.Pipeline`) hold a reference
to their context.

Lifetime contract (the same as ``wp.Mesh`` and ``wp.array``):

* Builds, refits, and launches record no events and hold no references to
  their inputs beyond what the resources themselves keep.
* ``close()`` (also called by ``with``, and when the last reference is dropped)
  makes a resource unusable at once and releases its device memory right away.
  Device memory is ``wp.array`` storage, whose release is stream-ordered by
  Warp, so work already submitted can still read it. It never blocks the host.
* Native OptiX objects of a pipeline (pipeline, module, program groups) are
  destroyed under the device's CUDA context once the launches submitted before
  the close have finished; while a graph capture is running on the device,
  destruction waits until it ends.
* Kernels see an acceleration structure only as an integer ``id``. Keep the
  :class:`~warp_optix.Accel` alive while work that reads the id may still be
  submitted, as with ``wp.Mesh.id``.
* OptiX work cannot be captured into CUDA graphs; every entry point refuses
  capture.
"""

from __future__ import annotations

import atexit
import functools
import logging
import threading
import weakref
from contextlib import contextmanager

import warp as wp

from warp_optix import config, native

CAPTURE_ERROR = "OptiX work cannot be recorded into CUDA graphs; call it outside wp.ScopedCapture"

_log = logging.getLogger("warp_optix")
_contexts: dict[str, Context] = {}
_lock = threading.Lock()


class Context:
    """The OptiX device context of one CUDA device.

    Obtain it with :func:`get_context`; it is created on first use and lives until
    interpreter exit. ``native`` gives the PyOptiX ``DeviceContext`` for advanced
    calls; it is borrowed and must not be destroyed.
    """

    def __init__(self, device: wp.Device):
        self.device = device
        self.validation = bool(config.validation)
        self.log_level = int(config.log_level)
        self.log_message_count = 0
        self._destroyed = False
        # Native destroys waiting for (events, streams, callback); events are recorded outside graph capture.
        self._releases: list[tuple[list[wp.Event] | None, tuple, object]] = []
        owner = weakref.ref(self)

        def log(level, tag, message):
            context = owner()
            if context is not None:
                context.log_message_count += 1
            _log.log(
                {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO}.get(level, logging.DEBUG),
                "[%s] %s",
                tag,
                message,
            )

        options = native.DeviceContextOptions(
            logCallbackFunction=log,
            logCallbackLevel=self.log_level,
            validationMode=native.DEVICE_CONTEXT_VALIDATION_MODE_ALL
            if self.validation
            else native.DEVICE_CONTEXT_VALIDATION_MODE_OFF,
        )
        with wp.ScopedDevice(device):
            self._native = native.deviceContextCreate(device.context, options)

    def __repr__(self):
        return f"Context(device={self.device.alias!r})"

    @property
    def native(self):
        """The PyOptiX ``DeviceContext`` (borrowed; do not destroy)."""
        return self._native

    def release(self, callback, streams=()) -> None:
        """Run ``callback`` under this device's CUDA context once work submitted to ``streams`` has finished.

        It runs immediately when that work is already done and no graph capture is
        running on the device; otherwise it runs on a later OptiX call on this
        device, or at interpreter exit.
        """
        if self._destroyed:
            # The context, and every native object created from it, is already gone (interpreter exit).
            return
        self._releases.append((None, tuple(streams), callback))
        self.collect()

    def collect(self) -> None:
        """Run pending releases whose work has finished, unless a graph capture is running."""
        if not self._releases or self.device.is_capturing:
            # Native destroys free device memory, which graph capture does not allow.
            return
        # Callbacks (or the garbage collector) may re-enter collect(); swap first so each release runs once.
        pending, self._releases = self._releases, []
        waiting = []
        with wp.ScopedDevice(self.device):
            for events, streams, callback in pending:
                if events is None:
                    events = [stream.record_event() for stream in streams]
                if all(event.is_complete for event in events):
                    _run(callback)
                else:
                    waiting.append((events, streams, callback))
        self._releases[:0] = waiting

    def check_arrays(self, values) -> None:
        """Check that arrays in ``values`` (including nested sequences and struct fields) are on this device."""
        for value in values:
            if isinstance(value, wp.array):
                if value.device != self.device:
                    raise ValueError("Array and OptiX resource must use the same CUDA device")
            elif isinstance(value, (tuple, list)):
                self.check_arrays(value)
            elif wp.types.is_struct(value):
                self._check_struct(value)

    def _check_struct(self, instance) -> None:
        arrays, structs = _struct_fields(instance._cls)
        device = self.device
        for name in arrays:
            value = getattr(instance, name)
            if value is not None and value.device != device:
                raise ValueError("Array and OptiX resource must use the same CUDA device")
        for name in structs:
            value = getattr(instance, name)
            if value is not None:
                self._check_struct(value)

    def _shutdown(self) -> None:
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            while self._releases:
                pending, self._releases = self._releases, []
                for _, _, callback in pending:
                    _run(callback)
            self._native.destroy()
            self._destroyed = True


def get_context(device=None) -> Context:
    """Return the OptiX context of a CUDA device, creating it on first use.

    Most code never needs this; it gives advanced code the native context
    through ``get_context(device).native``.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        raise ValueError("OptiX requires a CUDA device")
    with _lock:
        context = _contexts.get(device.alias)
        if context is None:
            if not _contexts:
                atexit.register(_shutdown_all)
            context = _contexts[device.alias] = Context(device)
        return context


def resolve(device=None, stream=None) -> tuple[Context, wp.Stream]:
    """Return the context and stream for a ``device``/``stream`` pair.

    Neither given: the current device and its current stream. Only ``stream``:
    the stream's device. Only ``device``: that device's current stream. Both:
    they must match.
    """
    if stream is not None:
        if not isinstance(stream, wp.Stream):
            raise TypeError("stream must be a Warp Stream")
        if device is not None and wp.get_device(device) != stream.device:
            raise ValueError("Stream and device must be the same CUDA device")
        device = stream.device
    context = get_context(device)
    if stream is None:
        stream = wp.get_stream(context.device)
    if stream.is_capturing:
        raise RuntimeError(CAPTURE_ERROR)
    context.collect()
    return context, stream


@contextmanager
def native_objects(context: Context):
    """Collect native objects created in the block; destroy them if the block fails."""
    objects: list = []
    with wp.ScopedDevice(context.device):
        try:
            yield objects
        except BaseException:
            # Failed operations may already have submitted GPU work that reads
            # buffers owned by the traceback; wait before destroying anything.
            wp.synchronize_device(context.device)
            for obj in reversed(objects):
                try:
                    obj.destroy()
                except Exception:
                    _log.exception("Failed to clean up an OptiX object")
            raise


@functools.cache
def _struct_fields(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Names of the array fields and the struct fields of a Warp struct type."""
    arrays = tuple(name for name, var in cls.vars.items() if wp.types.is_array(var.type))
    structs = tuple(name for name, var in cls.vars.items() if isinstance(var.type, type(cls)))
    return arrays, structs


def _run(callback) -> None:
    if callback is None:
        return
    try:
        callback()
    except Exception:
        _log.exception("Failed to release an OptiX resource")


def _shutdown_all() -> None:
    with _lock:
        contexts = list(_contexts.values())
        _contexts.clear()
    for context in contexts:
        try:
            context._shutdown()
        except Exception:
            _log.exception("Failed to destroy the OptiX context of %s", context.device)
