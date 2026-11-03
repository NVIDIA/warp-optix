# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-device OptiX contexts, device/stream resolution, and deferred release.

Like Warp's CUDA context, the OptiX device context is implicit: each CUDA
device gets one, created on first use with the settings in
:mod:`warp_optix.config` and destroyed at interpreter exit. Resources
(:class:`~warp_optix.Accel`, :class:`~warp_optix.Pipeline`,
:class:`~warp_optix.LaunchParams`) hold a reference to their context.

Lifetime contract:

* Every build, refit, and launch records a CUDA event and keeps the arrays,
  acceleration structures, pipelines, and parameter blocks it reads alive until
  that event completes.
* ``close()`` (also called by ``with`` and ``__del__``) makes a resource unusable
  at once and releases its memory and native objects only after all OptiX work
  submitted before the close has completed. It never blocks the host.
* Kernels see an acceleration structure only as an integer ``id``. Keep the
  :class:`~warp_optix.Accel` alive (or pass the object itself in launch inputs)
  while work that reads the id may still be submitted, as with ``wp.Mesh.id``.
* OptiX work cannot be captured into CUDA graphs; every entry point refuses
  capture.
"""

from __future__ import annotations

import atexit
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
        # Live acceleration structures by id, for validating instance children.
        self._accels: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        # (event, references) of submitted work, and (events, callback, references) of pending releases.
        self._pending: list[tuple[wp.Event, tuple]] = []
        self._releases: list[tuple[list[wp.Event], object, tuple]] = []
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

    def retain(self, stream: wp.Stream, references) -> None:
        """Keep ``references`` alive until the work just submitted to ``stream`` completes."""
        self.collect()
        self._pending.append((stream.record_event(), tuple(references)))

    def release(self, callback=None, references=()) -> None:
        """Run ``callback`` and drop ``references`` once all work submitted so far has completed."""
        if self._destroyed:
            # The context, and every native object created from it, is already gone (interpreter exit).
            return
        self._releases.append(([event for event, _ in self._pending], callback, tuple(references)))
        self.collect()

    def collect(self) -> None:
        """Drop references of completed work and run releases whose work has finished."""
        self._pending = [(event, refs) for event, refs in self._pending if not event.is_complete]
        waiting = []
        for events, callback, references in self._releases:
            if all(event.is_complete for event in events):
                _run(callback)
            else:
                waiting.append((events, callback, references))
        self._releases = waiting

    def check_arrays(self, values) -> None:
        for value in _arrays(values):
            if value.device != self.device:
                raise ValueError("Array and OptiX resource must use the same CUDA device")

    def _shutdown(self) -> None:
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            for _, callback, _ in self._releases:
                _run(callback)
            self._releases.clear()
            self._pending.clear()
            self._accels.clear()
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


def _arrays(values):
    for value in values:
        if isinstance(value, wp.array):
            yield value
        elif isinstance(value, (tuple, list)):
            yield from _arrays(value)
        elif hasattr(value, "_cls") and hasattr(value._cls, "vars"):
            yield from _arrays([getattr(value, name) for name in value._cls.vars])


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
