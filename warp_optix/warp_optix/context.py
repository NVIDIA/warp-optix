# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifetime and device checks shared by the functional OptiX API."""

import logging
import weakref
from contextlib import contextmanager

import warp as wp

from warp_optix import native


class Context:
    """Own native objects and device buffers until close(); also a context manager.

    Use ``native_handle`` only for advanced native calls. It is borrowed and
    must not be destroyed separately.
    """

    def __init__(self, device=None, *, validation=False, log_level=4):
        self._closed = True
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("OptiX requires a CUDA device")
        self._objects = []
        self._pipelines = []
        self._accels = {}
        self._pending = []
        self.log_message_count = 0
        owner = weakref.ref(self)

        def log(level, tag, message):
            context = owner()
            if context is not None:
                context.log_message_count += 1
            logging.getLogger("warp_optix").log(
                {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO}.get(level, logging.DEBUG),
                "[%s] %s",
                tag,
                message,
            )

        options = native.DeviceContextOptions(
            logCallbackFunction=log,
            logCallbackLevel=log_level,
            validationMode=native.DEVICE_CONTEXT_VALIDATION_MODE_ALL
            if validation
            else native.DEVICE_CONTEXT_VALIDATION_MODE_OFF,
        )
        with wp.ScopedDevice(self.device):
            self._native = native.deviceContextCreate(self.device.context, options)
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The OptiX context is closed")

    @property
    def native_handle(self):
        self._check_open()
        return self._native

    def _device(self, device=None):
        self._check_open()
        if device is not None and wp.get_device(device) != self.device:
            raise ValueError("Device and context must use the same CUDA device")
        return self.device

    def _stream(self, stream=None):
        self._check_open()
        stream = wp.get_stream(self.device) if stream is None else stream
        if not isinstance(stream, wp.Stream):
            raise TypeError("stream must be a Warp Stream")
        if stream.device != self.device:
            raise ValueError("Stream and context must use the same CUDA device")
        return stream

    def _arrays(self, values):
        for value in values:
            if isinstance(value, wp.array):
                yield value
            elif isinstance(value, (tuple, list)):
                yield from self._arrays(value)
            elif hasattr(value, "_cls") and hasattr(value._cls, "vars"):
                yield from self._arrays([getattr(value, name) for name in value._cls.vars])

    def _check_arrays(self, values):
        for value in self._arrays(values):
            if value.device != self.device:
                raise ValueError("Array and context must use the same CUDA device")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._check_open()
        member = getattr(self._native, name)
        if not callable(member):
            return member

        def call(*args, **kwargs):
            self._check_open()
            with wp.ScopedDevice(self.device):
                result = member(*args, **kwargs)
            if name in (
                "moduleCreate",
                "moduleCreateFromPTX",
                "builtinISModuleGet",
                "programGroupCreate",
                "pipelineCreate",
            ):
                objects = (
                    result[0]
                    if name == "programGroupCreate"
                    else [result[0] if name in ("moduleCreate", "moduleCreateFromPTX") else result]
                )
                for obj in objects:
                    if not any(type(obj) is type(old) and obj == old for old in self._objects):
                        self._objects.append(obj)
            return result

        return call

    @contextmanager
    def _transaction(self):
        self._check_open()
        start = len(self._objects)
        with wp.ScopedDevice(self.device):
            try:
                yield
            except BaseException:
                # Failed operations may already have submitted GPU work. Keep
                # their traceback-owned buffers alive until it has finished.
                wp.synchronize_device(self.device)
                while len(self._objects) > start:
                    obj = self._objects.pop()
                    try:
                        obj.destroy()
                    except Exception:
                        logging.getLogger("warp_optix").exception("Failed to clean up an OptiX object")
                raise

    def _retain_launch(self, stream, inputs):
        self._pending = [(event, values) for event, values in self._pending if not event.is_complete]
        # Keep input arrays (including arrays inside Warp structs) alive until
        # this stream has finished reading launch parameters and tracing rays.
        self._pending.append((stream.record_event(), tuple(self._arrays(inputs))))

    def close(self):
        if self._closed:
            return
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            self._native.destroy()
        self._closed = True
        self._objects.clear()
        self._pending.clear()
        self._pipelines.clear()
        self._accels.clear()

    destroy = close

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def require_context(ctx):
    if not isinstance(ctx, Context):
        raise TypeError("Use a context returned by warp_optix.create_context()")
    ctx._check_open()
    return ctx
