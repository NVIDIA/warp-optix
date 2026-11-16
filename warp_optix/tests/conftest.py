# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys

import pytest
import warp as wp


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail at startup if CUDA is unavailable instead of skipping GPU tests.",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda") and not wp.is_cuda_available():
        raise pytest.UsageError("--require-cuda needs an NVIDIA GPU and a working CUDA driver")


@wp.kernel(module="unique")
def _wait_for_gate(gate: wp.array[wp.int32], ticket: int, limit: int):
    i = int(0)
    while wp.atomic_add(gate, 0, 0) < ticket and i < limit:
        i += 1


class StreamGate:
    """Holds CUDA streams until :meth:`open`, so a test can inspect state while its launches are pending."""

    # Caps the wait at a few seconds, so a test that never opens the gate cannot hang.
    LIMIT = 20_000_000

    def __init__(self, device):
        self.device = device
        self._gate = wp.zeros(1, dtype=wp.int32, device=device)
        self._value = wp.empty(1, dtype=wp.int32, device="cpu", pinned=True)
        self._ticket = 0
        # Warp frees device memory on the legacy default stream, which waits for a held stream, and
        # every Warp stream waits for the default stream. Releasing therefore needs a non-blocking stream.
        self._cuda = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        self._stream = ctypes.c_void_p()
        with wp.ScopedDevice(device):
            self._check(self._cuda.cuStreamCreate(ctypes.byref(self._stream), 1))  # CU_STREAM_NON_BLOCKING
        # Loading a CUDA module synchronizes the device, so load this one before any stream is held.
        wp.launch(_wait_for_gate, 1, inputs=[self._gate, 0, 0], device=device)
        wp.synchronize_device(device)

    @staticmethod
    def _check(result):
        if result != 0:
            raise RuntimeError(f"CUDA driver error {result}")

    def block(self, stream: wp.Stream) -> wp.Event:
        """Hold ``stream`` until :meth:`open`; work queued on it afterwards stays pending."""
        # Each hold waits for a new ticket, so the gate is never reset and open() cannot be lost.
        self._ticket += 1
        wp.launch(_wait_for_gate, 1, inputs=[self._gate, self._ticket, self.LIMIT], stream=stream)
        return stream.record_event()

    def open(self) -> None:
        """Release every held stream."""
        self._value.numpy()[0] = self._ticket
        with wp.ScopedDevice(self.device):
            self._check(
                self._cuda.cuMemcpyHtoDAsync_v2(
                    ctypes.c_uint64(self._gate.ptr), ctypes.c_void_p(self._value.ptr), ctypes.c_size_t(4), self._stream
                )
            )
            self._check(self._cuda.cuStreamSynchronize(self._stream))


@pytest.fixture(scope="session")
def _stream_gate():
    return StreamGate(wp.get_device("cuda"))


@pytest.fixture
def gate(_stream_gate):
    yield _stream_gate
    _stream_gate.open()
    wp.synchronize()
