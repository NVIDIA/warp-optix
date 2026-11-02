# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from warp_optix.pathtracing.hdr_loader import load_hdr


@pytest.mark.parametrize("pixel_data", [b"\x02\x02\x00\x08\x00", b"\x02\x02\x00\x08\x89\x01"])
def test_hdr_loader_rejects_malformed_rle(tmp_path, pixel_data):
    path = tmp_path / "invalid.hdr"
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n" + pixel_data)

    with pytest.raises(ValueError, match="pixel data"):
        load_hdr(path)


@pytest.mark.parametrize("resolution", [b"-Y 1 +X 40000", b"-Y 20000 +X 20000"])
def test_hdr_loader_rejects_oversized_dimensions(tmp_path, resolution):
    path = tmp_path / "huge.hdr"
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + resolution + b"\n")

    with pytest.raises(ValueError, match="exceeds"):
        load_hdr(path)
