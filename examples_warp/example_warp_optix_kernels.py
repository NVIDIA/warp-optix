# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal OptiX pipeline built from ordinary Warp kernels.

The raygen kernel's arguments become the launch parameters. The miss kernel
binds the raygen argument ``scale`` by name. Tracing against a null traversable
always invokes the miss program, so no acceleration structure is needed.
"""

import numpy as np
import warp as wp

import warp_optix as wo


@wp.struct
class Payload:
    value: wp.uint32


@wp.kernel(module="unique")
def raygen(scale: wp.uint32, values: wp.array2d[wp.uint32]):
    y, x = wp.tid()
    payload = Payload()
    wp.optix_trace(
        wp.uint64(0),
        wp.vec3(0.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.0,
        1.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )
    values[y, x] = payload.value


@wp.kernel(module="unique")
def miss(scale: wp.uint32):
    index = wp.optix_get_launch_index()
    wp.optix_set_payload_0(scale * index[1] + index[0])


def main() -> None:
    ctx = wo.create_context()
    pipeline = wo.create_pipeline(ctx, raygen, miss=[miss], payload_words=1)
    values = wp.zeros((4, 8), dtype=wp.uint32)
    wo.launch(pipeline, dim=values.shape, inputs=[100, values])
    result = values.numpy()
    print("Miss program output (100 * row + column):")
    print(result)
    np.testing.assert_array_equal(result, 100 * np.arange(4)[:, None] + np.arange(8))


if __name__ == "__main__":
    main()
