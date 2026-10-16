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

"""Explicit schema, stage declarations, named SBT, and independent launch buffers.

Produces the same output as example_warp_optix_kernels.py. Only the miss
program needs scale; the shared schema defines it independently of raygen.
"""

import numpy as np
import warp as wp

import warp_optix as wo


@wp.struct
class Payload:
    value: wp.uint32


@wp.struct
class LaunchParams:
    scale: wp.uint32
    values: wp.array2d[wp.uint32]


@wo.optix_kernel(wo.OptixKernelType.RAYGEN, module="unique")
def raygen(values: wp.array2d[wp.uint32]):
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


@wo.optix_kernel(wo.OptixKernelType.MISS, module="unique")
def miss(scale: wp.uint32):
    index = wp.optix_get_launch_index()
    wp.optix_set_payload_0(scale * index[1] + index[0])


def main() -> None:
    with wo.create_context() as ctx:
        sbt = wo.SbtBuilder(["radiance"])
        sbt.set_miss("radiance", miss)
        pipeline = wo.create_pipeline(ctx, raygen, sbt=sbt, params_type=LaunchParams, payload_words=1)
        outputs = []
        for scale in (100, 200):
            stream = wp.Stream(ctx.device)
            values = wp.empty((4, 8), dtype=wp.uint32, device=ctx.device)
            params = LaunchParams()
            params.scale = wp.uint32(scale)
            params.values = values
            buffer = wo.create_launch_params_buffer(pipeline)
            wo.write_launch_params(buffer, params, stream=stream)
            wo.launch(pipeline, dim=values.shape, params_buffer=buffer, stream=stream)
            outputs.append((scale, values, stream))
        for scale, values, stream in outputs:
            wp.synchronize_stream(stream)
            np.testing.assert_array_equal(values.numpy(), scale * np.arange(4)[:, None] + np.arange(8))
        print(outputs[0][1].numpy())


if __name__ == "__main__":
    main()
