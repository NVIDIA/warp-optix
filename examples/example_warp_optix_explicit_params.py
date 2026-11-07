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

"""Explicit schema, stage declarations, named SBT, and independent LaunchParams.

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


@wo.kernel(kind=wo.ProgramKind.RAYGEN, module="unique")
def raygen(values: wp.array2d[wp.uint32]):
    y, x = wp.tid()
    payload = Payload()
    wp.optix_trace(wp.uint64(0), wp.vec3(0.0), wp.vec3(0.0, 0.0, -1.0), 1.0, payload)
    values[y, x] = payload.value


@wo.kernel(kind=wo.ProgramKind.MISS, module="unique")
def miss(scale: wp.uint32):
    index = wp.optix_get_launch_index()
    wp.optix_set_payload(0, scale * index[1] + index[0])


def main() -> None:
    sbt = wo.SbtBuilder(["radiance"])
    sbt.set_miss("radiance", miss)
    with wo.Pipeline(raygen, sbt=sbt, params_type=LaunchParams, payload_words=1) as pipeline:
        outputs = []
        for scale in (100, 200):
            stream = wp.Stream(pipeline.device)
            values = wp.empty((4, 8), dtype=wp.uint32, device=pipeline.device)
            fields = LaunchParams()
            fields.scale = wp.uint32(scale)
            fields.values = values
            params = wo.LaunchParams(pipeline)
            params.assign(fields, stream=stream)
            wo.launch(pipeline, dim=values.shape, stream=stream, params=params)
            outputs.append((scale, values, stream))
        for scale, values, stream in outputs:
            wp.synchronize_stream(stream)
            np.testing.assert_array_equal(values.numpy(), scale * np.arange(4)[:, None] + np.arange(8))
        print(outputs[0][1].numpy())


if __name__ == "__main__":
    main()
