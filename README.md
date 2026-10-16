# warp-optix

Warp ray tracing with PyOptiX 9.1.
Write ray-generation, hit, miss, and callable programs as Warp kernels, or use
the included path tracer. Applications import `warp_optix`. Installation brings
in `pyoptix==9.1.0` for the standard OptiX bindings and builds a small extension
for cooperative vectors, and missing binding fields and fixes.

## Installation

Build from the repository root with Python 3.10+, a CUDA Toolkit and a C++17
compiler. GPU execution requires a compatible NVIDIA driver. OptiX headers are fetched automatically.

The wheel includes unmodified OptiX SDK 9.1.0 headers under `warp_optix/include`
for runtime kernel compilation, so a separate OptiX SDK installation is not
required. The headers retain their NVIDIA SDK terms in `OptiX_LICENSE.txt`,
identified in package metadata as `LicenseRef-NVIDIA-OptiX`. Redistribution of
these unmodified headers in the public warp-optix pip package has been approved.

```powershell
pip install .
# Editable development with viewer dependencies:
pip install -e ".[pathtracing,ui,usd]"
```

This builds a platform wheel containing the Python package and supplemental
extension. PyOptiX is a separate dependency, installed automatically. Extras
are `pathtracing`, `ui`, `usd`, and `training`.

For a repository-local uv environment:

```bash
uv sync --python 3.12 --no-default-groups --extra pathtracing --extra ui
uv run --no-sync python examples_warp/example_warp_optix_pathtraced_hairball.py
```

## Public API

```python
import warp as wp
import warp_optix as wo
from warp_optix.pathtracing import PathTracerAPI
```

For custom ray tracers, a context owns geometry and pipelines:

```python
# Geometry arrays, PTX, entry kernels, and launch parameters are application inputs.
with wo.Context(device="cuda:0") as context:
    geometry = context.create_triangle_gas(vertices, indices)
    pipeline = context.create_pipeline(
        ptx, raygen, miss, closest_hit,
        num_payload_values=3, num_attribute_values=2,
    )
    # Store geometry.handle in your Warp launch parameters.
    pipeline.launch(params_buffer, width=640, height=480)
```

The context retains geometry buffers, modules, SBTs, and pipelines. Closing
it synchronizes the CUDA device and releases resources. Keep arrays referenced
by launch parameters alive until GPU work completes. Launches are asynchronous
on Warp's current stream, or on an explicitly supplied `wp.Stream`; geometry
builds and refits are synchronous.

See the [API guide](warp_optix/README.md), [migration notes](docs/migration-0.2.md),
and runnable [examples](examples_warp/README.md).

## Advanced native interface

```python
from warp_optix import native
```

`native` exposes PyOptiX's API plus the supplemental bindings. Native
objects use explicit destruction; `Context.native_handle` and
`Pipeline.native_handle` are borrowed and must not be destroyed by callers.
Standard objects are the dependency's actual `optix` types; both packages can
be imported in either order. The supplement adds methods and fields to those
types and repairs module-metadata and continuation-callable handling. Keep
objects on their owning OptiX/CUDA context and obey their lifetimes.

PyOptiX 9.1.0 has no public native-handle API. The extension therefore uses a
small, version-pinned pybind11 bridge and checks shared type identities and
sizes before accessing them. Builds reject OptiX headers that differ from the
PyOptiX dependency pin; imports reject mismatched compiled SDK versions before
attaching bindings. Both extensions must use a compatible pybind11
major version and C++ ABI. If the import reports an ABI mismatch, rebuild both
packages with pybind11 3.x. See [dependency maintenance](docs/pyoptix-dependency.md).

The distribution version is 0.2.0; the binding version is 9.1.0.

## Tests and dependency maintenance

Install feature extras and pytest, then run the test suite:

```powershell
python -m pytest warp_optix/tests
```

The suite includes regression coverage for the dependency and supplemental
bindings. The dependency is maintained in
[NVIDIA/otk-pyoptix](https://github.com/NVIDIA/otk-pyoptix). See
[maintenance instructions](docs/pyoptix-dependency.md). Extracted binding additions
retain their BSD license; Warp integration sources use Apache 2.0. OptiX headers retain their
NVIDIA license terms. License texts accompany the distribution.

Runnable samples are in `examples_warp/`.

## Licenses and contributions

See [third-party notices](THIRD_PARTY_NOTICES.txt) for component licenses and
source provenance. `src/extension.cpp`, `src/pyoptix_compat.h`, and
`CMakeLists.txt` use BSD-3-Clause; NVIDIA-authored Warp integration uses
Apache-2.0. The aggregate package metadata also identifies the OptiX SDK terms.

This project is currently not accepting contributions.
We plan to accept outside contributions in the future. See
[CONTRIBUTING.md](CONTRIBUTING.md) for updates.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction. `Pipeline.launch` rejects an
active capture before submitting native work. Capture ordinary Warp work in
graphs and submit OptiX launches between those graphs on the same stream.

When `enable_cuda_graphs` is requested for OptiX rendering, the renderer uses
direct asynchronous launches and logs an explanation. CUDA stream ordering is supported.
