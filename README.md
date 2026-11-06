# warp-optix

Warp ray tracing with PyOptiX 9.1.
Write ray-generation, hit, miss, and callable programs as Warp kernels, or use
the included path tracer. Applications import `warp_optix`. Installation brings
in `pyoptix==9.1.1` for the standard OptiX bindings and builds a small extension
for cooperative vectors and missing binding fields and fixes.

> [!WARNING]
> **Alpha software:** warp-optix is under active development and is not yet
> feature-complete or production-ready. The public API is experimental and may
> change between releases without backward compatibility. Features and behavior
> may change, and bugs are expected.

## Installation

Releases publish wheels for Python 3.12 on Linux x86-64 and Windows x64 to PyPI
(`pip install warp-optix`); they need only an NVIDIA driver. To build from source
you need Python 3.12 and a C++17 compiler. No CUDA Toolkit or OptiX SDK is
required: the build fetches the OptiX headers and takes `cuda.h` from the
`nvidia-cuda-runtime-cu12` build requirement.

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
from warp_optix.pathtracing import PathTracer
```

For custom ray tracers, OptiX programs are ordinary Warp kernels and launches
mirror `wp.launch`:

```python
@wp.kernel(module="unique")
def render(scene: wp.uint64, image: wp.array2d[float]):
    y, x = wp.tid()
    start = wp.vec3(float(x), float(y), 10.0)
    query = wo.accel_query_ray(scene, start, wp.vec3(0.0, 0.0, -1.0), 100.0)
    image[y, x] = query.t if query.result else 0.0


accel = wo.Accel([wo.triangles(points, indices)])  # Warp device arrays, like wp.Mesh
pipeline = wo.Pipeline(render)
wo.launch(pipeline, dim=(480, 640), inputs=[accel.id, image])
```

The raygen kernel's arguments are the launch parameters. Like Warp's CUDA context,
the OptiX context is implicit (one per device, configured through `wo.config`).
`wo.Accel`, `wo.Pipeline`, and `wo.LaunchParams` release themselves when the last
reference is dropped, or explicitly with `close()`/`with`. As with `wp.Mesh` and
`wp.array`, device memory is freed in stream order; a pipeline's native OptiX objects
are destroyed once its submitted launches have finished. Instances are written
with `wo.make_instance()` and refit with `accel.refit()`. Initial instance builds
validate child ids with a host readback; compaction also synchronizes. Refits and
launches remain asynchronous on the selected Warp stream.

SBT records are built automatically. For named ray types and explicit hit-group
layouts, pass a `wo.SbtBuilder` to `wo.Pipeline(..., sbt=builder)`.

See the [API guide](warp_optix/README.md) and runnable [examples](examples_warp/README.md).

## Advanced native interface

```python
from warp_optix import native
```

`native` exposes PyOptiX's API plus the supplemental bindings. Native
objects use explicit destruction. `wo.get_context(device).native` provides
borrowed access to a device's PyOptiX `DeviceContext`. Do not destroy that
borrowed handle separately.
Standard objects are the dependency's actual `optix` types; both packages can
be imported in either order, and importing warp-optix leaves the `optix` module
unmodified. Supplemental operations are module functions that take the context
first: `native.coopVecMatrixComputeSize(context, ...)`,
`native.coopVecMatrixConvert(context, ...)`, and
`native.getCoopVecProperty(context)`. `native.moduleCreate(context, ...)` and
`native.builtinISModuleGet(context, ...)` forward module compile metadata
(`boundValues`, `payloadTypes`) that the PyOptiX 9.1.1 methods ignore. PyOptiX
9.1.1 also ignores the `callablesModuleCC` argument of `ProgramGroupDesc`; set
the attribute after construction. Keep objects on their owning OptiX/CUDA
context and obey their lifetimes.

PyOptiX 9.1.1 has no public native-handle API. The extension therefore uses a
small, version-pinned pybind11 bridge and checks shared type identities and
sizes before accessing them. Builds reject OptiX headers that differ from the
PyOptiX dependency pin; imports reject mismatched compiled SDK versions before
using upstream objects. Both extensions must use a compatible pybind11
major version and C++ ABI. If the import reports an ABI mismatch, rebuild both
packages with pybind11 3.x. See [dependency maintenance](docs/pyoptix-dependency.md).

The distribution version is 0.2.0; the PyOptiX version is 9.1.1 (OptiX SDK 9.1.0).

## Tests and dependency maintenance

Install feature extras and pytest, then run the test suite:

```powershell
python -m pytest warp_optix/tests
python -m pytest examples_warp/neural_texture
```

Tests can be selected with the markers registered in `pyproject.toml`
(`gpu`, `slow`, `usd`, `render`, `training`), for example `-m "not slow"`.

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
`CMakeLists.txt` use [BSD-3-Clause](LICENSE.txt); NVIDIA-authored Warp integration uses
[Apache-2.0](LICENSE.md). OptiX headers retain their [NVIDIA SDK terms](OptiX_LICENSE.txt).
The aggregate package metadata identifies all three licenses.

This project is currently not accepting contributions.
We plan to accept outside contributions in the future. See
[CONTRIBUTING.md](CONTRIBUTING.md) for updates.

## Security

See [SECURITY.md](SECURITY.md) for instructions on privately reporting potential
security vulnerabilities to NVIDIA PSIRT.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction; on Blackwell with driver 595.99,
OptiX launches and acceleration-structure builds are not recorded into the
captured graph at all. `wo.launch()`, `wo.Accel`, `refit()`/`rebuild()`, and
`wo.Pipeline` reject an active capture before submitting native work.
Capture ordinary Warp work in graphs and submit OptiX work between those
graphs on the same stream. CUDA stream ordering is supported.
