# warp-optix

[![](docs/images/banner.webp)](docs/images/banner.webp)

_Rendered with the included path tracer: a chess set in glass and gold, [Glass of Water](https://benedikt-bitterli.me/resources/) by aXel (CC0), and [The Breakfast Room](https://benedikt-bitterli.me/resources/) by Wig42 ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), relit with a procedural sun). Scenes from Benedikt Bitterli's rendering resources._

warp-optix brings NVIDIA OptiX ray tracing to [Warp](https://github.com/NVIDIA/warp).
OptiX programs (ray generation, hit, miss, callables) are ordinary Warp kernels,
acceleration structures are built from Warp arrays, and `wo.launch` mirrors
`wp.launch`. A complete path tracer built on the same API is included.

> [!WARNING]
> **Alpha software:** warp-optix is under active development. The public API is
> experimental and may change between releases without backward compatibility.

## Installation

```bash
pip install warp-optix
# with the path tracer, viewer UI and USD loading:
pip install "warp-optix[pathtracing,ui,usd]"
```

Wheels are published for Python 3.12 on Linux x86-64 and Windows x64 and need
only an NVIDIA driver; no CUDA Toolkit or OptiX SDK is required. PyOptiX 9.1.1
is installed automatically, and the wheel bundles the OptiX SDK 9.1.0 headers
used for runtime compilation.

To work from a checkout (Python 3.12 and a C++17 compiler):

```bash
uv sync --extra pathtracing --extra ui --extra usd
uv run python examples/example_pathtracing.py --scene hairball
```

## Quick start

Trace a depth image of a triangle mesh with a single ray-generation kernel:

```python
import warp as wp
import warp_optix as wo


@wp.kernel(module="unique")
def render(scene: wp.uint64, depth: wp.array2d[float]):
    y, x = wp.tid()
    start = wp.vec3(float(x) / 64.0 - 1.0, float(y) / 64.0 - 1.0, 5.0)
    query = wo.accel_query_ray(scene, start, wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[y, x] = query.t if query.result else -1.0


accel = wo.Accel([wo.triangles(points, indices)])  # Warp device arrays, like wp.Mesh
pipeline = wo.Pipeline(render)
depth = wp.empty((128, 128), dtype=float)
wo.launch(pipeline, dim=depth.shape, inputs=[accel.id, depth])
```

Importing `warp_optix` registers the `wp.optix_*` device builtins. They are valid
only in kernels compiled into a `wo.Pipeline`.

## API

### Programs and pipelines

The raygen kernel's arguments are the launch parameters, and `wp.tid()` returns
the launch index, so `dim` means the same as in `wp.launch`. Every other program
(miss, closest-hit, any-hit, intersection, callables, exception) declares any
subset of the raygen arguments **by name, with identical types**:

```python
@wp.struct
class Payload:
    color: wp.vec3


@wp.kernel(module="unique")
def raygen(scene: wp.uint64, tint: wp.vec3, image: wp.array2d[wp.vec3]):
    y, x = wp.tid()
    payload = Payload()
    wp.optix_trace(scene, wp.vec3(float(x), float(y), 5.0), wp.vec3(0.0, 0.0, -1.0), 100.0, payload)
    image[y, x] = payload.color


@wp.kernel(module="unique")
def shade(tint: wp.vec3):  # binds raygen's `tint`
    payload = Payload()
    payload.color = tint * wp.optix_get_ray_tmax()
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def background():
    wp.optix_store_payload(Payload())


pipeline = wo.Pipeline(raygen, miss=[background], hit_groups=[wo.HitGroup(closest_hit=shade)], payload_words=3)
wo.launch(pipeline, dim=image.shape, inputs=[accel, wp.vec3(1.0, 0.5, 0.2), image])
```

Programs use `module="unique"` because kernels that call `wp.optix_*` builtins
cannot be compiled into an ordinary Warp module. `@wo.kernel(kind=wo.ProgramKind.MISS, module="unique")`
also declares the stage, which `wo.Pipeline` checks against the slot the kernel is used in.

- `wp.optix_trace(id, start, dir, max_t, payload, min_t, time, mask, flags, sbt_offset, sbt_stride, miss_index)`
  follows `wp.mesh_query_ray`. Trailing arguments may be omitted (not skipped by keyword).
  `wp.optix_traverse()` takes the same arguments and leaves the result in the hit object.
- `wo.Pipeline(raygen, *, miss, hit_groups, ray_types, sbt, callables, exception, exception_flags, payload_words, attribute_words, max_trace_depth, motion_blur, params_type, device)`
  compiles all programs into one OptiX module and builds the shader binding table.
  `hit_groups` entries are `wo.HitGroup(closest_hit=, any_hit=, intersection=, builtin=)`;
  `builtin=wo.PrimitiveType.*` selects OptiX's curve intersectors. The hit record for geometry
  `g` and ray type `r` is `hit_groups[g * ray_types + r]`.
- `wo.launch(pipeline, dim, inputs, device=None, stream=None, *, params=None)` writes the
  parameters and launches on `stream` without synchronizing the host. `wo.Accel` objects in
  `inputs` are passed as their `id`.
- Callables run with `wp.optix_direct_call(index)`; user exceptions are raised with
  `wp.optix_throw_exception(code, ...)` and enabled with `exception_flags=wo.ExceptionFlags.USER`.

For named ray types and explicit SBT layouts, use `wo.SbtBuilder`:

```python
sbt = wo.SbtBuilder(ray_types=("radiance", "shadow"))
sbt.set_miss("radiance", radiance_miss)
sbt.set_miss("shadow", shadow_miss)
metal = sbt.add_hit_group("metal", radiance=wo.HitGroup(closest_hit=metal_hit), shadow=wo.HitGroup(any_hit=metal_shadow))
pipeline = wo.Pipeline(raygen, sbt=sbt, payload_words=1)
offset = pipeline.sbt_layout.get_sbt_offset(metal)  # for wo.make_instance(..., sbt_offset=offset)
```

To share one parameter block between programs, or to own parameter storage,
pass `params_type=MyParams` (a `@wp.struct`) and use `wo.LaunchParams`:

```python
params = wo.LaunchParams(pipeline)
params.assign(my_params, stream=stream)
wo.launch(pipeline, dim=(height, width), stream=stream, params=params)
```

### Acceleration structures

`wo.Accel` mirrors `wp.Mesh`/`wp.Bvh`: it is built from inputs that keep their
Warp device arrays, and kernels see it as the `wp.uint64` `accel.id`.

| Input | Data |
| --- | --- |
| `wo.triangles(points, indices, flags=)` | `vec3` points and index triplets; a list of point arrays gives motion keys |
| `wo.curves(points, widths, indices, curve_type=)` | control points, per-point radii, first control point of each segment |
| `wo.aabbs(lowers, uppers, sbt_indices=)` | bounds of custom primitives, intersected by an `intersection` program |
| `wo.instances(records)` | `wp.array(dtype=wo.Instance)` |

`wo.Accel(inputs, *, flags=, compact=, motion=, device=, stream=)` builds on a Warp
stream. With `flags=wo.BuildFlags.ALLOW_UPDATE`, `accel.refit()` updates it to the
current contents of its inputs without allocating or synchronizing;
`accel.rebuild()` rebuilds in place. Instances are written on the GPU with
`wo.make_instance(pose, accel_id, instance_id, sbt_offset, mask, flags)`, where the
pose is a `wp.transform`, a `wp.mat44`, or a `wp.mat33` and translation:

```python
@wp.kernel
def write_instances(gas: wp.uint64, poses: wp.array[wp.transform], records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(poses[i], gas, i)


records = wp.empty(world_count, dtype=wo.Instance)
wp.launch(write_instances, dim=world_count, inputs=[gas.id, poses, records])
ias = wo.Accel([wo.instances(records)], flags=wo.BuildFlags.ALLOW_UPDATE)
# Each step: rewrite the records, then refit.
ias.refit()
```

For vertex motion blur, pass a list of vertex keys with
`motion=native.MotionOptions(...)`, build the pipeline with `motion_blur=True`,
and trace with a ray time.

### Ray queries

`wo.accel_query_ray(id, start, dir, max_t)` and `wo.accel_query_ray_anyhit(...)`
mirror `wp.mesh_query_ray`/`wp.mesh_query_ray_anyhit`, so a raygen-only pipeline
can query geometry without hit programs. The result has `result`, `t`, `u`, `v`,
`instance_id` and `primitive_index`.

### Device builtins

All OptiX device functions use Warp naming with an `optix_` prefix, for example
`wp.optix_get_instance_index()`, `wp.optix_get_world_ray_direction()` and
`wp.optix_transform_point_world_to_object(p)`:

- payloads: `wp.optix_set_payload(i, v)`/`wp.optix_get_payload(i)`, or whole structs with
  `wp.optix_load_payload()`/`wp.optix_store_payload()`;
- intersection programs: `wp.optix_report_intersection(t, kind, attributes...)` and
  `wp.optix_get_attribute(i)`;
- hit data: triangle barycentrics and vertices, curve control points and parameter,
  transform lists, and `wp.optix_get_object_to_world_matrix()`;
- cooperative vectors: matrix multiply, activation and reduction functions for FP16, FP8 and int8.

Flags are `IntEnum`/`IntFlag` types usable on the host and in kernels:
`wo.RayFlags`, `wo.HitKind`, `wo.BuildFlags`, `wo.GeometryFlags`, `wo.InstanceFlags`,
`wo.PrimitiveType`, `wo.ExceptionFlags`.

### Devices, streams and lifetime

As with Warp's CUDA context, each device has one implicit OptiX context,
configured through `wo.config` (`validation`, `log_level`). Functions take
`device=` and `stream=` like Warp: by default the current device and its current
stream.

`wo.Accel`, `wo.Pipeline` and `wo.LaunchParams` release themselves when the last
reference is dropped, or explicitly with `close()`/`with`. As with `wp.array`,
device memory is freed in stream order, so work already submitted can still
use it; native OptiX objects are destroyed once their submitted launches finish.
Keep a `wo.Accel` alive while work that reads its `id` may still be submitted.
An instance structure keeps its children alive.

OptiX work cannot be recorded into CUDA graphs: `wo.launch`, `wo.Accel`,
`refit()`/`rebuild()` and `wo.Pipeline` raise during stream capture. Capture
ordinary Warp work in graphs and submit OptiX work between them on the same stream.

### Native interface

`from warp_optix import native` exposes PyOptiX plus the supplemental bindings
(cooperative vectors, temporal denoiser fields), and `wo.get_context(device).native`
is a borrowed PyOptiX `DeviceContext`. Native objects need explicit destruction.
See [PyOptiX dependency maintenance](docs/pyoptix-dependency.md).

## Path tracer

`warp_optix.pathtracing` is a PBR path tracer built on this API, with triangle
meshes, native curves, glTF and USD loading, OptiX temporal denoising and 2x
upscaling, and an interactive OpenGL viewer:

```python
from warp_optix.pathtracing import PathTracer

api = PathTracer(denoiser="auto")
api.initialize()
api.load_scene_from_gltf("scene.glb")
api.render_frame()
image = api.get_frame()
```

See the [path tracer guide](docs/pathtracing.md).

## Examples

Runnable programs are in [`examples/`](examples/README.md), with images of each:

```bash
uv run python examples/example_minimal.py                       # smallest pipeline
uv run python examples/example_raytracing.py --live             # Cornell box ray tracer
uv run python examples/example_pathtracing.py --scene hairball  # chess, hairball, arrowball, contact_lines
uv run python examples/example_pathtracing.py --usd scene.usd   # or --gltf scene.glb
```

`example_curves.py`, `example_mixed_geometry.py` and `example_motion_blur.py` cover
built-in curves, custom primitives with GPU-written instances, and motion blur.

## Tests

```bash
uv run pytest warp_optix/tests
uv run pytest examples/neural_texture
```

Tests can be selected with the markers in `pyproject.toml` (`gpu`, `slow`, `usd`,
`render`, `training`), for example `-m "not slow"`.

## License

Warp integration uses [Apache-2.0](LICENSE.md); `src/extension.cpp`,
`src/pyoptix_compat.h` and `CMakeLists.txt` use [BSD-3-Clause](LICENSE.txt). The
bundled OptiX headers retain their [NVIDIA SDK terms](OptiX_LICENSE.txt);
redistribution of these unmodified headers in the warp-optix package has been
approved. See [third-party notices](THIRD_PARTY_NOTICES.txt).

## Contributing and security

This project is currently not accepting contributions; see
[CONTRIBUTING.md](CONTRIBUTING.md). Report security vulnerabilities privately
as described in [SECURITY.md](SECURITY.md).
