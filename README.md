# warp-optix

[![Glass and gold chess, a glass of water, and a sunlit breakfast room, all rendered with warp-optix](docs/images/banner.webp)](docs/images/banner.webp)

warp-optix brings NVIDIA OptiX ray tracing to [Warp](https://github.com/NVIDIA/warp). Scenes are
built from Warp arrays, OptiX programs are written as ordinary Warp kernels, and launches work like
`wp.launch`. A physically based path tracer built on the same API is included.

> [!WARNING]
> warp-optix is alpha software. The API will change between releases, sometimes without a
> deprecation path.

## Install

```bash
pip install warp-optix
pip install "warp-optix[pathtracing,ui,usd]"   # adds the path tracer, viewer, and USD loading
```

Wheels are available for Python 3.12 on Linux x86-64 and Windows x64 and require only an NVIDIA
driver. Neither the CUDA Toolkit nor the OptiX SDK is needed: PyOptiX 9.1.1 is installed as a
dependency, and the wheel includes the OptiX 9.1.0 headers that kernels compile against.

Building from a checkout also requires a C++17 compiler:

```bash
uv sync --extra pathtracing --extra ui --extra usd
uv run python examples/example_pathtracing.py --scene hairball
```

## Overview

warp-optix can be used at four levels. Most applications need only the first or second.

1. **Ray queries in a Warp kernel.** `wo.accel_query_ray()` is called from a kernel, in the same way
   as `wp.mesh_query_ray()`. It needs a single kernel and no other programs, and suits depth images,
   visibility tests, sensors, and collision rays.
2. **Custom OptiX programs.** Ray generation, closest-hit, any-hit, miss, intersection, and callable
   programs are written as Warp kernels, and OptiX invokes them as rays traverse the scene.
3. **The path tracer.** `warp_optix.pathtracing` renders glTF, USD, or procedural scenes with
   physically based materials, denoising, and an interactive viewer, without any shader code.
4. **Native OptiX.** `warp_optix.native` exposes PyOptiX directly for functionality the other levels
   do not cover.

All four levels share the same scene representation, which is described first. The
[API reference](docs/api.md) lists exact signatures and every device builtin.

## Building a scene

OptiX traces rays against *acceleration structures*: BVHs that it builds and owns on the GPU. In
warp-optix these are `wo.Accel` objects, which work much like `wp.Mesh`: they are built from Warp
arrays, and kernels refer to them by a `wp.uint64` id.

```python
import numpy as np
import warp as wp

import warp_optix as wo

wp.init()

points = wp.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)], dtype=wp.vec3)
indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
quad = wo.Accel([wo.triangles(points, indices)])
```

Four kinds of build input are available:

| Input | What goes in |
| --- | --- |
| `wo.triangles(points, indices)` | `vec3` points and index triplets |
| `wo.curves(points, widths, indices, curve_type=)` | control points, a radius per point, and the first point of each segment (round linear, B-spline, Catmull-Rom, Bézier, or flat ribbon curves) |
| `wo.aabbs(lowers, uppers)` | a bounding box per custom primitive, like `wp.Bvh(lowers, uppers)`, with a user-supplied intersection program |
| `wo.instances(records, children)` | `wo.Instance` records that place other structures in the world |

Instances place many copies of a structure in a scene, or many worlds in one scene. Instance
records are Warp structs, so a single kernel launch can write thousands of them directly from
simulation state. `wo.instances` also receives the structures that the records reference. The build
therefore never reads the records back, and the referenced structures stay alive for as long as the
instance structure needs them:

```python
@wp.kernel
def place(gas: wp.uint64, heights: wp.array[float], records: wp.array[wo.Instance]):
    i = wp.tid()
    pose = wp.transform(wp.vec3(3.0 * float(i), 0.0, heights[i]), wp.quat_identity())
    records[i] = wo.make_instance(pose, gas, i)  # instance id = i


heights = wp.array([0.0, 1.0, 2.0], dtype=float)
records = wp.empty(3, dtype=wo.Instance)
wp.launch(place, dim=3, inputs=[quad.id, heights, records])
scene = wo.Accel([wo.instances(records, [quad])], flags=wo.BuildFlags.ALLOW_UPDATE)
```

When geometry moves, update the records or points and call `scene.refit()`. A refit neither
allocates memory nor waits on the host, so it can run every simulation step. Because trace quality
degrades over many refits, call `scene.rebuild()` periodically. Static geometry can be compacted
instead: `compact=True` reduces a structure to about a third of its size, and `wo.compact(accels)`
compacts a batch of structures with a single synchronization.

For vertex motion blur, pass `wo.triangles` one point array per time step, together with OptiX
motion options:

```python
from warp_optix import native

later = wp.array(points.numpy() + np.float32([0.0, 0.5, 0.0]), dtype=wp.vec3)
motion = native.MotionOptions(numKeys=2, flags=native.MOTION_FLAG_NONE, timeBegin=0.0, timeEnd=1.0)
blurred = wo.Accel([wo.triangles([points, later], indices)], motion=motion)
```

Dynamic tessellation can use OptiX grid clusters through the owned API:

```python
# CUDA arrays: vec3 points, vec2i grid sizes (in quads), int32 vertex offsets.
surface = wo.ClusterAccel(points, grid_sizes, vertex_offsets)
pipeline = wo.Pipeline(render, clustered_geometry=True)
wo.launch(pipeline, dim=(height, width), inputs=[surface, image])
surface.rebuild()  # after changing the device arrays
```

`ClusterAccel` manages templates, CLAS/GAS builds, scratch memory and lifetimes.
It can also be instanced with `wo.instances`. Ray queries expose `hit.cluster_id`;
there are no compiler workarounds in user code. See the
[cluster example](examples/example_clusters.py) for an animated parametric knot,
generated directly in Warp. It uses the standard installation, with no additional
geometry library or build step.

## 1. Ray queries in a Warp kernel

Ray queries are the simplest way to trace a scene. `wo.accel_query_ray` works like
`wp.mesh_query_ray`: given a ray, it returns whether anything was hit, at what distance, and what
was hit.

```python
@wp.kernel(module="unique")
def depth_image(scene: wp.uint64, depth: wp.array2d[float]):
    y, x = wp.tid()
    origin = wp.vec3(float(x) / 32.0 - 1.0, float(y) / 32.0 - 1.0, 5.0)
    hit = wo.accel_query_ray(scene, origin, wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[y, x] = hit.t if hit.result else -1.0


pipeline = wo.Pipeline(depth_image)
depth = wp.empty((64, 64), dtype=float)
wo.launch(pipeline, dim=depth.shape, inputs=[scene, depth])
```

Three details differ from plain Warp:

- The kernel becomes a `wo.Pipeline` and runs with `wo.launch`, which takes the same `dim`, `inputs`,
  and `stream` as `wp.launch`. `wp.tid()` works as usual.
- `module="unique"` gives the kernel its own Warp module. Kernels that use OptiX builtins cannot be
  compiled together with ordinary kernels.
- A `wo.Accel` may be passed directly in `inputs`. It is passed to the kernel as its id and kept
  alive until the launch completes.

In addition to `t`, the result contains the instance id, primitive index, and barycentrics (`u`,
`v`). `wo.accel_query_ray_anyhit` only reports whether anything was hit and stops at the first hit,
which makes it the efficient choice for shadow and visibility rays.

## 2. Custom OptiX programs

OptiX programs are needed for per-surface shading, custom primitives, and rays that bounce. The ray
generation kernel calls `wp.optix_trace`, OptiX traverses the scene, and then runs the closest-hit
program for the nearest hit, or the miss program if nothing was hit. The programs exchange data
through a *payload*, a small Warp struct carried with the ray.

```python
@wp.struct
class Payload:
    color: wp.vec3


@wp.kernel(module="unique")
def camera(scene: wp.uint64, tint: wp.vec3, image: wp.array2d[wp.vec3]):
    y, x = wp.tid()
    origin = wp.vec3(float(x) / 32.0 - 1.0, float(y) / 32.0 - 1.0, 5.0)
    payload = Payload()
    wp.optix_trace(scene, origin, wp.vec3(0.0, 0.0, -1.0), 100.0, payload)
    image[y, x] = payload.color


@wp.kernel(module="unique")
def shade(tint: wp.vec3):
    payload = Payload()
    payload.color = tint * (1.0 / wp.optix_get_ray_tmax())
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def sky():
    payload = Payload()
    payload.color = wp.vec3(0.4, 0.6, 1.0)
    wp.optix_store_payload(payload)


sbt = wo.SbtLayout(miss=sky, hit_groups={"surface": wo.HitGroup(closest_hit=shade)})
pipeline = wo.Pipeline(camera, sbt=sbt, payload_words=3)
image = wp.empty((64, 64), dtype=wp.vec3)
wo.launch(pipeline, dim=image.shape, inputs=[scene, wp.vec3(1.0, 0.5, 0.2), image])
```

The ray generation kernel's arguments are the launch parameters, and every other program may
declare any of them **by name**; `shade` declares only `tint`. Inside a program, the `wp.optix_*`
builtins describe the hit: its distance (`wp.optix_get_ray_tmax()`), the instance and primitive that
were hit, the barycentrics, the face orientation, and transforms between object and world space. The
[API reference](docs/api.md#device-builtins) lists all of them.

### The shader binding table

`wo.SbtLayout` determines which program runs for each hit. OptiX calls this the shader binding
table, or SBT. A layout names the *ray types* (most renderers use a camera ray and a shadow ray) and
the *hit groups*, one per kind of surface. Each hit group can use different programs for each ray
type:

```python
@wp.kernel(module="unique")
def blocked():
    wp.optix_set_payload(0, wp.uint32(0))


@wp.kernel(module="unique")
def unblocked():
    wp.optix_set_payload(0, wp.uint32(1))


lights = wo.SbtLayout(
    ray_types=("camera", "shadow"),
    miss={"camera": sky, "shadow": unblocked},
    hit_groups={
        "glass": {"camera": wo.HitGroup(closest_hit=shade), "shadow": wo.HitGroup(closest_hit=blocked)},
        "metal": {"camera": wo.HitGroup(closest_hit=shade), "shadow": wo.HitGroup(closest_hit=blocked)},
    },
)
print(lights.describe())
```

The layout provides the indices OptiX needs, so records never have to be counted by hand.
`lights.get_sbt_offset("metal")` is the SBT offset passed to `wo.make_instance` for each metal
instance, and `lights.ray_type_index("shadow")` is the SBT offset and miss index of a shadow
`wp.optix_trace`, with `lights.trace_stride` as its stride. When one program serves every ray type,
a single `wo.HitGroup` can be given instead of a mapping, as in the first example.

### Per-surface data

A hit group can carry a `@wp.struct`, so that one closest-hit program shades many materials
instead of requiring a program per material. The program reads the data with
`wp.optix_load_sbt_data`:

```python
@wp.struct
class Material:
    color: wp.vec3


@wp.kernel(module="unique")
def shade_material():
    material = Material()
    wp.optix_load_sbt_data(material)
    payload = Payload()
    payload.color = material.color
    wp.optix_store_payload(payload)


red, blue = Material(), Material()
red.color, blue.color = wp.vec3(1.0, 0.1, 0.1), wp.vec3(0.1, 0.1, 1.0)
painted = wo.Pipeline(
    camera,
    sbt=wo.SbtLayout(
        miss=sky,
        hit_groups={
            "red": wo.HitGroup(closest_hit=shade_material, data=red),
            "blue": wo.HitGroup(closest_hit=shade_material, data=blue),
        },
    ),
    payload_words=3,
)
red.color = wp.vec3(1.0, 0.5, 0.0)
painted.update_sbt_data()  # uploads the new color; no rebuild
```

Miss and callable programs can carry data as well, given as `wo.Record(kernel, data=...)`.

### Launching in a loop

`wo.launch` costs about 13 µs of host time per call on an RTX PRO 6000, compared with roughly
5.5 µs for the OptiX launch itself. In a simulation loop, a launch can be recorded once and
replayed, as with `wp.launch(..., record_cmd=True)`:

```python
launch = wo.launch(pipeline, dim=image.shape, inputs=[scene, wp.vec3(1.0, 0.5, 0.2), image], record_cmd=True)
for step in range(3):
    launch.set_param_by_name("tint", wp.vec3(1.0, 0.5, 0.1 * step))
    launch.launch()
```

### More program kinds

- **Custom primitives.** Build the geometry with `wo.aabbs` and give its hit group an
  `intersection` program that calls `wp.optix_report_intersection(t, kind, ...)`.
- **Built-in curves.** Use `wo.HitGroup(builtin=wo.PrimitiveType.ROUND_CUBIC_BEZIER, ...)`. OptiX
  specializes its curve intersector on the structure's build flags, so pass `build_flags=` for curves
  built with flags other than the default `PREFER_FAST_TRACE`.
- **Callables.** `wo.Pipeline(..., callables=[a, b])` makes `a` and `b` callable from any program
  with `wp.optix_direct_call(0)` and `wp.optix_direct_call(1)`. Continuation callables are declared
  with `@wo.kernel(kind=wo.ProgramKind.CONTINUATION_CALLABLE)`.
- **Exceptions.** `exception=` adds an exception program, and `exception_flags=` enables OptiX's
  stack-overflow and trace-depth checks for debugging.
- **A shared parameter struct.** By default, the ray generation kernel's arguments define the launch
  parameters. `params_type=MyParams` uses a `@wp.struct` instead, so that hit programs can read fields
  the ray generation kernel does not declare.

[`examples/example_mixed_geometry.py`](examples/example_mixed_geometry.py) combines most of these
features: triangles, curves, and six analytic shapes under GPU-written instances, with a named SBT.

## 3. The path tracer

`warp_optix.pathtracing` is a physically based path tracer built on the API above, for rendering
without writing OptiX programs. It loads glTF and USD, renders native curves for hair and fur,
denoises with OptiX (with optional 2x upscaling), and includes an OpenGL viewer.

```python
from warp_optix.pathtracing import PathTracer

tracer = PathTracer(width=320, height=240, denoiser="none")
tracer.initialize()
tracer.scene.create_cornell_box()
tracer.build_scene()
for _ in range(16):
    tracer.render_frame()
frame = tracer.get_frame()  # (240, 320, 4) float RGBA
tracer.close()
```

Scenes can be loaded with `load_scene_from_gltf(path)` or `load_scene_from_usd(path)`, or built
from meshes, curves, spheres, and materials. Dynamic meshes and curves are updated from device
arrays, so a simulation can drive them every frame without host copies. The [path tracer
guide](docs/pathtracing.md) covers materials, denoising, the viewer, and USD.

## 4. Native OptiX

`warp_optix.native` is PyOptiX 9.1.1 with a few additions: cooperative-vector matrix conversion,
missing enum values, and fixes for module compile options. The OptiX context of a device is
available through `wo.get_context`:

```python
context = wo.get_context("cuda").native  # borrowed PyOptiX DeviceContext; do not destroy it
print(native.version(), native.getCoopVecProperty(context))
```

Native objects follow PyOptiX's conventions: they are destroyed explicitly and must stay on their
own device. Importing warp-optix does not modify the `optix` module, so other PyOptiX code in the
same process is unaffected.

## Usage notes

**Lifetime.** `wo.Accel` and `wo.Pipeline` behave like `wp.array`. They are released when the last
reference is dropped, or immediately with `close()` or a `with` block. Memory is freed in stream
order, so work already queued on the GPU can still use it.

**Streams.** All calls accept Warp's `stream=` and `device=` arguments and follow Warp's rules.
Builds, refits, and launches are asynchronous; only compaction and the optional
`wo.config.verify_instances` check wait on the host.

**CUDA graphs.** OptiX launches cannot currently be captured into CUDA graphs, and warp-optix raises
an error during capture. Capture Warp work in graphs and run OptiX work between them on the same
stream.

**Debugging.** `wo.config.validation = True` enables OptiX's validation layer, which reports API
misuse through the log. It is slow and intended for debugging only.

## Examples

The [examples](examples/README.md) folder contains a gallery with images and commands. Good
starting points are:

```bash
uv run python examples/example_minimal.py                       # the smallest pipeline
uv run python examples/example_raytracing.py --live             # Cornell box with an interactive viewer
uv run python examples/example_pathtracing.py --scene glass_chess
```

## Tests

```bash
uv run pytest warp_optix/tests
uv run pytest examples/neural_texture
```

With a warm Warp kernel cache, the suite runs in a few seconds.

## License

The Warp integration is licensed under [Apache-2.0](LICENSE.md). `src/extension.cpp`,
`src/pyoptix_compat.h`, and `CMakeLists.txt` are licensed under [BSD-3-Clause](LICENSE.txt). The
bundled OptiX headers retain their [NVIDIA SDK terms](OptiX_LICENSE.txt), and their redistribution,
unmodified, in this package has been approved. See the [third-party notices](THIRD_PARTY_NOTICES.txt)
for other components.

This project is not yet accepting outside contributions; see [CONTRIBUTING.md](CONTRIBUTING.md) for
updates. To report a security vulnerability, follow [SECURITY.md](SECURITY.md).
