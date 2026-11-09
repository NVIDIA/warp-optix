# warp-optix

[![Glass and gold chess, a glass of water, and a sunlit breakfast room, all rendered with warp-optix](docs/images/banner.webp)](docs/images/banner.webp)

warp-optix puts NVIDIA's OptiX ray tracing engine inside [Warp](https://github.com/NVIDIA/warp).
You build scenes from Warp arrays, write OptiX programs as ordinary Warp kernels, and launch them
the way you launch any other kernel. If you just want pictures, there's a full path tracer on top.

> [!WARNING]
> warp-optix is alpha software. The API will change between releases, sometimes without a
> deprecation path.

## Install

```bash
pip install warp-optix
pip install "warp-optix[pathtracing,ui,usd]"   # adds the path tracer, viewer, and USD loading
```

Wheels exist for Python 3.12 on Linux x86-64 and Windows x64. All you need is an NVIDIA driver. No
CUDA Toolkit, no OptiX SDK: PyOptiX 9.1.1 comes in as a dependency, and the wheel carries the OptiX
9.1.0 headers that kernels compile against.

Working from a checkout needs a C++17 compiler as well:

```bash
uv sync --extra pathtracing --extra ui --extra usd
uv run python examples/example_pathtracing.py --scene hairball
```

## Four ways in

warp-optix has layers. Most people only need the first one or two.

1. **Ray queries in a Warp kernel.** Call `wo.accel_query_ray()` from a kernel, the same way you'd
   call `wp.mesh_query_ray()`. One kernel, no other programs. Good for depth images, visibility,
   sensors, and collision rays.
2. **Your own OptiX programs.** Write ray generation, closest-hit, any-hit, miss, intersection, and
   callable programs as Warp kernels, and let OptiX call them as rays hit things.
3. **The path tracer.** `warp_optix.pathtracing` renders glTF, USD, or procedural scenes with
   physically based materials, denoising, and an interactive viewer. You don't write any shaders.
4. **Raw OptiX.** `warp_optix.native` hands you PyOptiX itself when you need something the layers
   above don't cover.

All four use the same scenes, so let's start there. When you need exact signatures or the full
list of device builtins, the [API reference](docs/api.md) has them.

## Building a scene

OptiX traces rays against *acceleration structures*: BVHs that it builds and owns on the GPU. In
warp-optix that's `wo.Accel`, and it works a lot like `wp.Mesh`. You hand it Warp arrays, it
builds, and kernels see it as a `wp.uint64` id.

```python
import numpy as np
import warp as wp

import warp_optix as wo

wp.init()

points = wp.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)], dtype=wp.vec3)
indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
quad = wo.Accel([wo.triangles(points, indices)])
```

Triangles are one kind of input. There are three others:

| Input | What goes in |
| --- | --- |
| `wo.triangles(points, indices)` | `vec3` points and index triplets |
| `wo.curves(points, widths, indices, curve_type=)` | control points, a radius per point, and the first point of each segment (round linear, B-spline, Catmull-Rom, Bézier, or flat ribbon curves) |
| `wo.aabbs(lowers, uppers)` | a box per custom primitive, like `wp.Bvh(lowers, uppers)`; you supply the intersection program |
| `wo.instances(records, children)` | `wo.Instance` records that place other structures in the world |

Instances are how you get many copies of a mesh, or many worlds in one scene. Records are
plain Warp structs, so a kernel can write a thousand of them in one launch, straight from your
simulation state. You also tell `wo.instances` which structures the records point at. That lets the
build skip reading the records back, and it keeps those children alive for as long as the
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

When things move, rewrite the records (or the points) and call `scene.refit()`. A refit doesn't
allocate and doesn't wait on the host, so it's cheap enough to run every simulation step. Refits do
slowly lose trace quality, so call `scene.rebuild()` every so often. Static geometry can shrink
instead: `compact=True` cuts a structure to about a third of its size, and `wo.compact(accels)` does
a whole batch with a single sync.

For vertex motion blur, give `wo.triangles` a list of point arrays, one per time step, plus OptiX
motion options:

```python
from warp_optix import native

later = wp.array(points.numpy() + np.float32([0.0, 0.5, 0.0]), dtype=wp.vec3)
motion = native.MotionOptions(numKeys=2, flags=native.MOTION_FLAG_NONE, timeBegin=0.0, timeEnd=1.0)
blurred = wo.Accel([wo.triangles([points, later], indices)], motion=motion)
```

## 1. Ray queries in a Warp kernel

This is the shortest path from a scene to an answer. `wo.accel_query_ray` works like
`wp.mesh_query_ray`: give it a ray, get back whether it hit, how far, and what.

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

Three things here are different from plain Warp:

- The kernel becomes a `wo.Pipeline` and runs with `wo.launch`, which takes the same `dim`, `inputs`,
  and `stream` as `wp.launch`. `wp.tid()` works as usual.
- `module="unique"` gives the kernel its own Warp module. OptiX kernels can't compile next to normal
  kernels, and this keeps them apart.
- You can pass the `wo.Accel` itself in `inputs`. It arrives as its id and stays alive until the
  launch is done.

Besides `t`, the result has the instance id, primitive index, and barycentrics (`u`, `v`).
`wo.accel_query_ray_anyhit` only answers "did it hit anything?", and stops at the first hit, which
makes it the cheap choice for shadow and visibility rays.

## 2. Writing your own OptiX programs

Queries are enough until you want different code for different surfaces, custom primitives, or
rays that bounce. That's what OptiX programs are for. Your ray generation kernel calls
`wp.optix_trace`, OptiX walks the scene, and then it runs your closest-hit program on the nearest
hit, or your miss program if there wasn't one. The programs talk through a *payload*, a small Warp
struct that rides along with the ray.

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

The ray generation kernel's arguments are the launch parameters, and every other program can take
any of them **by name**. `shade` asks for `tint`, and that's all it gets. Inside a program, the
`wp.optix_*` builtins read the hit: the distance (`wp.optix_get_ray_tmax()`), which instance and
primitive got hit, the barycentrics, front or back face, and transforms between object and world
space. The [API reference](docs/api.md#device-builtins) lists them all.

### The shader binding table

`wo.SbtLayout` decides which program runs where. OptiX calls this the shader binding table, or SBT.
The idea is simple. You name your *ray types* (most renderers have a camera ray and a shadow ray),
and you name your *hit groups*, one per kind of surface. Each hit group can have different programs
for each ray type:

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

The layout answers the two numbers OptiX needs. `lights.get_sbt_offset("metal")` is what you pass to
`wo.make_instance` for every metal instance, and `lights.ray_type_index("shadow")` is what a shadow
`wp.optix_trace` passes as its SBT offset and miss index, with `lights.trace_stride` as the stride.
Nobody has to count records by hand. If one program works for every ray type, give a bare
`wo.HitGroup` instead of a mapping, like the first example does.

### Per-surface data

A hit group can carry a `@wp.struct`, so one closest-hit program can shade a hundred materials
instead of you writing a hundred programs. The program reads it with `wp.optix_load_sbt_data`:

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

Miss and callable programs can carry data too: wrap them as `wo.Record(kernel, data=...)`.

### Launching in a loop

`wo.launch` is cheap: about 13 µs of host time per call on an RTX PRO 6000, against roughly
5.5 µs for OptiX's own launch. In a tight simulation loop, record the launch once and replay it,
just like `wp.launch(..., record_cmd=True)`:

```python
launch = wo.launch(pipeline, dim=image.shape, inputs=[scene, wp.vec3(1.0, 0.5, 0.2), image], record_cmd=True)
for step in range(3):
    launch.set_param_by_name("tint", wp.vec3(1.0, 0.5, 0.1 * step))
    launch.launch()
```

### More program kinds

- **Custom primitives.** Build `wo.aabbs`, then give the hit group an `intersection` program that
  calls `wp.optix_report_intersection(t, kind, ...)`.
- **Built-in curves.** Use `wo.HitGroup(builtin=wo.PrimitiveType.ROUND_CUBIC_BEZIER, ...)`. OptiX
  specializes its curve intersector on the structure's build flags, so pass `build_flags=` when your
  curves weren't built with the default `PREFER_FAST_TRACE`.
- **Callables.** `wo.Pipeline(..., callables=[a, b])` makes `a` and `b` callable from any program
  with `wp.optix_direct_call(0)` and `wp.optix_direct_call(1)`. Declare a continuation callable with
  `@wo.kernel(kind=wo.ProgramKind.CONTINUATION_CALLABLE)`.
- **Exceptions.** `exception=` adds an exception program, and `exception_flags=` turns on OptiX's
  stack-overflow and trace-depth checks while you debug.
- **A shared parameter struct.** By default the ray generation kernel's arguments define the
  launch parameters. Pass `params_type=MyParams` to use a `@wp.struct` instead, so that hit programs
  can read fields the camera kernel never mentions.

[`examples/example_mixed_geometry.py`](examples/example_mixed_geometry.py) puts most of this in one
place: triangles, curves, and six analytic shapes under GPU-written instances, with a named SBT.

## 3. The path tracer

If what you want is a rendered image, skip the programs. `warp_optix.pathtracing` is a physically
based path tracer built on everything above. It reads glTF and USD, has native curves for hair and
fur, denoises with OptiX (optionally upscaling 2x), and comes with an OpenGL viewer.

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

For your own scenes, use `load_scene_from_gltf(path)` or `load_scene_from_usd(path)`, or build one
from meshes, curves, spheres, and materials. Dynamic meshes and curves update from device arrays, so a
simulation can drive them every frame without copying to the host. The [path tracer
guide](docs/pathtracing.md) covers materials, denoising, the viewer, and USD.

## 4. Raw OptiX

`warp_optix.native` is PyOptiX 9.1.1 plus a few things it's missing: cooperative-vector matrix
conversion, the extra enum values, and fixes for module compile options. The device's OptiX context
is one call away:

```python
context = wo.get_context("cuda").native  # PyOptiX DeviceContext, borrowed: don't destroy it
print(native.version(), native.getCoopVecProperty(context))
```

Native objects follow PyOptiX's rules. You destroy them yourself, and you keep them on their own
device. Importing warp-optix doesn't change the `optix` module, so other PyOptiX code in the same
process keeps working as before.

## Things to know

**Lifetime.** `wo.Accel` and `wo.Pipeline` work like `wp.array`. They free themselves when the last
reference goes away, or right away with `close()` or a `with` block. Memory is freed in stream order,
so GPU work that's already queued can still use it.

**Streams.** Every call takes Warp's `stream=` and `device=` arguments with Warp's rules. Builds,
refits, and launches are asynchronous. Only compaction and the opt-in `wo.config.verify_instances`
check wait on the host.

**CUDA graphs.** OptiX launches can't be captured into CUDA graphs yet, and warp-optix raises an
error if you try. Capture your Warp work in graphs and run the OptiX work between them on the same
stream.

**Debugging.** `wo.config.validation = True` turns on OptiX's validation layer, which catches API
misuse and reports it through the log. It's slow, so leave it off otherwise.

## Examples

The [examples](examples/README.md) folder has a gallery with images and commands. The best places
to start:

```bash
uv run python examples/example_minimal.py                       # the smallest pipeline
uv run python examples/example_raytracing.py --live             # a Cornell box you can fly through
uv run python examples/example_pathtracing.py --scene glass_chess
```

## Tests

```bash
uv run pytest warp_optix/tests
uv run pytest examples/neural_texture
```

The suite takes a few seconds once Warp's kernel cache is warm.

## License

The Warp integration is [Apache-2.0](LICENSE.md). `src/extension.cpp`, `src/pyoptix_compat.h`, and
`CMakeLists.txt` are [BSD-3-Clause](LICENSE.txt). The bundled OptiX headers keep their
[NVIDIA SDK terms](OptiX_LICENSE.txt), and NVIDIA has approved shipping them, unmodified, in this
package. See the [third-party notices](THIRD_PARTY_NOTICES.txt) for the rest.

We aren't taking outside contributions yet; [CONTRIBUTING.md](CONTRIBUTING.md) will say when that
changes. To report a security issue, follow [SECURITY.md](SECURITY.md).
