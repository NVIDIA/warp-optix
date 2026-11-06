# warp_optix

OptiX ray tracing for [Warp](https://github.com/NVIDIA/warp), using PyOptiX and a small supplemental native extension.

Acceleration structures are built directly from Warp device arrays, OptiX programs are ordinary
Warp kernels, and launches mirror `wp.launch`:

```python
import warp as wp
import warp_optix as wo


@wp.kernel(module="unique")
def render(scene: wp.uint64, depth: wp.array2d[float]):
    y, x = wp.tid()
    start = wp.vec3(float(x) / 64.0 - 1.0, float(y) / 64.0 - 1.0, 5.0)
    query = wo.accel_query_ray(scene, start, wp.vec3(0.0, 0.0, -1.0), 100.0)
    depth[y, x] = query.t if query.result else -1.0


accel = wo.Accel([wo.triangles(points, indices)], compact=True)
pipeline = wo.Pipeline(render)
depth = wp.empty((128, 128), dtype=float)
wo.launch(pipeline, dim=depth.shape, inputs=[accel.id, depth])
```

Importing `warp_optix` registers the `wp.optix_*` device builtins with Warp. They are valid only
inside kernels compiled into a `wo.Pipeline`.

## Install

Install from the repository root; PyOptiX is installed as a dependency:

```bash
pip install -e .
```

`warp_optix` requires `warp-lang>=1.17`.

## Devices, contexts, and lifetime

Like Warp's CUDA context, the OptiX device context is implicit: each CUDA device gets one, created
on first use and destroyed at interpreter exit. Its settings come from `wo.config`
(`wo.config.validation = True`, `wo.config.log_level = 4`), which is read when a device's context is
first created; later changes only affect devices whose context does not exist yet.
`wo.get_context(device).native` gives the PyOptiX `DeviceContext` for advanced calls; it is borrowed
and must not be destroyed.

Constructors and free functions take `device=` and, where they submit work, `stream=`:

- neither given: the current device and its current stream;
- only `stream`: the stream's device;
- only `device`: that device's current stream;
- both: they must name the same device, otherwise `ValueError`.

`wo.Accel` without either uses the device of its inputs; `wo.launch` defaults to the pipeline's
device. Methods of device-bound objects (`accel.refit()`, `params.assign()`) take only `stream=`.

Resources (`wo.Accel`, `wo.Pipeline`, `wo.LaunchParams`, `wo.Denoiser`) follow Warp's lifetime model
(`wp.Mesh`, `wp.array`): they release themselves when the last reference is dropped, or explicitly
with `.close()` or `with`:

- Builds, refits, and launches record no CUDA events and keep nothing alive for in-flight work.
- `close()` makes a resource unusable at once (`RuntimeError` on further use) and frees its device
  memory right away. That memory is `wp.array` storage, which Warp frees in stream order, so work
  already submitted can still read it. It never blocks the host.
- A pipeline's native OptiX objects (pipeline, module, program groups) and a denoiser's native
  denoiser are destroyed under the device's CUDA context once the work submitted before the close
  has finished, and never during a graph capture on the device.
- Kernels see an acceleration structure only as the integer `accel.id`. As with `wp.Mesh.id`, keep
  the `wo.Accel` alive while work that reads the id may still be submitted. A `wo.Accel` passed in
  launch inputs is referenced by the pipeline's parameter storage until the next launch on that
  stream replaces it.
- An instance structure keeps the memory of its child structures alive, even if a child is closed.
- Failed pipeline construction destroys only the native objects it created.

## Programs and pipelines

OptiX programs can use ordinary `@wp.kernel(module="unique")` kernels. By default, the raygen kernel's arguments
become the pipeline's launch parameters, and `wp.tid()` returns the launch index, so the same
`dim` is used as with `wp.launch` (`dim=(height, width)` gives an OptiX launch of `width x
height`). Every other program — miss, closest-hit, any-hit, intersection, direct callable, and
exception — declares any subset of the raygen arguments **by name with identical types**:

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
    wp.optix_set_payload(0, wp.optix_float_as_uint32(0.0))


pipeline = wo.Pipeline(
    raygen,
    miss=[background],
    hit_groups=[wo.HitGroup(closest_hit=shade)],
    payload_words=3,
)
wo.launch(pipeline, dim=image.shape, inputs=[accel.id, wp.vec3(1.0, 0.5, 0.2), image])
```

`wp.optix_trace(id, start, dir, max_t, payload, min_t=0.0, time=0.0, mask=0xFF, flags=0, sbt_offset=0,
sbt_stride=1, miss_index=0)` follows `wp.mesh_query_ray(id, start, dir, max_t)`. Trailing arguments
may be omitted but not skipped by keyword, because Warp's external-builtin API has no default
arguments; `wp.optix_traverse()` takes the same arguments with an optional payload. Flags, masks,
SBT indices, and register indices accept any integer type, so `wo.RayFlags` members need no cast.

`wo.Pipeline(raygen, *, miss, hit_groups, ray_types, sbt, callables, exception, exception_flags,
payload_words, attribute_words, max_trace_depth, max_traversable_depth, graph_flags, motion_blur,
max_registers, device)` compiles all programs into one module and builds the shader binding table:

- `miss` lists one miss program per ray type; `None` is a record without a program.
- `hit_groups` holds `wo.HitGroup(closest_hit=, any_hit=, intersection=, builtin=)` entries.
  The hit record for geometry type `g` and ray type `r` is `hit_groups[g * ray_types + r]`;
  trace with `sbt_offset=r` and `sbt_stride=ray_types`, and give instances `sbt_offset=g *
  ray_types`. `builtin=wo.PrimitiveType.*` selects OptiX's built-in curve intersector; a
  group with an `intersection` program serves custom primitives. Primitive-type flags are derived
  from the groups.
- `callables` are direct-callable programs, invoked with `wp.optix_direct_call(index)` in list
  order. Callables use a no-argument, void interface: they read launch parameters and write
  referenced Warp arrays.
- `exception` enables user exceptions. Raise one with `wp.optix_throw_exception(code, ...)` (up
  to eight optional details) and read it with `wp.optix_get_exception_code()` and
  `wp.optix_get_exception_detail(i)`.
- `exception_flags` selects `wo.ExceptionFlags.NONE`, `STACK_OVERFLOW`,
  `TRACE_DEPTH`, and `USER`, including bitwise combinations. When omitted,
  the default is `USER` if an exception program is supplied, otherwise `NONE`.
  For example, `exception_flags=wo.ExceptionFlags.STACK_OVERFLOW | wo.ExceptionFlags.TRACE_DEPTH`
  enables those debugging checks. Include `USER` when using `optix_throw_exception`.

`wo.launch(pipeline, dim, inputs=None, device=None, stream=None, *, params=None)` mirrors `wp.launch`:
it writes the launch parameters with a Warp kernel on `stream` and then launches OptiX on the same
stream. Neither step synchronizes the host. `wo.Accel` objects in `inputs` are passed as their `id`
and kept alive for the launch. OptiX does not record its work into CUDA graphs, so `wo.launch`,
`wo.Accel`, `refit()`/`rebuild()`, and `wo.Pipeline` raise `RuntimeError` during stream capture.
Kernels written against a single launch-parameter struct (`def raygen(params: MyParams)`) work
unchanged: launch them with `inputs=[params]`.

Automatic parameter storage is separate for each CUDA stream. Sequential launches
on one stream reuse its buffer. Calls on shared resources from multiple host
threads require external synchronization.

For a shared schema independent of raygen, pass `params_type=MyParams`, where
`MyParams` is a `@wp.struct`. Each program declares any subset of the schema's
fields by name and identical type. This lets a miss or hit program use fields
that raygen does not declare. Launch with `inputs=[my_params_instance]`.
The schema is the parameter block itself; to use a single `params: MyParams`
argument in every shader, keep the default inferred schema instead.

Advanced callers can own parameter storage explicitly:

```python
params = wo.LaunchParams(pipeline)
params.assign(my_params_instance, stream=stream)
wo.launch(pipeline, dim=(height, width), stream=stream, params=params)
```

For inferred schemas, `assign()` takes the argument sequence instead.
`LaunchParams` belongs to its pipeline (and its device) and must be assigned before use.
Supply either `inputs` or `params`, never both. Updates and launches on shared
parameters are ordered on the GPU, including across streams. Separate
`LaunchParams` objects allow independent concurrent launches. Referenced arrays remain retained
until replaced or until the `LaunchParams` is released (`close()`/`with`); callers must still order
their own writes to those arrays. Directly modifying
`params.data` bypasses the API's ordering and validation.

An optional `@wo.kernel(kind=wo.ProgramKind.RAYGEN, module="unique")`
decorator replaces `@wp.kernel` when a shader should declare its intended stage.
It also accepts an existing Warp kernel. Pipeline construction rejects a declared
stage that disagrees with its slot. Undecorated Warp kernels continue to get their
stage from their pipeline slot.

Compare `examples_warp/example_warp_optix_kernels.py` with
`examples_warp/example_warp_optix_explicit_params.py`: both compute the same
output; the latter uses a shared schema, stage declarations, named SBT layout,
and separate buffers on separate streams.


### Explicit SBT layouts

The default pipeline API packs SBT records automatically. For explicit control
over ray names, group ordering, and instance offsets, use `wo.SbtBuilder`:

```python
sbt = wo.SbtBuilder(ray_types=("radiance", "shadow"))
sbt.set_miss("radiance", radiance_miss)
sbt.set_miss("shadow", shadow_miss)
glass = sbt.add_hit_group(
    "glass",
    radiance=wo.HitGroup(closest_hit=glass_hit),
    shadow=wo.HitGroup(any_hit=glass_shadow),
)
metal = sbt.add_hit_group(
    "metal",
    radiance=wo.HitGroup(closest_hit=metal_hit),
    shadow=wo.HitGroup(any_hit=metal_shadow),
)
print(sbt.describe())
pipeline = wo.Pipeline(raygen, sbt=sbt, payload_words=1)
layout = pipeline.sbt_layout
metal_offset = layout.get_sbt_offset(metal)  # 2 records; use in wo.make_instance()
shadow_index = layout.ray_type_index("shadow")  # trace SBT offset and miss index
trace_stride = layout.trace_stride  # 2 records
```

Programs are ordinary Warp kernels, with their kind determined by the assigned
slot. Each group specifies every ray type, using the same geometry kind across
those rays. `set_miss(name, None)` explicitly creates an empty miss slot;
`HitGroup()` creates an empty hit group for hit-object traversal. Ray names must
be unique, and the trace stride supports at most 15 ray types.

Layouts use group-major, ray-type-minor order. Append groups to control their
order; previously returned `HitGroupHandle` offsets remain stable. Pass offsets
obtained from `layout.get_sbt_offset(handle)` to the Warp kernel that writes
instances with `wo.make_instance()`. This lookup rejects handles from another
builder or groups added after the snapshot. Offsets and trace strides count
records, not bytes.

`wo.Pipeline(..., sbt=builder)` takes an immutable snapshot. Later builder
edits do not change existing pipelines. You may also pass `sbt=builder.snapshot()`
to reuse a frozen layout. Builders own no native resources and can be reused
across devices; each pipeline owns its compiled programs and packed buffers.
`pipeline.sbt_layout` exposes the named snapshot; automatic pipelines have `sbt_layout=None`.

An explicit `sbt` cannot be combined with `miss`, `hit_groups`, or `ray_types`.
Callables and exception programs remain independent pipeline options. See
`examples_warp/example_warp_optix_mixed_geometry.py` for a complete example.

`wo.ProgramKind` names the program kinds and their OptiX entry-name prefixes. Flags are typed
enums usable both on the host and inside kernels (Warp supports `IntEnum`/`IntFlag` in kernels):
`wo.RayFlags` and `wo.HitKind` for device code, and `wo.BuildFlags`, `wo.GeometryFlags`,
`wo.InstanceFlags`, `wo.PrimitiveType`, `wo.TraversableGraphFlags`, and `wo.ExceptionFlags`.
Compile options remain available as `wo.native.COMPILE_*`.

## Ray queries without hit programs

`wo.accel_query_ray(id, start, dir, max_t, min_t=0.0, mask=0xFF, flags=0) -> wo.AccelQueryRay`
and `wo.accel_query_ray_anyhit(id, start, dir, max_t, min_t=0.0, mask=0xFF) -> bool` mirror
`wp.mesh_query_ray`/`wp.mesh_query_ray_anyhit`. They traverse with `wp.optix_traverse()` and read the
result from the OptiX hit object, so a raygen-only pipeline can query built-in geometry. The
`query` naming keeps them distinct from closest-hit and any-hit *programs*.

`AccelQueryRay` fields:

- `result`: whether anything was hit. On a miss, no other field is defined.
- `t`: the hit distance.
- `u`, `v`: the barycentrics for triangles, and the ribbon parameters for flat (ribbon) curves. For
  other built-in curves, `u` is the curve parameter and `v` is 0. For custom primitives, they hold the
  bits of the first two reported attributes, as floats.
- `instance_id` and `primitive_index`.

After a query, the `wp.optix_hit_object_*` builtins expose further hit data until the next traversal.

## Acceleration structures

`wo.Accel` mirrors `wp.Mesh`/`wp.Bvh`: construction builds the structure from build inputs, which
retain their Warp **device** arrays:

| Builder | Input |
| --- | --- |
| `wo.triangles(points, indices, flags=)` | `vec3` (or `float32[N, 3]`) points, int32/uint32 index triplets; a list of point arrays gives motion keys |
| `wo.aabbs(lowers, uppers, flags=, sbt_indices=)` | `vec3` bounds per custom primitive, like `wp.Bvh(lowers, uppers)`; or packed `wo.aabbs(bounds=float32[N, 6])` to avoid a packing copy |
| `wo.curves(points, widths, indices, curve_type=, flags=)` | control points, per-point radii, first control point of each segment |
| `wo.instances(records)` | `wp.array(dtype=wo.Instance)` |

Geometry defaults to `GeometryFlags.DISABLE_ANYHIT`; pass `flags=wo.GeometryFlags.NONE` when an
any-hit program must run.

`triangles()` and `curves()` read indices back once to validate their bounds; pass
`validate=False` to skip that synchronization for indices already checked on the host.
Keep those indices unchanged when reusing a build input; create a new input
after changing topology. Point-only refits do not repeat this validation.

`wo.Accel(inputs, *, flags=, compact=, motion=, device=, stream=)` builds on a Warp stream and
exposes `.id` (the `wp.uint64` passed to kernels), `.device`, and `.flags`. Initial instance builds
read child ids back once to verify that every child is a live structure on the same device.
`compact=True` also synchronizes to shrink static structures; `wo.compact(accels, stream=)` compacts
many structures built with `wo.BuildFlags.ALLOW_COMPACTION` with a single synchronization. Compaction
changes `.id`, so compact children before building instance structures over them. `accel.refit(stream=)` refits a
structure built with `wo.BuildFlags.ALLOW_UPDATE` to its inputs' current contents using the
retained scratch buffer, so it neither allocates nor synchronizes on success; `accel.rebuild(stream=)`
rebuilds in place to restore trace quality (structures without `ALLOW_UPDATE` allocate a temporary
scratch buffer per rebuild). Keep indices, primitive counts, and instance child ids unchanged when
reusing an input; build a new input and structure for topology changes.

`wo.Instance` is the 80-byte `OptixInstance` layout as a Warp struct, and the `wo.make_instance()`
Warp function writes one from a pose and an acceleration-structure id. It takes a rigid
`wp.transform`, a full affine `wp.mat44` (scale and shear allowed), or a `wp.mat33` linear part plus a
`wp.vec3` translation, followed by `accel_id, instance_id=0, sbt_offset=0, mask=0xFF, flags=0`.
The integer arguments accept any integer type. In kernels compiled with backward passes enabled
(Warp's default for `wp.launch` kernels), pass them positionally: Warp 1.18 miscompiles keyword
arguments to generic integer parameters of a `@wp.func` in the adjoint code. OptiX programs are
compiled without backward passes and are unaffected.
Instance records are therefore written on the GPU, e.g. one per simulated world:

```python
@wp.kernel
def write_instances(gas: wp.uint64, poses: wp.array[wp.transform], records: wp.array[wo.Instance]):
    i = wp.tid()
    records[i] = wo.make_instance(poses[i], gas, i)  # instance_id = i


records = wp.empty(world_count, dtype=wo.Instance)
wp.launch(write_instances, dim=world_count, inputs=[gas.id, poses, records])
ias = wo.Accel([wo.instances(records)], flags=wo.BuildFlags.ALLOW_UPDATE)
# Each step: rewrite the records, then refit.
wp.launch(write_instances, dim=world_count, inputs=[gas.id, poses, records])
ias.refit()
```

For vertex motion blur, pass a list of vertex keys and OptiX motion options, then enable motion
on the pipeline and trace with a ray time:

```python
from warp_optix import native

motion = native.MotionOptions(numKeys=2, flags=native.MOTION_FLAG_NONE, timeBegin=0.0, timeEnd=1.0)
gas = wo.Accel([wo.triangles([keys0, keys1], indices)], motion=motion)
pipeline = wo.Pipeline(raygen, ..., motion_blur=True)
```

## Device builtins

Warp kernels expose the common OptiX hit context directly: ray origin/direction (world and
object space), `tmin`/`tmax`, time, flags, and visibility mask; instance ID and index; primitive
index, SBT GAS index, GAS handle, and primitive type; front/back-face tests; and
point/vector/normal transforms in both directions. Names follow Warp's snake-case convention with
the `optix_` domain prefix, for example `wp.optix_get_instance_index()` and
`wp.optix_transform_point_world_to_object(point)`. Every builtin is documented (`doc=`). Payloads
are moved register by register with `wp.optix_set_payload(i, value)`/`wp.optix_get_payload(i)`
(`i` in `[0, 32)`; use a constant index, which compiles to the single OptiX intrinsic) or as whole
Warp structs with `wp.optix_load_payload()`/`wp.optix_store_payload()`. The numbered forms
`wp.optix_get_payload_<i>()`/`wp.optix_set_payload_<i>()` remain as deprecated aliases.

Inside an intersection program, use `wp.optix_get_object_ray_origin()`,
`wp.optix_get_object_ray_direction()`, and `wp.optix_report_intersection()`, which accepts a hit
distance, hit kind, and zero to eight `wp.uint32` attributes. Closest-hit and any-hit programs read
them with `wp.optix_get_attribute(i)` for `i` in `[0, 8)`;
`wp.optix_float_as_uint32()` and `wp.optix_uint32_as_float()` reinterpret the bits of floats.

### Geometry and transform queries

Triangle hits expose barycentrics, object-space vertices (`wp.optix_get_triangle_vertex_data()`
returns a `wp.mat33` whose rows are the three vertices), and generic and triangle-specific face
predicates. Built-in curve hits expose their object-space control points as fixed Warp matrices
whose rows are `(x, y, z, radius)`, and `wp.optix_get_curve_parameter()`:

| Query family | Return type |
| --- | --- |
| Linear curve | `2 x 4` float matrix |
| Quadratic B-spline and ribbon | `3 x 4` float matrix |
| Cubic B-spline, Catmull-Rom, and cubic Bezier | `wp.mat44` |

Ribbons additionally expose their two intersection parameters and derived normal. Geometry
current-hit queries are valid in any-hit and closest-hit programs for the corresponding primitive
type.

The current transform list can be inspected with `wp.optix_get_transform_list_size()`,
`wp.optix_get_transform_list_handle(index)`, and
`wp.optix_get_transform_type_from_handle(handle)`. Composite affine transforms are returned as
ordinary homogeneous `wp.mat44` values:

```python
object_to_world = wp.optix_get_object_to_world_matrix()
world_to_object = wp.optix_get_world_to_object_matrix()
```

The current transform matrix and transform-list queries are valid in intersection, any-hit, and
closest-hit programs. Random-access geometry fetches and raw SDK transform pointers are
deliberately not exposed.

### Cooperative vectors

OptiX 9 cooperative vectors are native SDK facilities, not Warp operations. The Python extension
exposes the host-side SDK pieces needed to prepare matrices:

- `DEVICE_PROPERTY_COOP_VEC` and `DevicePropertyCoopVecFlags` for support queries.
- `CoopVecElemType`, `CoopVecMatrixLayout`, and `CoopVecMatrixDescription`.
- `DeviceContext.coopVecMatrixComputeSize()` and `DeviceContext.coopVecMatrixConvert()`.

The conversion call is asynchronous and follows OptiX's 64-byte alignment and allocation-lifetime
requirements. Device-side intrinsics use explicit output vectors, the convention required by
Warp's public addon API:

```python
Vec32h = wp.types.vector(length=32, dtype=wp.float16)
Vec16h = wp.types.vector(length=16, dtype=wp.float16)


@wp.kernel(module="unique")
def closest_hit(matrices: wp.uint64, weight_offset: wp.uint32, biases: wp.uint64, bias_offset: wp.uint32):
    inputs = Vec32h(wp.float16(0.0))
    outputs = Vec16h()
    wp.optix_coop_vec_matmul_bias_fp8_e4m3(inputs, matrices, weight_offset, biases, bias_offset, wp.uint32(0), outputs)
```

The registered family includes load, conversion, elementwise arithmetic, `exp2`, `log2`, `tanh`,
fused multiply-add, reductions, outer products, matrix-size queries, and matrix multiply variants
for FP16, FP8 E4M3/E5M2, signed int8, and unsigned int8. Named matrix variants use an
inference-optimal, non-transposed matrix with the element interpretation in the function name.

## Denoising

`wo.Denoiser` runs the OptiX temporal denoiser on Warp arrays, optionally with 2x upscaling.
Pass contiguous `(height, width)` arrays: float4 HDR color, float3 or float4 albedo and
world-space normals (`w` is ignored), and float2 motion, the offset from each pixel to its
position in the previous frame in input pixels. The denoiser owns its scratch, state, and
temporal history and queues all work on one stream:

```python
denoiser = wo.Denoiser(width, height, upscale=True, stream=stream)
filtered = denoiser.apply(color, albedo=albedo, normal=normal, motion=motion, reset=True)
# filtered has shape (2 * height, 2 * width) and stays valid until the next apply().
denoiser.close()
```

`reset=True` discards the temporal history after a cut.

## Path tracer

The path tracer and its OpenGL viewer are a separate package built on this API,
`warp_optix_pathtracing`, installed with `warp-optix`; see its
[README](../warp_optix_pathtracing/README.md).

## Layout

- `warp_optix/__init__.py` — public API.
- `warp_optix/context.py` — implicit per-device contexts (`get_context`), device/stream resolution, and native-object release.
- `warp_optix/config.py` — global settings (`validation`, `log_level`).
- `warp_optix/accel.py` — build inputs, `Accel`, and `Instance`/`make_instance`.
- `warp_optix/pipeline.py` — `Pipeline`, `HitGroup`, `LaunchParams`, and `launch`.
- `warp_optix/program.py` — rewrites Warp kernels into OptiX programs (`ProgramKind`, `kernel`).
- `warp_optix/rays.py` — `accel_query_ray`, `accel_query_ray_anyhit`, and `AccelQueryRay`.
- `warp_optix/enums.py` — flag and primitive-type enums, usable on the host and in kernels.
- `warp_optix/builtins.py` — OptiX builtin registrations.
- `warp_optix/_native/include/warp_optix_builtins.h` — device-side C++ wrappers.
- `warp_optix/native.py` — PyOptiX plus the supplemental bindings (`_backend.py`, `_ext`).
- `warp_optix/addon.py` — registers the builtins and provides the OptiX module build options.
- `warp_optix/denoiser.py` — `Denoiser`.

See `examples_warp/` for complete programs: `example_warp_optix_kernels.py` (minimal pipeline),
`example_warp_optix_curves.py` (built-in curves), `example_warp_optix_mixed_geometry.py`
(triangles, curves, and custom primitives under GPU-written instances),
`example_warp_optix_motion_blur.py` (vertex motion keys), and
`example_warp_optix_tiny_raytracer.py` (interactive viewer).
