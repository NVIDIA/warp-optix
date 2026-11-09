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
`wo.config.verify_instances` is read on every instance build.
`wo.get_context(device).native` gives the PyOptiX `DeviceContext` for advanced calls; it is borrowed
and must not be destroyed.

Constructors and free functions take `device=` and, where they submit work, `stream=`:

- neither given: the current device and its current stream;
- only `stream`: the stream's device;
- only `device`: that device's current stream;
- both: they must name the same device, otherwise `ValueError`.

As with `wp.launch(stream=...)`, work submitted to a stream is ordered only after earlier work on
that stream, not after the current stream; order producers on other streams with
`stream.wait_stream()` or `stream.wait_event()`.

`wo.Accel` without either uses the device of its inputs; `wo.launch` defaults to the pipeline's
device. Methods of device-bound objects (`accel.refit()`, `params.assign()`) take only `stream=`.

Resources (`wo.Accel`, `wo.Pipeline`) follow Warp's lifetime model
(`wp.Mesh`, `wp.array`): they release themselves when the last reference is dropped, or explicitly
with `.close()` or `with`:

- Builds, refits, and launches record no CUDA events and keep nothing alive for in-flight work.
- `close()` makes a resource unusable at once (`RuntimeError` on further use) and frees its device
  memory right away. That memory is `wp.array` storage, which Warp frees in stream order, so work
  already submitted can still read it. It never blocks the host.
- A pipeline's native OptiX objects (pipeline, module, program groups) are destroyed under the
  device's CUDA context once the launches submitted before the close have finished, and never during
  a graph capture on the device.
- Kernels see an acceleration structure only as the integer `accel.id`. As with `wp.Mesh.id`, keep
  the `wo.Accel` alive while work that reads the id may still be submitted. A `wo.Accel` passed in
  launch inputs is referenced by the pipeline's parameter storage until the next launch on that
  stream replaces it.
- An instance structure keeps the memory of the children listed in `wo.instances(records, children)`
  alive, at every level of nesting, even if a child is closed or dropped.
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

`wo.Pipeline(raygen, *, miss, hit_groups, ray_types, sbt, callables, continuation_callables,
exception, exception_flags,
payload_words, attribute_words, max_trace_depth, max_traversable_depth, graph_flags, motion_blur,
max_registers, device)` compiles all programs into one module and builds the shader binding table:

- `miss` lists one miss program per ray type; `None` is a record without a program.
- `hit_groups` holds `wo.HitGroup(closest_hit=, any_hit=, intersection=, builtin=, build_flags=, data=)` entries.
  The hit record for geometry type `g` and ray type `r` is `hit_groups[g * ray_types + r]`;
  trace with `sbt_offset=r` and `sbt_stride=ray_types`, and give instances `sbt_offset=g *
  ray_types`. `builtin=wo.PrimitiveType.*` selects OptiX's built-in curve intersector, which
  OptiX specializes on build flags: `build_flags` (default `PREFER_FAST_TRACE`, like `wo.Accel`)
  must match the curve structures' flags, so curves built with other flags need their own group. A
  group with an `intersection` program serves custom primitives. Primitive-type flags are derived
  from the groups.
- `callables` are direct-callable programs, invoked with `wp.optix_direct_call(index)` in list
  order. `continuation_callables` are invoked with `wp.optix_continuation_call(index)`; they
  follow the direct callables in the same SBT section, so `continuation_callables[k]` has index
  `len(callables) + k`. Callables use a no-argument, void interface: they read launch parameters
  and write referenced Warp arrays. Miss and callable entries may be
  `wo.Record(kernel, data=)` to carry [record data](#per-record-sbt-data).
- `exception` enables user exceptions. Raise one with `wp.optix_throw_exception(code, ...)` (up
  to eight optional details) and read it with `wp.optix_get_exception_code()` and
  `wp.optix_get_exception_detail(i)`.
- `exception_flags` selects `wo.ExceptionFlags.NONE`, `STACK_OVERFLOW`,
  `TRACE_DEPTH`, and `USER`, including bitwise combinations. When omitted,
  the default is `USER` if an exception program is supplied, otherwise `NONE`.
  For example, `exception_flags=wo.ExceptionFlags.STACK_OVERFLOW | wo.ExceptionFlags.TRACE_DEPTH`
  enables those debugging checks. Include `USER` when using `optix_throw_exception`.

`wo.launch(pipeline, dim, inputs=None, outputs=None, *, device=None, stream=None, record_cmd=False)`
mirrors `wp.launch`: `inputs` and `outputs` are concatenated into the raygen arguments, and a raygen
that calls `wp.tid()` must unpack as many indices as `dim` has entries.
it packs the launch parameters on the host, copies them to the device on `stream`, and then launches
OptiX on the same stream. Neither step synchronizes the host. `wo.Accel` objects in `inputs` are passed as their `id`
and kept alive for the launch. OptiX does not record its work into CUDA graphs, so `wo.launch`,
`wo.Accel`, `refit()`/`rebuild()`, and `wo.Pipeline` raise `RuntimeError` during stream capture.
Kernels written against a single launch-parameter struct (`def raygen(params: MyParams)`) work
unchanged: launch them with `inputs=[params]`.

Automatic parameter storage is separate for each CUDA stream. Sequential launches
on one stream reuse its buffer. Calls on shared resources from multiple host
threads require external synchronization.

For launches in a simulation loop, record the launch once, as with `wp.launch(..., record_cmd=True)`:

```python
launch = wo.launch(pipeline, dim=(height, width), inputs=[accel, image], record_cmd=True)
for step in range(steps):
    launch.set_param_by_name("accel", accels[step])  # or set_param_at_index, set_params, set_dim
    launch.launch(stream=stream)
```

A `wo.Launch` checks and packs its parameters once and keeps its own parameter block; `launch()`
copies the parameters to the device only after they change, so an unchanged replay costs little more
than the OptiX launch itself. Replays on different streams are ordered on the GPU; use separate
`wo.Launch` objects for independent concurrent launches. Arrays and acceleration structures set as
parameters stay referenced until replaced or until the `wo.Launch` is dropped.

For a shared schema independent of raygen, pass `params_type=MyParams`, where
`MyParams` is a `@wp.struct`. Each program declares any subset of the schema's
fields by name and identical type. This lets a miss or hit program use fields
that raygen does not declare. Launch with `inputs=[my_params_instance]`.
The schema is the parameter block itself; to use a single `params: MyParams`
argument in every shader, keep the default inferred schema instead.

An optional `@wo.kernel(kind=wo.ProgramKind.RAYGEN)`
decorator replaces `@wp.kernel` when a shader should declare its intended stage.
It defaults to `module="unique"`, because a kernel using `wp.optix_*` builtins
cannot compile as part of an ordinary Warp module and would break `wp.launch` of the
other kernels in it; an explicit `module=` is honored. It also accepts an existing
Warp kernel, which keeps the module chosen by its `@wp.kernel`; a kernel in its Python module's
default Warp module is rejected for the same reason. Pipeline construction
rejects a declared stage that disagrees with its slot. Undecorated Warp kernels
continue to get their stage from their pipeline slot.

Compare `examples_warp/example_warp_optix_kernels.py` with
`examples_warp/example_warp_optix_explicit_params.py`: both compute the same
output; the latter uses a shared schema, stage declarations, named SBT layout,
and separate buffers on separate streams.


### Per-record SBT data

Like most OptiX applications, a hit group, miss, or callable record can carry its own data, so one
closest-hit program shades many materials. `data` is a `@wp.struct` instance; the program reads it
with `wp.optix_load_sbt_data(value)`, which fills a struct of the same type:

```python
@wp.struct
class Material:
    color: wp.vec3


@wp.kernel(module="unique")
def shade():
    material = Material()
    wp.optix_load_sbt_data(material)
    payload = Payload()
    payload.color = material.color
    wp.optix_store_payload(payload)


red, blue, sky = Material(), Material(), Material()
red.color, blue.color, sky.color = wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.5, 0.7, 1.0)
pipeline = wo.Pipeline(
    raygen,
    miss=[wo.Record(background, data=sky)],  # background also loads a Material
    hit_groups=[wo.HitGroup(closest_hit=shade, data=red), wo.HitGroup(closest_hit=shade, data=blue)],
    payload_words=3,
)
red.color = wp.vec3(1.0, 0.5, 0.0)
pipeline.update_sbt_data()  # re-upload every record's data; no rebuild
```

Records in one section (miss, hit group, callables) may use different struct types; the
section's stride fits the largest. The device does not check the type, so load the type that
was stored. Arrays referenced by record data stay alive with the pipeline, and must be on its
device. `update_sbt_data(stream=None)` copies like `wp.copy` from host memory, ordered with
launches on that stream.

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
| `wo.instances(records, children)` | `wp.array(dtype=wo.Instance)`, plus the `wo.Accel` objects the records reference |

Geometry defaults to `GeometryFlags.DISABLE_ANYHIT`; pass `flags=wo.GeometryFlags.NONE` when an
any-hit program must run. Any build input may be empty (no primitives or records); the structure
then has nothing to intersect, which suits dynamic geometry that has no elements yet.

`triangles()` and `curves()` read indices back once to validate their bounds; pass
`validate=False` to skip that synchronization for indices already checked on the host.
Keep those indices unchanged when reusing a build input; create a new input
after changing topology. Point-only refits do not repeat this validation.

`wo.Accel(inputs, *, flags=, compact=, motion=, device=, stream=)` builds on a Warp stream and
exposes `.id` (the `wp.uint64` passed to kernels), `.device`, and `.flags`. Instance records are
written on the GPU, so builds never read them back: `children` names the structures they reference,
and the instance structure keeps those alive. Like `wp.config.verify_cuda`,
`wo.config.verify_instances = True` makes instance builds synchronize and check that every record
references a listed child; otherwise an unlisted or invalid child id is undefined behavior.
`compact=True` also synchronizes to shrink static structures; `wo.compact(accels, stream=)` compacts
many structures built with `wo.BuildFlags.ALLOW_COMPACTION`, which keep their scratch buffer until then,
with a single synchronization. Compaction changes `.id`, so compact children before building instance
structures over them. `accel.refit(stream=)` refits a
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
ias = wo.Accel([wo.instances(records, [gas])], flags=wo.BuildFlags.ALLOW_UPDATE)
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

## OpenGL display

`warp_optix.gl` provides optional pyglet-based helpers for interactive examples:
`GLInteropViewer` presents a CUDA-mapped `uint32` RGBA pixel buffer each frame without a host
copy, and `GLLineOverlay` draws depth-tested lines over it (see below).

## Path-tracing viewer

The example path tracer is also installed as `warp_optix.pathtracing`. Install
its windowing and image extras with:

```bash
pip install -e ".[pathtracing]"
```

`Viewer` is a standalone OptiX path-tracing viewer with OptiX temporal
denoising. `ViewerBackend` exposes a renderer-facing `log_*`
API without importing a simulation framework.

### Denoising and upscaling

`PathTracer` and `Viewer` accept the same optional backend
selector and support switching without rebuilding the scene:

```python
from warp_optix.pathtracing import PathTracer

api = PathTracer(denoiser="auto", optix_upscale=True)
api.initialize()
api.set_denoiser("none")  # Raw rendering.
api.set_denoiser("optix", optix_upscale=False)  # Native-resolution denoising.
```

The default `denoiser="auto"` selects OptiX, with raw rendering as a fallback.
Explicit `"optix"` selection requires successful initialization; `"none"`
disables denoising. Inspect `active_denoiser` and `denoiser_error` for the active
backend and fallback diagnostics.

OptiX uses temporal denoising at native resolution by default;
`optix_upscale=True` selects its temporal 2x upscaling model. Odd output sizes
use rounded-up input dimensions and crop the extra output row or column.
Switches and resizes reset reconstruction history and affected CUDA graphs.
Use `reset_temporal_history()` after a discontinuous scene change.
Rendering respects `samples_per_frame`. The rendering panel exposes denoiser
selection and upscaling controls.

For custom Warp renderers, `warp_optix.pathtracing` also exports
`DenoiserInputs` and `OptixDenoiser`. Pass float4 HDR color, float4 world-space
normals, float4 albedo, and float2 current-to-previous motion in input pixels
(without jitter). The wrapper manages scratch/state allocations, temporal
history, and stream ordering:

```python
from warp_optix.pathtracing import DenoiserInputs, OptixDenoiser

denoiser = OptixDenoiser(width, height, stream=stream, upscale=True)
filtered = denoiser.apply(DenoiserInputs(color, normal, albedo, motion), reset=True)
# Consume filtered on the same stream; it is borrowed until the next apply().
denoiser.close()
```

The supplemental extension adds the temporal models and guide-layer fields
to the PyOptiX dependency's denoiser bindings. `OptixDenoiser` takes `device=`/`stream=` like other
constructors and uses that device's implicit OptiX context.

### Interactive viewer features

The optional interactive features are installed separately:

```bash
pip install -e ".[pathtracing,ui]"
```

The viewer follows the earlier hybrid viewer controls:

- WASD or arrow keys move the camera; Q/E move along the model up axis.
- Left drag looks around and the scroll wheel changes field of view.
- Right click and drag picks bodies when a `picking_factory` is supplied.
- Space pauses, Escape closes, and 0-8 select path-tracing debug buffers.
- P saves a screenshot.

`register_ui_callback()` adds application controls to the optional ImGui panel.
The panel also exposes rendering statistics, denoiser state, visualization flags,
camera state, debug buffers, picking, pause, and screenshot controls. Picking
uses an optional `picking_factory` supplied by the application when constructing
the viewer.

The backend accepts Warp arrays for meshes, transforms, colors, and material
parameters. It handles X/Y/Z up-axis conversion, mesh and instance
caching, visibility updates, and roughness/metallic PBR materials. Its default
physical-sky values and sRGB-to-linear color conversion intentionally match
the earlier hybrid viewer, including the light ground, slight haze, and soft
horizon used for untextured simulation geometry.

The four material values passed to `log_instances()` are roughness, metallic,
U subdivisions, and V subdivisions. Positive subdivision values enable a
procedural UV checker overlay; alternating cells multiply the sampled base
color by `base_color_scale` (default `0.75`). A zero subdivision disables the
overlay. The same `u_subdiv`, `v_subdiv`, and `base_color_scale` controls are
available when creating PBR or glTF materials directly.

The framework-neutral class can also be driven directly through
`log_mesh()`, `log_instances()`, `begin_frame()`, and `end_frame()`. Textured
glTF and USD scenes remain available through `PathTracer`; USD loading uses
the optional `usd-core` extra and maps UsdPreviewSurface and common NVIDIA MDL
inputs to the same internal PBR material representation. Composed DomeLight
HDR textures can be enabled explicitly with
`load_scene_from_usd(..., load_usd_environment=True)`. The framework
`log_mesh` adapter currently maps vertex colors and PBR values but does not
ingest its optional texture argument.

`PathTracer` submits ray-tracing launches directly on its rendering stream.
OptiX launch capture is unsupported; `enable_cuda_graphs` logs a diagnostic
and uses asynchronous launches. USD transform/TLAS device updates remain
independently graph-capturable.

Large dynamic arrow sets use one fixed-capacity native-curve geometry rather
than one OptiX instance per arrow. Each arrow is a constant-radius shaft plus
a linearly tapered tip. Inactive slots have zero width, so a changing contact
count does not reallocate buffers or rebuild the whole scene. By default only
the arrow GAS is rebuilt, followed by a TLAS update:

```python
material = api.create_pbr_material((0.9, 0.25, 0.03), 0.65, 0.0)
arrows = api.create_arrow_batch(
    capacity=max_contact_count,
    small_radius=0.002,
    large_radius=0.006,
    tip_length_ratio=0.2,
    material_id=material,
)
api.build_scene()

# CUDA wp.vec3 buffers with at least max_contact_count entries. The one-element
# CUDA int32 count avoids a device-to-host synchronization when contacts vary;
# it is clamped to max_contact_count on-device.
api.update_arrow_batch_device(
    arrows,
    contact_starts_cuda,
    contact_ends_cuda,
    contact_count_cuda,
    material_ids=contact_material_ids_cuda,  # Optional int32 ID per arrow.
    stream=simulation_stream,
)
```

The host equivalent is `update_arrow_batch(arrows, starts, ends)`. Per-arrow
materials use the existing per-curve-primitive material table; the API assigns
the same ID to an arrow's shaft and tip. When updating several dynamic batches,
pass `rebuild_tlas=False` to each update and call `api.rebuild_tlas()` once at
the end.

A fast GAS rebuild is deliberately the default because contact ordering may
shuffle completely between frames. Use `rebuild_gas=False` only when arrow IDs
remain spatially coherent; refitting is correct in either case, but arbitrary
reordering can substantially degrade the refitted BVH's traversal quality.

For contact vectors that only need lines, `GLLineOverlay` avoids OptiX GAS and
TLAS updates entirely. It follows Newton's wireframe renderer: a geometry
shader expands `GL_LINES` into constant-pixel-width quads. A fixed-capacity VBO
is registered with CUDA and updated directly from device arrays; the active
count and ordering may change on every frame. The path tracer's positive
view-space depth is transferred through a CUDA/OpenGL PBO to an R32F texture,
so fragments behind path-traced geometry are discarded:

```python
from warp_optix.gl import GLLineOverlay

overlay = GLLineOverlay(
    gl_viewer.gl,
    capacity=max_contact_count,
    device="cuda",
    depth_buffer=api.linear_depth_output,
    stream=gl_viewer.render_stream,
)
overlay.update_device(contact_starts_cuda, contact_ends_cuda, contact_colors_cuda)


def draw_contacts():
    camera = api.viewer.camera
    overlay.draw(
        camera.get_view_matrix(),
        camera.get_projection_matrix(),
        (gl_viewer.width, gl_viewer.height),
        camera_near=camera.near,
        camera_far=camera.far,
    )


gl_viewer.set_draw_overlay(draw_contacts)
```

Call `set_depth_buffer(api.linear_depth_output)` after a path-tracer resize and
`close()` while the OpenGL context is current. See
`examples_warp/example_warp_optix_pathtraced_contact_lines.py` for 100k
contacts with a changing count and fully shuffled ordering.

The high-level curve API also accepts native cubic Bézier segments. Their
radius is a cubic Bézier scalar using the same parameter and control topology
as position:

```python
curve = api.create_curve(control_points, radii, basis="cubic_bezier")
api.create_instance(curve)
```

Omitting indices requires `3*N + 1` control points and creates starts
`0, 3, 6, ...`; explicit starts can pack disjoint strands in one geometry.

USD loads also retain a path-addressable transform hierarchy instead of
baking composed transforms into vertices:

```python
api.load_scene_from_usd("scene.usd")
usd = api.usd_scene
body = usd.require_transform("/World/Robot/base")

# Host batch (local 4x4 matrices).
usd.update_local_transforms([body], matrices)

# Zero-staging CUDA batch on a caller-owned Warp stream.
transform_count_cuda = wp.array([capacity], dtype=wp.int32, device="cuda")
usd.update_local_transforms_device(transform_count_cuda, body_ids_cuda, local_mat44_cuda, stream=simulation_stream)
# Newton-style wp.transform + wp.vec3 scale arrays are accepted directly too.
usd.update_local_transform_trs_device(
    transform_count_cuda,
    body_ids_cuda,
    local_poses_cuda,
    local_scales_cuda,
    stream=simulation_stream,
)

# Or decouple hierarchy writes from the OptiX update.
usd.update_local_transforms_device(
    transform_count_cuda,
    body_ids_cuda,
    local_mat44_cuda,
    stream=simulation_stream,
    rebuild_tlas=False,
)
usd.update_tlas(stream=simulation_stream)
```

The ID and transform arrays define a fixed launch capacity. A CUDA kernel may
change the single active-count value inside a captured graph to update any
prefix of those arrays without graph recapture or host synchronization.

`usd.transforms` enumerates a stable `UsdTransformHandle` for every composed
transformable prim path. `usd.get_transform(path)` performs a non-throwing
lookup, `usd.get_prim(path)` accesses any prim on the retained OpenUSD stage,
and the CUDA local/world `wp.mat44` arrays are exposed after scene build. A
batch composes hierarchy levels, updates all affected OptiX instance and
motion-vector transforms, and updates the TLAS on the selected CUDA stream.
With `rebuild_tlas=False`, the device methods launch only Warp kernels and
are CUDA graph capture compatible after one warm-up call; call `update_tlas()`
outside the capture, because OptiX acceleration builds and updates cannot be
recorded into graphs. Keep the device batch length and array addresses stable
across graph replays; a batch length of one is supported. The NumPy
convenience method performs allocation/upload and is intentionally not the
graph-capture path.

The old hybrid viewer's Vulkan-backed OpenGL transform VBO was specific to its
C# bridge. Compatibility queries remain available, but return unavailable;
dynamic simulation transforms instead use Warp arrays through `log_instances()`
or `update_instance_transforms()` and trigger an OptiX TLAS refit. Transform
matrix construction is vectorized before the retained OptiX instance buffer is
updated. Window presentation uses CUDA/OpenGL interop with Warp's copy fallback
when direct registration is unavailable.

## Layout

- `warp_optix/__init__.py` — public API.
- `warp_optix/context.py` — implicit per-device contexts (`get_context`), device/stream resolution, and native-object release.
- `warp_optix/config.py` — global settings (`validation`, `log_level`, `verify_instances`).
- `warp_optix/accel.py` — build inputs, `Accel`, and `Instance`/`make_instance`.
- `warp_optix/pipeline.py` — `Pipeline`, `HitGroup`, `Launch`, and `launch`.
- `warp_optix/program.py` — rewrites Warp kernels into OptiX programs (`ProgramKind`, `kernel`).
- `warp_optix/rays.py` — `accel_query_ray`, `accel_query_ray_anyhit`, and `AccelQueryRay`.
- `warp_optix/enums.py` — flag and primitive-type enums, usable on the host and in kernels.
- `warp_optix/builtins.py` — OptiX builtin registrations.
- `warp_optix/_native/include/warp_optix_builtins.h` — device-side C++ wrappers.
- `warp_optix/native.py` — PyOptiX plus the supplemental bindings (`_backend.py`, `_ext`).
- `warp_optix/addon.py` — registers the builtins and provides the OptiX module build options.
- `warp_optix/gl/` — optional OpenGL interop viewer and line overlay.
- `warp_optix/pathtracing/` — the path tracer.

See `examples_warp/` for complete programs: `example_warp_optix_kernels.py` (minimal pipeline),
`example_warp_optix_curves.py` (built-in curves), `example_warp_optix_mixed_geometry.py`
(triangles, curves, and custom primitives under GPU-written instances),
`example_warp_optix_motion_blur.py` (vertex motion keys), and
`example_warp_optix_tiny_raytracer.py` (interactive viewer).
