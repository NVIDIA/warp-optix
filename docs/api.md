# API reference

The [README](../README.md) walks through warp-optix by example. This page has the details: argument
lists, lifetime rules, and every device builtin family. The path tracer has its own
[guide](pathtracing.md).

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
exception — declares any subset of the raygen arguments **by name with identical types** (a
generic `typing.Any` argument takes the type of the launch parameter it binds):

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


sbt = wo.SbtLayout(miss=background, hit_groups={"triangles": wo.HitGroup(closest_hit=shade)})
pipeline = wo.Pipeline(raygen, sbt=sbt, payload_words=3)
wo.launch(pipeline, dim=image.shape, inputs=[accel.id, wp.vec3(1.0, 0.5, 0.2), image])
```

`wp.optix_trace(id, start, dir, max_t, payload, min_t=0.0, time=0.0, mask=0xFF, flags=0, sbt_offset=0,
sbt_stride=1, miss_index=0)` follows `wp.mesh_query_ray(id, start, dir, max_t)`. Trailing arguments
may be omitted but not skipped by keyword, because Warp's external-builtin API has no default
arguments; `wp.optix_traverse()` takes the same arguments with an optional payload. Flags, masks,
SBT indices, and register indices accept any integer type, so `wo.RayFlags` members need no cast.

`wo.Pipeline(raygen, *, sbt, callables, exception, exception_flags,
payload_words, attribute_words, max_trace_depth, max_traversable_depth, max_callable_depth, graph_flags, motion_blur,
max_registers, module_options, device)` compiles all programs into one module and builds the shader binding table:

- `sbt` is a [`wo.SbtLayout`](#sbt-layouts) of ray types, miss records, and hit groups. The
  default is one ray type with a miss record without a program and one empty hit group.
- `callables` are callable programs; `index` is the position in the list. Call a direct callable
  with `wp.optix_direct_call(index)`, and declare a continuation callable with
  `@wo.kernel(kind=wo.ProgramKind.CONTINUATION_CALLABLE)` and call it with
  `wp.optix_continuation_call(index)`. Callables use a no-argument, void interface: they read
  launch parameters and write referenced Warp arrays. Miss and callable entries may be `wo.Record(kernel, data=)` to carry
  [record data](#per-record-sbt-data).
- `exception` enables user exceptions. Raise one with `wp.optix_throw_exception(code, ...)` (up
  to eight optional details) and read it with `wp.optix_get_exception_code()` and
  `wp.optix_get_exception_detail(i)`.
- `exception_flags` selects `wo.ExceptionFlags.NONE`, `STACK_OVERFLOW`,
  `TRACE_DEPTH`, and `USER`, including bitwise combinations. When omitted,
  the default is `USER` if an exception program is supplied, otherwise `NONE`.
  For example, `exception_flags=wo.ExceptionFlags.STACK_OVERFLOW | wo.ExceptionFlags.TRACE_DEPTH`
  enables those debugging checks. Include `USER` when using `optix_throw_exception`.
- `module_options` sets Warp module options (`wp.set_module_options`) for the pipeline's module,
  for example `module_options={"fast_math": True}`. Options not given there come from the Warp
  modules of the programs' kernels, which must agree; a conflict raises `ValueError` naming the
  option, rather than silently using one kernel's setting. As in Warp, `"mode": "debug"` (or
  `wp.config.mode = "debug"`) makes the programs debuggable: OptiX then compiles them unoptimized
  with full debug information, for source-level debugging in Nsight.
- `max_trace_depth` bounds recursive `optix_trace` calls, `max_traversable_depth` the depth of the
  traversable graph, and `max_callable_depth` how deeply callables call callables (default 1). They
  size the pipeline's stack.

`wo.launch(pipeline, dim, inputs=None, outputs=None, *, device=None, stream=None, record_cmd=False)`
mirrors `wp.launch`: `inputs` and `outputs` are concatenated into the raygen arguments, and a raygen
that calls `wp.tid()` must unpack as many indices as `dim` has entries.
It packs the launch parameters on the host, copies them to the device on `stream`, and then launches
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

### SBT layouts

`wo.SbtLayout(ray_types=("radiance",), miss=None, hit_groups={"default": wo.HitGroup()})` is the
one description of a pipeline's miss and hit records. It is plain, validated, immutable data that
owns no native resources, so it can be built before any scene or pipeline and reused across
devices:

```python
sbt = wo.SbtLayout(
    ray_types=("radiance", "shadow"),
    miss={"radiance": radiance_miss, "shadow": shadow_miss},
    hit_groups={
        "glass": {"radiance": wo.HitGroup(closest_hit=glass_hit), "shadow": wo.HitGroup(any_hit=glass_shadow)},
        "metal": {"radiance": wo.HitGroup(closest_hit=metal_hit), "shadow": wo.HitGroup(any_hit=metal_shadow)},
    },
)
print(sbt.describe())
metal_offset = sbt.get_sbt_offset("metal")  # 2 records; use in wo.make_instance()
shadow_index = sbt.ray_type_index("shadow")  # trace SBT offset and miss index
trace_stride = sbt.trace_stride  # 2 records
pipeline = wo.Pipeline(raygen, sbt=sbt, payload_words=1)
```

`miss` maps every ray type to a miss program, a `wo.Record`, or `None` (a record without a
program). `hit_groups` maps each group name to a `wo.HitGroup(closest_hit=, any_hit=,
intersection=, builtin=, data=)` per ray type. A bare miss entry or `HitGroup` applies to every
ray type, so a single ray type needs no inner mapping:
`wo.SbtLayout(miss=background, hit_groups={"triangles": wo.HitGroup(closest_hit=shade)})`.
Programs are ordinary Warp kernels, with their kind determined by their slot.
`builtin=wo.PrimitiveType.*` selects OptiX's built-in curve intersector; a group with an
`intersection` program serves custom primitives; each group uses the same geometry kind for every
ray type. `HitGroup()` is an empty hit group for hit-object traversal. Primitive-type flags are
derived from the groups. Ray names must be unique, and the trace stride supports at most 15 ray
types.

Records are group-major and ray-type-minor in mapping order. Pass `get_sbt_offset(name)` to the
Warp kernel that writes instances with `wo.make_instance()` (or use it as the GAS's offset), and
trace with `sbt_offset=ray_type_index(name)`, `sbt_stride=trace_stride`, and
`miss_index=ray_type_index(name)`. Offsets and strides count records, not bytes.
`pipeline.sbt_layout` returns the layout. See `examples/example_mixed_geometry.py` for a complete
example.

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
sbt = wo.SbtLayout(
    miss=wo.Record(background, data=sky),  # background also loads a Material
    hit_groups={"red": wo.HitGroup(closest_hit=shade, data=red), "blue": wo.HitGroup(closest_hit=shade, data=blue)},
)
pipeline = wo.Pipeline(raygen, sbt=sbt, payload_words=3)
red.color = wp.vec3(1.0, 0.5, 0.0)
pipeline.update_sbt_data()  # re-upload every record's data; no rebuild
```

Records in one section (miss, hit group, callables) may use different struct types; the
section's stride fits the largest. The device does not check the type, so load the type that
was stored. Arrays referenced by record data stay alive with the pipeline, and must be on its
device. `update_sbt_data(stream=None)` is ordered with launches on that stream, like writing a
`wp.array`, and does not wait on the host.


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

- `getCoopVecProperty(context)` and `DevicePropertyCoopVecFlags` for support queries.
- `CoopVecElemType`, `CoopVecMatrixLayout`, and `CoopVecMatrixDescription`.
- `coopVecMatrixComputeSize(context, ...)` and `coopVecMatrixConvert(context, ...)`.

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
