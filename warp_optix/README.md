# warp_optix

Warp ray tracing using PyOptiX and a small supplemental native extension.

This package replaces the `warp.optix` Python module that used to live inside
the `warp` source tree. It registers itself with warp on import:

```python
import warp as wp
import warp_optix as wo  # registers OptiX builtins, headers, and entry-point specs
```

Once imported, you can decorate kernels with the OptiX entry-point kinds:

```python
@wp.struct
class LaunchParams:
    value: wp.uint32

@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def raygen_program(params: LaunchParams):
    idx = wp.optix_get_launch_index()
    ...
```

## Install

Install from the repository root; PyOptiX is installed as a dependency:

```bash
pip install -e .
```

See [migration notes](../docs/migration-0.2.md) for the owned API introduced in 0.2.

`warp_optix` requires `warp-lang>=1.17` and uses Warp's supported addon
registration, external-entry kernel, and AOT compilation APIs. A
version-checked private compatibility adapter remains only for older development
environments; it does not patch or overwrite the installed Warp package.


## OptiX cooperative vectors

OptiX 9 cooperative vectors are native SDK facilities, not Warp operations.
The Python extension exposes the host-side SDK pieces needed to prepare
matrices:

- `DEVICE_PROPERTY_COOP_VEC` and `DevicePropertyCoopVecFlags` for support
  queries.
- `CoopVecElemType`, `CoopVecMatrixLayout`, and
  `CoopVecMatrixDescription`.
- `DeviceContext.coopVecMatrixComputeSize()` and
  `DeviceContext.coopVecMatrixConvert()`.

The conversion call is asynchronous and follows OptiX's 64-byte alignment and
allocation-lifetime requirements. The higher-level neural-texture uploader
synchronizes before returning and owns all GPU allocations.

Importing `warp_optix` also registers Warp spellings of the OptiX device
intrinsics. They are available only inside `@warp_optix.optix_kernel`
programs and use explicit output vectors, which is the convention required by
Warp's public addon API:

```python
Vec32h = wp.types.vector(length=32, dtype=wp.float16)
Vec16h = wp.types.vector(length=16, dtype=wp.float16)

@wo.optix_kernel(wo.OptixKernelType.CLOSEST_HIT)
def closest_hit(params: LaunchParams):
    inputs = Vec32h(wp.float16(0.0))
    outputs = Vec16h()
    wp.optix_coop_vec_matmul_bias_fp8_e4m3(
        inputs,
        params.matrices,
        params.weight_offset,
        params.biases,
        params.bias_offset,
        wp.uint32(0),
        outputs,
    )
```

The registered family includes load, conversion, elementwise arithmetic,
`exp2`, `log2`, `tanh`, fused multiply-add, reductions, outer products,
matrix-size queries, and matrix multiply variants for FP16, FP8 E4M3/E5M2,
signed int8, and unsigned int8. Named matrix variants use an
inference-optimal, non-transposed matrix with the element interpretation in
the function name.

## Geometry and transform queries

Current triangle hits expose barycentrics, object-space vertices, generic and
triangle-specific face predicates. Built-in curve hits expose their object-space
control points as fixed Warp matrices whose rows are `(x, y, z, radius)`:

| Query family | Return type |
| --- | --- |
| Linear curve | `2 x 4` float matrix |
| Quadratic B-spline and ribbon | `3 x 4` float matrix |
| Cubic B-spline, Catmull-Rom, and cubic Bezier | `wp.mat44` |

Ribbons additionally expose their two intersection parameters and derived
normal. Geometry current-hit queries are valid in any-hit and closest-hit
programs for the corresponding primitive type.

The current transform list can be inspected with
`wp.optix_get_transform_list_size()`,
`wp.optix_get_transform_list_handle(index)`, and
`wp.optix_get_transform_type_from_handle(handle)`. Composite affine
transforms are returned as ordinary homogeneous `wp.mat44` values:

```python
object_to_world = wp.optix_get_object_to_world_transform_matrix()
world_to_object = wp.optix_get_world_to_object_transform_matrix()
```

The current transform matrix and transform-list queries are valid in
intersection, any-hit, and closest-hit programs.

Random-access geometry fetches and raw SDK transform pointers are deliberately
not exposed.

## Neural textures

`warp_optix.neural_texture` is a small, independent inference-on-sample
pipeline inspired by NVIDIA's neural material texture research. It stores two
4-bit latent grids and a fixed per-material `32 -> 32 -> 32 -> 16` decoder.
Portable `.wnt` files contain memory-mappable packed latents, row-major FP16
biases and FP8-representable weights, checksums, channel mappings, and JSON
metadata. On upload, weights are converted once into OptiX's device-specific
FP8 E4M3 inference-optimal layout.

Only offline compression depends on `warp-nn`:

```bash
pip install -e ".[training]"
python examples_warp/example_warp_optix_neural_texture.py \
  --output material.wnt --reconstruction material-decoded.npy
```

The Python API is equally direct:

```python
import numpy as np
from warp_optix.neural_texture import (
    compress_texture_set, load_asset, save_asset, upload_asset,
)

textures = {
    "albedo": np.load("albedo.npy", allow_pickle=False),
    "normal": np.load("normal.npy", allow_pickle=False),
    "roughness": np.load("roughness.npy", allow_pickle=False),
}
result = compress_texture_set(
    textures,
    color_spaces={"albedo": "srgb"},
    channel_weights={"normal": 2.0},
)
save_asset("material.wnt", result.asset)
print(result.psnr, result.psnr_by_texture, result.selection_reason)

# Loading, upload, and OptiX inference do not import warp-nn.
asset = load_asset("material.wnt")
runtime = upload_asset(asset, context)
```

All maps must have the same resolution and may contain one to four channels;
the set may contain at most 16 channels. Integer images are normalized
automatically, while floating-point images must already be in `[0, 1]`.
`color_spaces` records metadata and does not transform input values.
`compress_texture()` remains the short single-image helper.

Inference needs one device view and one Warp call:

```python
from warp_optix.neural_texture import NeuralTextureView, neural_texture_sample

@wp.struct
class Params:
    material: NeuralTextureView

@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def closest_hit(params: Params):
    values = neural_texture_sample(params.material, wp.vec2(0.25, 0.75))
    albedo = wp.vec3(values[0], values[1], values[2])

params.material = runtime.device_view()
```

The device path is Warp-generated OptiX code plus cooperative-vector
intrinsics and has no `warp-nn` dependency.

By default, compression trains two candidates (latent scales 4 and 5) and keeps
the smaller asset only if every named texture stays within 0.5 dB of the larger
candidate. This is a bounded heuristic, not an exhaustive quality search. No
settings are required: `compress_texture(image)` and
`compress_texture_set(textures)` both select automatically. Both use exactly
the same inference function and asset format.

Training and quantization-aware refinement run in captured CUDA graphs.
Source values are staged as FP16 for training and FP32 for quality scoring:
allow six bytes per source channel per pixel plus model, scoring and training
buffers. Training loss and scoring use `tile_sum` with one atomic add per tile.
Refinement trains both decoder and latents through 4-bit quantization using
straight-through gradients and projected FP8 E4M3 weights, without increasing
storage or inference work.

On CUDA, quality scoring calls the same OptiX texel sampler used by rendering.
The candidates share a deterministic, stratified sample of up to 65,536 pixels;
only per-channel error sums return to the CPU. `CompressionResult.psnr` and
`psnr_by_texture` are therefore estimates when `evaluation_pixels` is smaller
than the image pixel count. CPU compression uses the NumPy reference decoder.
`selection_reason` explains the automatic choice.

Advanced users can override `latent_scale` to train just one candidate;
explicit-scale compression scores every pixel (with larger scoring buffers).
Larger scales use smaller grids. For large images the latent rate is roughly
`40 / latent_scale**2` bits per material texel, shared across all channels:

| `latent_scale` | Latent bits/texel | Intended use |
| --- | ---: | --- |
| 3 | 4.44 | More detail |
| 4 | 2.50 | Larger automatic candidate |
| 5 | 1.60 | Smaller automatic candidate |

All use the same decoder and number of latent samples; grid sizes can still
affect cache behavior. Increasing `steps` can improve quality without changing
asset size or device code. The default is 1,000 training steps followed by
250 refinement steps per candidate.

The format is deliberately safe to memory-map: it uses no pickle data and
validates shapes, dtypes, bounds, alignment, and SHA-256 payload checksums.

This is a research-informed, compact integration rather than a reimplementation
of NVIDIA RTXNTC. Version 1 has one fixed architecture, two learned latent
feature grids, base-level decoding, wrap addressing, no entropy coding, and no stochastic texture
filtering or renderer material integration. Its `.wnt` format is not
compatible with RTXNTC's `.ntc` format. For production compression, full mip
pyramids, and filtering guidance, consult NVIDIA's
[Random-Access Neural Compression of Material Textures](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures),
[RTXNTC SDK](https://github.com/NVIDIA-RTX/RTXNTC), and
[inference-on-sample integration guide](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/docs/integration/InferenceOnSample.md).

## Path-tracing viewer

The example path tracer is also installed as `warp_optix.pathtracing`. Install
its windowing and image extras with:

```bash
pip install -e ".[pathtracing]"
```

`PathTracingViewer` is a standalone OptiX path-tracing viewer with DLSS Ray
Reconstruction. `PathTracingViewerBackend` exposes a renderer-facing `log_*`
API without importing a simulation framework.

### Denoising and upscaling

`PathTracerAPI` and `PathTracingViewer` accept the same optional backend
selector and support switching without rebuilding the scene:

```python
from warp_optix.pathtracing import PathTracerAPI

api = PathTracerAPI(denoiser="dlss", dlss_quality="quality", optix_upscale=True)
api.initialize()
api.set_denoiser("optix")
api.set_denoiser("dlss")  # Retains the DLSS quality setting.
api.set_denoiser("optix", optix_upscale=False)  # Native-resolution denoising.
```

`denoiser="auto"` prefers DLSS RR, then OptiX, then raw rendering. Explicit
`"dlss"` and `"optix"` selections require successful initialization;
`"none"` disables denoising. Inspect `active_denoiser` and `denoiser_error`
to see the selected backend and automatic fallback diagnostics. Omitting
`denoiser` preserves the existing `enable_dlss_rr` behavior, including its
raw-rendering fallback. An explicit selector takes precedence over that flag.

DLSS queries NGX for input dimensions using the existing `dlss_quality` modes.
OptiX uses temporal denoising at native resolution by default;
`optix_upscale=True` selects its temporal 2x upscaling model. Odd output sizes
use rounded-up input dimensions and crop the extra output row or column.
Switches and resizes reset reconstruction history and affected CUDA graphs.
Use `reset_temporal_history()` after a discontinuous scene change. OptiX
respects `samples_per_frame`; DLSS continues to render one sample per frame.
Both backends share the renderer's guide buffers and final tonemapping path.
The rendering panel exposes backend selection and their quality settings.

For custom Warp renderers, `warp_optix.pathtracing` also exports
`DenoiserInputs` and `OptixDenoiser`. Pass float4 HDR color, float4 world-space
normals, float4 albedo, and float2 current-to-previous motion in input pixels
(without jitter). The wrapper manages scratch/state allocations, temporal
history, and stream ordering:

```python
from warp_optix.pathtracing import DenoiserInputs, OptixDenoiser

denoiser = OptixDenoiser(context, width, height, stream=stream, upscale=True)
filtered = denoiser.apply(DenoiserInputs(color, normal, albedo, motion), reset=True)
# Consume filtered on the same stream; it is borrowed until the next apply().
denoiser.close()
```

The supplemental extension adds the temporal models and guide-layer fields
to the PyOptiX dependency's denoiser bindings.
The denoiser takes an owned `warp_optix.Context`.

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
The panel also exposes rendering statistics, DLSS state, visualization flags,
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
glTF and USD scenes remain available through `PathTracerAPI`; USD loading uses
the optional `usd-core` extra and maps UsdPreviewSurface and common NVIDIA MDL
inputs to the same internal PBR material representation. Composed DomeLight
HDR textures can be enabled explicitly with
`load_scene_from_usd(..., load_usd_environment=True)`. The framework
`log_mesh` adapter currently maps vertex colors and PBR values but does not
ingest its optional texture argument.

`PathTracerAPI` submits ray-tracing launches directly when DLSS RR or the
temporal OptiX denoiser is active. DLSS resource event queries are forbidden
during CUDA stream capture, and replaying a captured OptiX launch on the tested
runtime snapshots camera/sample parameters, freezing temporal inputs.
Raw rendering retains the CUDA graph option; USD transform/TLAS device updates
remain independently graph-capturable.
Pass `enable_cuda_graphs=False` to diagnose a driver or
capture compatibility issue; `api.cuda_graph_active` reports successful
capture after the first rendered sample.

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
overlay = wo.GLLineOverlay(
    gl_viewer.gl,
    capacity=max_contact_count,
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
`destroy()` while the OpenGL context is current. See
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
usd.update_local_transforms_device(
    transform_count_cuda, body_ids_cuda, local_mat44_cuda, stream=simulation_stream
)
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

`usd.transforms` enumerates a stable `USDTransformHandle` for every composed
transformable prim path. `usd.get_transform(path)` performs a non-throwing
lookup, `usd.get_prim(path)` accesses any prim on the retained OpenUSD stage,
and the CUDA local/world `wp.mat44` arrays are exposed after scene build. A
batch composes hierarchy levels, updates all affected OptiX instance and
motion-vector transforms, and updates the TLAS on the selected CUDA stream.
The device methods and allocation-free TLAS `UPDATE` path are CUDA graph
capture compatible after one warm-up call (which compiles kernels and sizes
the reusable TLAS buffers). Keep the device batch length and array addresses
stable across graph replays; a batch length of one is supported. The NumPy
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

- `warp_optix/__init__.py` — public re-exports of the runtime API.
- `warp_optix/_runtime/` — runtime helpers for OptiX integration.
- `warp_optix/_builtins.py` — OptiX `add_builtin(...)` registrations.
- `warp_optix/_codegen.py` — `OptixKernelType` enum and codegen entry-point specs.
- `warp_optix/_compat.py` — vanilla Warp private-API fallback.
- `warp_optix/_native/include/warp_optix_builtins.h` — device-side C++ wrappers.
- `warp_optix/_addon.py` — runs on import; wires the above into warp.

## Custom primitives

Custom geometry uses OptiX AABBs plus a Warp intersection program. The runtime
helper accepts either `(N, 6)` AABBs or `(N, 2, 3)` min/max pairs:

```python
gas = context.create_custom_primitive_gas(
    aabbs,
)

pipeline = context.create_pipeline(
    ptx,
    raygen_program,
    miss_program,
    closest_hit_program,
    num_payload_values=2,
    num_attribute_values=3,
    intersection_entry=intersection_program,
)
```

See `examples_warp/example_warp_optix_motion_blur.py` for temporal sampling
across a two-key triangle deformation.

Inside an `INTERSECTION` kernel, use `wp.optix_get_object_ray_origin()`,
`wp.optix_get_object_ray_direction()`, and `wp.optix_report_intersection()`.
The latter accepts a hit distance, hit kind, and zero to eight `wp.uint32`
attributes. Closest-hit and any-hit kernels can retrieve them with
`wp.optix_get_attribute_0()` through `wp.optix_get_attribute_7()`.

The context retains geometry and pipeline allocations until it closes. Use
`gas.handle` in device launch parameters and `gas.buffers` for geometry arrays.

For mixed geometry, build one GAS per geometry type, place them under an IAS
with `create_instance_acceleration_structure()`, and pass one `HitKernel` per
SBT record through `hit_groups`. The pipeline helper infers combined primitive
flags and stores all hit records contiguously. See
`examples_warp/example_warp_optix_mixed_geometry.py` for triangle, native curve,
and analytical custom geometry sharing one pipeline.

`pipeline.hit_group_offsets` contains the SBT offsets in hit-group order.
Use these when constructing `wo.Instance` descriptors.

Warp kernels also expose the common OptiX hit context directly: ray time,
flags and visibility mask; instance ID/index; primitive, SBT GAS and GAS
handles; front/back-face tests; and point/vector/normal transforms in both
directions. The names follow Warp's existing snake-case convention, for
example `wp.optix_get_instance_index()` and
`wp.optix_transform_point_from_world_to_object_space()`.

Triangle any-hit and closest-hit kernels can call
`wp.optix_get_triangle_vertex_data()`. It returns a `wp.mat33` whose rows are
the three object-space vertices of the current triangle; this current-hit form
does not require random vertex access or extra launch-parameter arrays.

Acceleration-structure builders accept `compact=True` for static data and a
normal OptiX `build_flags` value for advanced use. For dynamic geometry, build
with `BUILD_FLAG_ALLOW_UPDATE`, update the retained Warp buffer, then refit in
place:

```python
gas = context.create_triangle_gas(
    vertices,
    indices,
    build_flags=wo.BuildFlags.ALLOW_UPDATE,
)
gas.buffers["d_vertices"].assign(updated_vertices)
gas.refit()
```

Compaction and update are deliberately mutually exclusive in these helpers:
compaction targets immutable geometry, while update retains the original output
capacity required by OptiX refits.

For vertex motion blur, pass triangle vertices as `(K, N, 3)` motion keys and
enable motion on the pipeline:

```python
gas = context.create_triangle_gas(
    vertex_keys, indices, motion_time_range=(0.0, 1.0)
)
pipeline = context.create_pipeline(
    ...,
    uses_motion_blur=True,
)
```

`create_curve_gas()` builds native round linear, B-spline, Catmull-Rom, and
Bezier curves. Associate its hit record with OptiX's built-in intersector by
setting `HitKernel(builtin_intersection_type=curve_type)`. Curve shaders can
read the current parameter with `wp.optix_get_curve_parameter()`. Curve vertex
and width arrays may also contain motion keys.

Exception, direct-callable, and continuation-callable programs use the same
`optix_kernel()` decorator as other stages. Pass them through
`exception_entry`, `direct_callable_entries`, or
`continuation_callable_entries` when creating the pipeline. The helper packs
their SBT records and computes the additional stack requirements. Callable
indices are exposed by `pipeline.direct_callable_indices` and
`pipeline.continuation_callable_indices`.

Callables intentionally use a no-argument, void interface:
`wp.optix_direct_call(index)` or `wp.optix_continuation_call(index)`. They can
read launch parameters and write referenced Warp arrays. Arbitrary typed
callable arguments and returns are outside Warp's current external entry ABI
and are therefore not exposed. Exception programs can use
`wp.optix_get_exception_code()` and detail accessors; user code can raise an
exception with `wp.optix_throw_exception()` and up to eight `wp.uint32`
details.
