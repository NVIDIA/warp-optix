# Path tracer

The path tracer is installed as `warp_optix.pathtracing`. Install
its windowing and image extras with:

```bash
pip install "warp-optix[pathtracing]"
```

`Viewer` is a standalone OptiX path-tracing viewer with OptiX temporal
denoising. `ViewerBackend` exposes a renderer-facing `log_*`
API without importing a simulation framework.

## Denoising and upscaling

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

## Interactive viewer features

The optional interactive features are installed separately:

```bash
pip install "warp-optix[pathtracing,ui]"
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

The USD loader composes references, payloads, and instance proxies, triangulates
polygon meshes, and imports UVs, normals, material subsets, DDS textures, and UDIM
sets. It does not tessellate subdivision surfaces or import curves, point
instancers, camera animation, or lights other than DomeLight.

Triangle backfaces are culled for camera, bounce, and shadow rays by default.
Pass `backface_culling=False` to `PathTracer` or `ViewerBackend`, or call
`set_backface_culling(False)`, for two-sided traversal.

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
`examples/example_pathtracing.py --scene contact_lines` for 100k
contacts with a changing count and fully shuffled ordering.

The high-level curve API also accepts native cubic Bézier segments. Their
radius is a cubic Bézier scalar using the same parameter and control topology
as position:

```python
curve = api.create_curve(control_points, radii, basis="cubic_bezier")
api.create_instance(curve)
```

Omitting indices requires `3*N + 1` control points and creates starts
`0, 3, 6, ...`; explicit starts can pack disjoint strands in one geometry. The
default `basis="linear"` makes each consecutive point pair a segment.
`material_ids` optionally assigns one material per segment; `create_mesh`
accepts the per-triangle equivalent.

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

## OpenGL display

`warp_optix.gl` provides optional pyglet-based helpers for interactive examples:
`GLInteropViewer` presents a CUDA-mapped `uint32` RGBA pixel buffer each frame without a host
copy, and `GLLineOverlay` draws depth-tested lines over it (see [Interactive viewer features](#interactive-viewer-features)).
