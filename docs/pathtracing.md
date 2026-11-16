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
DLSS Ray Reconstruction is opt-in and is never selected by `"auto"`.
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

## DLSS Ray Reconstruction

DLSS Ray Reconstruction (RR) denoises and, in its upscaling quality modes,
reconstructs a lower-resolution render at the output resolution. warp-optix uses
the RR runtime that ships with the NVIDIA driver since R620
(`ngx-builtins/nvngx_dlssd`), the same way Blender Cycles does: it compiles the
NGX SDK in header-only mode (`NVSDK_NGX_HEADER_ONLY`), links no NVIDIA library,
and distributes no DLSS binary.

DLSS support is **off by default**, and published wheels are built without it.
Header-only mode compiles portions of the proprietary DLSS SDK into the
extension: the NGX runtime loader (`nvsdk_ngx_loader.h`) and the SDK's
declarations and parameter names. Build it in explicitly; the SDK headers and
license (8 files, pinned commit, SHA-256 verified) are fetched at build time
into the git-ignored `third_party/` and never committed:

```bash
WARP_OPTIX_ENABLE_DLSS=ON pip install .
# or: pip install . -C cmake.define.WARP_OPTIX_ENABLE_DLSS=ON
# DLSS_ROOT=/path/to/DLSS-SDK uses a local SDK instead of downloading headers.
```

```python
from warp_optix.pathtracing import PathTracer, dlss_rr_requirements

print(dlss_rr_requirements())  # {'supported': True, ...} or a 'reason' (e.g. driver older than R620)
api = PathTracer(denoiser="dlss", dlss_quality="balanced")
```

`dlss_quality` defaults to `"balanced"`. Other modes are `"dlaa"` (denoise
at full resolution), `"max_quality"`, `"max_perf"`, or `"ultra_performance"`; the
upscaling modes render at the size NGX recommends for the output and
reconstruct to full resolution. The model preset defaults to `F` (the DLSS 4.5
transformer model). Without a DLSS build or on older drivers, selecting `"dlss"`
raises, and `dlss_rr_requirements()` reports why.

Custom renderers use `DlssDenoiser` directly. It needs every guide of
`DenoiserInputs`, including linear `depth` and `specular_albedo`, plus the
camera's world-to-view and view-to-clip matrices and the frame's jitter:

```python
from warp_optix.pathtracing import DenoiserInputs, DlssDenoiser

render_size = DlssDenoiser.optimal_render_size((1920, 1080), "balanced")
denoiser = DlssDenoiser(*render_size, output_size=(1920, 1080), quality="balanced", stream=stream)
output = denoiser.apply(inputs, view_matrix=view, projection_matrix=proj, jitter=(jx, jy), reset=False)
denoiser.close()
```

RR is a large transformer model, and its cost depends strongly on the GPU
generation. Measured for the reconstruction step alone at 1920x1080 output on an
RTX 3080 Laptop GPU (Ampere): about 60 ms per frame in DLAA mode and 24 ms in
balanced mode. Newer GPUs run the model much faster.

By default warp-optix identifies itself to NGX with its own project ID, shared
by every application built on it. Applications that enable DLSS should register
their own GUID before creating a DLSS denoiser:

```python
from warp_optix.pathtracing import set_dlss_project_id

set_dlss_project_id("a0f57b54-1daf-4934-90ae-c4035c19df04")  # your app's fixed GUID
```

`feature_search_path` (on `DlssDenoiser` and `dlss_rr_requirements`) names a
folder with an application-supplied `nvngx_dlssd` library that takes precedence
over the driver's runtime. Distributing that library is governed by your own
DLSS SDK license, and its NVIDIA signature must stay intact.

### DLSS licensing

The DLSS SDK is licensed under the NVIDIA RTX SDKs License, not Apache-2.0.
Building with `WARP_OPTIX_ENABLE_DLSS=ON` means using the SDK and accepting its
license (`third_party/dlss-headers-*/LICENSE.txt`), and DLSS-enabled builds
install it as `warp_optix/licenses/NVIDIA_RTX_SDKs_LICENSE.txt`. Among other
terms, DLSS:

- is licensed for systems with NVIDIA GPUs only;
- may be deployed in a cloud service only on systems that consume NVIDIA vGPU
  software;
- may not be used where its failure could cause personal injury, death, or
  catastrophic loss (for example avionics, navigation, military, medical, or
  life-support applications) without a separate agreement with NVIDIA.

If you distribute an application that uses DLSS through warp-optix, the license
also requires you to notify NVIDIA before its commercial release
(https://developer.nvidia.com/sw-notification) and to credit NVIDIA DLSS in its
about box or credits, if it has one. See `THIRD_PARTY_NOTICES.txt`; license
questions go to nvidia-rtx-license-questions@nvidia.com.

NVIDIA, DLSS, and DLSS-RR are trademarks and/or registered trademarks of NVIDIA
Corporation in the U.S. and other countries.

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

The high-level curve API also accepts native cubic BÃ©zier segments. Their
radius is a cubic BÃ©zier scalar using the same parameter and control topology
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
