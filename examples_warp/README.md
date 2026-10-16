# Warp + OptiX examples

These examples were moved from `warp/examples/core/pyoptix/` into this directory.

Run these commands from the repository root after installing the package with
the example extras (see the root [README](../README.md) for system prerequisites):

```bash
uv sync --python 3.12 --extra pathtracing --extra ui --extra usd --extra training
uv run --no-sync python examples_warp/neural_texture/compress.py --output material.wnt
uv run --no-sync python examples_warp/example_warp_optix_kernels.py
uv run --no-sync python examples_warp/example_warp_optix_curves.py
uv run --no-sync python examples_warp/example_warp_optix_motion_blur.py
uv run --no-sync python examples_warp/example_warp_optix_mixed_geometry.py

uv run --no-sync python examples_warp/example_warp_optix_tiny_raytracer.py
uv run --no-sync python examples_warp/example_warp_optix_basic_pathtracing.py
uv run --no-sync python examples_warp/example_warp_optix_basic_pathtracing.py --denoiser optix --optix-upscale
uv run --no-sync python examples_warp/example_warp_optix_pathtraced_hairball.py
uv run --no-sync python examples_warp/example_warp_optix_pathtraced_arrowball.py
uv run --no-sync python examples_warp/example_warp_optix_pathtraced_contact_lines.py
uv run --no-sync python examples_warp/example_warp_optix_usd_pathtracing.py path/to/scene.usd
```

With pip, `pip install -e ".[pathtracing,ui,usd,training]"` installs the same set.

The kernels, curves, motion-blur, mixed-geometry, and tiny-raytracer examples
use the core `warp_optix` API directly (see the [package README](../warp_optix/README.md)):
OptiX programs are ordinary `@wp.kernel(module="unique")` kernels, the raygen
kernel's arguments are the launch parameters, acceleration structures are built
from Warp device arrays with `wo.build_accel()`, and `wo.launch()` mirrors
`wp.launch()`. `example_warp_optix_kernels.py` is the smallest complete
pipeline; `example_warp_optix_mixed_geometry.py` combines triangle, built-in
curve, and custom-intersection geometry under instances written on the GPU with
`wo.make_instance()`; `example_warp_optix_motion_blur.py` builds a two-key
motion GAS. The image examples write a BMP (`--output`); the tiny ray tracer
opens an OpenGL viewer with `--live` (`--max-frames N` exits after N frames).

The neural-texture example (`neural_texture/compress.py`, see
[neural_texture/README.md](neural_texture/README.md)) procedurally generates a
material texture (or loads an HWC NumPy array with `--input`), trains packed
latents and a small decoder with the optional warp-nn backend, writes a
memory-mappable `.wnt` asset, and reports its stored size and FP8-projected
reconstruction PSNR. Compression needs the `training` extra. Loading and
OptiX inference do not depend on warp-nn.

The hair-ball example procedurally packs thousands of randomized, tapered
helical strands into one native round-cubic-Bezier curve geometry. Twelve saturated
materials form gently rippled rainbow bands from the bottom to the top without
entering the pink part of the hue wheel. Use
`--hair-count`, `--segments`, `--hair-length`, `--curl-radius`, and
`--curl-turns` to change its shape; `--seed` makes variations reproducible.

The arrow-ball example renders true path-traced arrows with the reusable native
OptiX `ArrowBatch` API. Each arrow uses a constant-radius shaft and tapered tip;
both are round-linear curve primitives sharing one per-arrow material. The
contact-lines example instead draws 100k changing direction indicators through
the depth-aware CUDA/OpenGL overlay API. It is intended for the fastest dynamic
contact visualization and does not add those lines to the path-traced scene.

Triangle backfaces are culled by default for camera, bounce, and shadow rays.
Pass `backface_culling=False` to `PathTracerAPI`, `PathTracingRenderer`, or
`PathTracingViewerBackend` for globally two-sided traversal. The setting can be
changed at runtime with `api.set_backface_culling(False)`. The glTF and USD
examples expose the same behavior as `--no-backface-culling`.

The full PBR path tracer also accepts native OptiX round-linear and
round-cubic-Bezier curves through the same reusable geometry/instance API as
triangle meshes:

```python
import numpy as np

from warp_optix.pathtracing import PathTracerAPI

api = PathTracerAPI()
api.initialize()
material = api.create_pbr_material((0.8, 0.2, 0.05), roughness=0.35, metallic=0.0)

points = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)
radii = np.array((0.03, 0.08, 0.03), dtype=np.float32)
curve_id = api.create_curve(
    points,
    radii,
    material_id=material,
    material_ids=np.array((material, material), dtype=np.uint32),
)
instance_id = api.create_instance(curve_id)
api.build_scene()
```

Radii are specified per control point and interpolated using the selected curve
basis. The default `basis="linear"` makes every consecutive point pair a
segment; `basis="cubic_bezier"` consumes four control points per segment. Pass
explicit
`segment_indices` containing each segment's starting control-point index to
store multiple disjoint strands in one geometry. Curve instances support the
same transforms, visibility updates, material overrides, and TLAS rebuilds as
mesh instances. `material_ids` optionally selects one shared material-table
entry per curve segment; `create_mesh` accepts the equivalent per-triangle
array. Curves use the existing PBR material, lighting, alpha,
transmission, guide-buffer, and shadow paths; USD curve import is not required.

The basic path-tracing example automatically downloads its default A Beautiful
Game scene into the platform cache on first use. On Linux this defaults to
`~/.cache/warp_optix/assets/`. Pass `--scene-gltf PATH` to use a local `.gltf`
or `.glb` scene instead.

The USD path-tracing example loads a user-supplied composed USD stage and maps
UsdPreviewSurface plus common NVIDIA OmniPBR/OmniGlass MDL inputs to the
renderer's glTF-style PBR materials. It needs the `usd` extra:

```bash
uv run --no-sync python examples_warp/example_warp_optix_usd_pathtracing.py \
  "D:\Sample_Scenes_NVD@10013\Samples\Marbles\Marbles_Assets.usd"
```

DDS images use Pillow's built-in decoder; no separate DDS dependency is
needed. UDIM sets are discovered from `<UDIM>` paths, assembled into grid
atlases, and addressed with normalized texture transforms. OpenUSD composes
referenced USD layers, payloads, and instance proxies before the
loader imports polygon meshes, UVs, normals, material subsets, base color,
normal, ORM, separate roughness/metallic maps, emissive, and glass parameters.
Separate linear maps are packed into glTF's roughness/metallic channels during
parallel texture decoding. The example limits each texture or UDIM tile to
2048 pixels by default. Stitched atlases grow with their tile count, so three
2048-pixel tiles produce a 6144-pixel-wide atlas. Use a smaller value for very
large aggregate stages such as `Marbles_Assets.usd`, or pass
`--max-texture-size 0` to keep full tile resolution.
Pass `--load-usd-environment` to opt into the first composed lat-long
`DomeLight` texture (`--usd-environment-scale` adjusts its intensity). The
stage is composed only once; texture decoding is parallelized and identical
face corners are shared to reduce CPU load, upload size, and BLAS memory.
Both path-tracing examples use Newton ViewerOptix's display defaults (0.68
exposure, 1.08 contrast, and 1.1 saturation), with command-line overrides.
The USD example first uses the camera selected by `UsdRenderSettings` or the
stage's `customLayerData.cameraSettings.boundCamera`; Marbles therefore starts
from `/stage/Overview`. If no suitable perspective camera is authored, it
frames transform-aware world-space geometry bounds. Use `--usd-camera PATH`
to select a camera explicitly or `--no-usd-camera` to force bounds framing.
Extreme flat ground planes are excluded from fallback camera fitting when
other geometry is present, but remain loaded and rendered.
The loader triangulates authored polygons but does not tessellate subdivision
surfaces or import camera animation, curves, or point instancers. Other USD
light types are not currently imported.
