# Migrating to warp-optix 0.2

This version makes a clean break from the backend-taking helper API. Install
from the repository root. Use a fresh environment when moving from the old
split pyoptix/warp_optix installation.

OptiX programs are now ordinary Warp kernels. The raygen kernel's arguments
are the launch parameters, every other program declares a subset of them by
name, and `wo.launch()` mirrors `wp.launch()`. PTX compilation, entry names,
launch-parameter buffers, and SBT packing are no longer user-facing.

| Earlier API | 0.2 API |
| --- | --- |
| `require_optix()` | No setup needed; advanced access is `from warp_optix import native` |
| `create_context(optix, cuda_context)` | `ctx = wo.create_context("cuda:0")`; `wo.destroy_context(ctx)` |
| `@optix_kernel(OptixKernelType.RAYGEN)` | `@wp.kernel(module="unique")`; the kind is given by the pipeline slot |
| `compile_warp_module_to_ptx(...)` | Done by `wo.create_pipeline()` |
| `create_triangle_gas(...)`, `create_curve_gas(...)`, `create_custom_primitive_gas(...)` | `wo.build_accel(ctx, [wo.triangles(...)])`, `wo.curves(...)`, `wo.aabbs(...)` over Warp device arrays |
| `optix.Instance(...)` and `create_instance_acceleration_structure(...)` | `wo.make_instance()` in a Warp kernel, then `wo.build_accel(ctx, [wo.instances(records)])` |
| `(handle, buffers)` | `accel.handle`; inputs stay owned by the caller |
| `refit_acceleration_structure(...)` | `wo.update_accel(ctx, accel, stream=...)` |
| `create_pipeline_and_sbt(...)`, `HitKernel(...)` | `wo.create_pipeline(ctx, raygen, miss=..., hit_groups=[wo.HitGroup(...)])` |
| `launch(optix, pipeline, sbt, width, height, params)` | `wo.launch(pipeline, dim=(height, width), inputs=[...], stream=...)` |
| SBT manager handles and offsets | Hit record of geometry `g` and ray type `r` is `hit_groups[g * ray_types + r]`; callables are numbered in list order |
| `upload_asset(asset, ctx, optix, device=...)` | Neural textures moved to `examples_warp/neural_texture` |

Configuration enums are exported by warp_optix: `BuildFlags`, `GeometryFlags`,
`InstanceFlags`, `PrimitiveType`, and `TraversableGraphFlags`. Members omit
redundant SDK prefixes, for example `BuildFlags.ALLOW_UPDATE` and
`PrimitiveType.ROUND_LINEAR`. `wo.constants` holds the same values as plain
integers for use inside kernels.

`wo.closest_hit()` and `wo.any_hit()` query the scene from a raygen kernel
without hit programs, which is all a renderer that only casts rays needs.

Acceleration structures reference the caller's device arrays, which must
outlive them. Builds, refits, and launches are asynchronous on Warp's current
stream or an explicitly supplied `wp.Stream`; only compaction synchronizes.
`destroy_context()` synchronizes the device and destroys every pipeline and
module created from the context; keep acceleration structures and pipelines
referenced until it returns.

`OptixDenoiser` takes the context returned by `wo.create_context()`.
`DlssRRContext.init()` now requires the application's NGX project ID as its
first argument, and `native.dlss_rr_available(project_id)` takes it too.
`native` is version-coupled advanced access; native allocations require native
lifetime management.
Standard native objects now come from the `pyoptix==9.1.0` dependency. A separate
`warp_optix._ext` extension supplies DLSS and missing binding operations, without
registering duplicate standard OptiX types. Importing `optix` and `warp_optix`
together is supported. The previous `warp_optix._bindings` backend is removed.
PyOptiX's wrapper types are augmented when `warp_optix` is imported; use
`warp_optix.native` for the complete API, including supplemental enum values.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction; on Blackwell with driver 595.99,
OptiX launches and acceleration-structure builds are not recorded into the
captured graph at all. `launch()`, `build_accel()`, `update_accel()`, and
`create_pipeline()` reject an active capture before submitting native work.
Capture ordinary Warp work in graphs and submit OptiX work between those graphs
on the same stream.

The renderer's `enable_cuda_graphs` argument is removed; it always submits
OptiX work directly. DLSS and ordinary stream ordering remain supported.
