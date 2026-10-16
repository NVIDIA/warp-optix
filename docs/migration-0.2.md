# Migrating to warp-optix 0.2

This version makes a clean break from the backend-taking helper API. Install
from the repository root. Use a fresh environment when moving from the old
split pyoptix/warp_optix installation.

| Earlier API | Owned API |
| --- | --- |
| `require_optix()` | No setup needed; advanced access is `from warp_optix import native` |
| `create_context(optix, cuda_context)` | `Context(device="cuda:0")` |
| GAS/IAS helpers with backend, context, and device arguments | Same method names on `context`, without those arguments |
| `(handle, buffers)` | `geometry.handle`, `geometry.buffers`, and build statistics |
| `optix.Instance(...)` | `Instance(geometry, transform=..., instance_id=..., sbt_offset=...)` |
| `create_pipeline_and_sbt(...)` | `context.create_pipeline(...)` returning `Pipeline` |
| `launch(optix, pipeline, sbt, width, height, params)` | `pipeline.launch(params, width=width, height=height)` |
| `refit_acceleration_structure(...)` | `geometry.refit(stream=warp_stream)` |
| SBT manager handles | `pipeline.hit_group_offsets`, `direct_callable_indices`, `continuation_callable_indices` |
| `upload_asset(asset, ctx, optix, device=...)` | `upload_asset(asset, context, stream=warp_stream)` |

Configuration enums are exported by warp_optix: `BuildFlags`, `GeometryFlags`,
`InstanceFlags`, `PrimitiveType`, `PrimitiveTypeFlags`, `TraversableGraphFlags`,
and `ExceptionFlags`. Members omit redundant SDK prefixes, for example
`BuildFlags.ALLOW_UPDATE` and `PrimitiveType.ROUND_LINEAR`.

`HitKernel` accepts decorated Warp kernels or entry names, and a public
primitive type. Native-module injection and native SBT management belong to
the advanced binding interface. Private runtime helpers are not public API.

Use `with Context(...) as context:` for deterministic cleanup. The context
retains geometry, pipelines, and owned OptiX denoisers until closure.
`close()` is idempotent and synchronizes the device. Closed resources and
cross-context instances are rejected; streams and arrays must match the device.

Builders accept NumPy inputs or Warp arrays on the context's device. They
currently stage through host memory into retained geometry buffers. Update
`geometry.buffers` before refitting: the mapping is read-only, but the arrays
inside it are mutable. Pipeline launches retain their parameter buffers;
callers retain arrays referenced by pointer inside those parameters until work
completes.

Path-tracing and viewer APIs are preserved. `OptixDenoiser` now accepts an
owned Context. `native` is version-coupled advanced access; borrowed handles
must not be destroyed, and native allocations require native lifetime management.
Standard native objects now come from the `pyoptix==9.1.0` dependency. A separate
`warp_optix._ext` extension supplies missing binding operations, without
registering duplicate standard OptiX types. Importing `optix` and `warp_optix`
together is supported. The previous `warp_optix._bindings` backend is removed.
PyOptiX's wrapper types are augmented when `warp_optix` is imported; use
`warp_optix.native` for the complete API, including supplemental enum values.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction. `Pipeline.launch` rejects an
active capture before submitting native work. Capture ordinary Warp work in
graphs and submit OptiX launches between those graphs on the same stream.

The renderer retains its `enable_cuda_graphs` argument for caller compatibility,
but requests for OptiX capture use direct asynchronous launches with a logged
explanation. CUDA stream ordering remains supported.
