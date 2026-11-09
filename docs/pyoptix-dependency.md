# Maintaining the PyOptiX dependency

`pyproject.toml` pins `pyoptix==9.1.1`, which wraps OptiX SDK 9.1.0 and
publishes manylinux_2_28 and Windows wheels built with pybind11 3.x. PyOptiX owns the standard native types,
contexts, geometry, pipelines, launches, and denoisers. The warp-optix
extension supplies supplemental bindings.

CMake downloads the OptiX headers as an archive of the immutable
`NVIDIA/optix-dev` commit behind the `v9.1.0` tag and verifies its SHA512. It then
checks the actual `OPTIX_VERSION` in `optix.h` and a digest of every header file
before compiling, including when a local copy is supplied through
`FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS`, which must therefore be unmodified. A
pin change to a new SDK must update the SDK version, commit, archive hash, and
header digest in `CMakeLists.txt` together; CMake rejects a pin whose
major.minor differs from the pinned headers. A compiler assertion
also protects the OptiX 9.1.0 wrapper bridge against unexpected include paths or a
pin change without updating the bridge. At import, the versions compiled into
PyOptiX (`optix.version()`) and our extension (`_ext.optix_version()`) must both
be 9.1.0. Mismatches raise `ImportError` before upstream types are modified.
An isolated build does not inspect the user's installed PyOptiX binary; the
import check catches dependency binaries built with different headers.

`src/extension.cpp` registers only additional cooperative-vector types
and operations. `_backend.py` composes these with the dependency and repairs
two upstream 9.1.1 bugs: ignored module compile metadata and a missing module
in continuation-only callable descriptors. Temporal denoiser properties are
added to the existing upstream types, so upstream functions accept them.
Importing this package leaves `sys.modules["optix"]` pointing to real PyOptiX.

The dependency does not expose native handles or a C interface.
`src/pyoptix_compat.h` contains four upstream wrapper definitions
for cross-module pybind11 casts. These are private ABI, not a general plugin
interface. The package checks the dependency version and the extension checks
registered type identities and sizes before accessing objects. Different
pybind11 major versions or incompatible C++ ABIs fail with an import error.
Current source builds use pybind11 3.x. A public borrowed-handle/capsule API
upstream would allow the private wrapper bridge to be removed.

When updating PyOptiX:

1. Compare those four wrapper definitions with the released dependency and
   align the pinned version, OptiX headers, and supported build ABI.
2. Remove supplemental operations or repairs that upstream now provides.
3. Run `python -m pytest warp_optix/tests`, build the wheel, and test
   clean wheel and editable installs.
4. Verify both import orders, upstream type identity, module metadata,
   continuation callables, temporal denoising, cooperative-vector conversion,
   and an OptiX-denoised frame on a supported GPU.
5. Inspect wheels: `warp_optix/_ext` is expected; `warp_optix/_bindings`,
   `optix/_optix`, and a top-level replacement `optix` package are not.

Runnable samples live in `examples_warp/`; dependency and supplemental-binding regressions live
in `warp_optix/tests`. Licenses for extracted binding code are retained.
OptiX SDK 9.1.0 headers are fetched for compilation and installed unmodified
under `warp_optix/include`, together with the upstream SDK license, in every
wheel. Runtime lookup uses the package directory or, for editable
installs, the native extension's directory. No separate SDK install is required.
Redistribution of these unmodified headers in the public warp-optix pip package
has been approved. The SDK terms remain separate from the Apache-2.0 and
BSD-3-Clause source licenses.
