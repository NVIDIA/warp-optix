# Warp-OptiX: Ray Tracing with NVIDIA Warp

[Documentation](https://github.com/NVIDIA/warp-optix/blob/main/warp_optix/README.md) |
[Examples](https://github.com/NVIDIA/warp-optix/blob/main/examples_warp/README.md)

Warp-OptiX brings NVIDIA OptiX ray tracing to Warp. Write ray-generation, hit,
miss, and callable programs as Warp kernels, or use the included path tracer.
The Python API provides managed contexts, acceleration structures, automatic
shader binding tables, and launches that mirror `wp.launch`.

## Installation

```bash
pip install warp-optix
```

GPU execution requires a compatible NVIDIA GPU and driver. The 0.2.0 PyPI wheel
supports Python 3.12 on Windows x86-64. Source builds require Python 3.10+,
a CUDA Toolkit, and a C++17 compiler. See the
[installation guide](https://github.com/NVIDIA/warp-optix/blob/main/README.md#installation)
for source and development setup.

PyOptiX 9.1.0 is installed automatically. The package includes OptiX SDK 9.1.0
headers for runtime kernel compilation, so a separate OptiX SDK installation
is not required.

Optional extras add path tracing, UI, USD loading, and neural-network training:

```bash
pip install "warp-optix[pathtracing,ui,usd]"
pip install "warp-optix[training]"
```

## Getting started

```python
import warp as wp
import warp_optix as wo
from warp_optix.pathtracing import PathTracer
```

Follow the [API guide](https://github.com/NVIDIA/warp-optix/blob/main/warp_optix/README.md)
to build a custom ray tracer, or explore the
[runnable examples](https://github.com/NVIDIA/warp-optix/blob/main/examples_warp/README.md).
Advanced applications can access PyOptiX and supplemental bindings through
`warp_optix.native`.

**CUDA graph capture is currently unsupported for OptiX operations.** Submit
OptiX work between captured Warp graphs on the same stream; CUDA stream
ordering is supported.

## Support and contributions

Report bugs and feature requests through
[GitHub Issues](https://github.com/NVIDIA/warp-optix/issues).
The project is currently not accepting outside contributions; see the
[contribution policy](https://github.com/NVIDIA/warp-optix/blob/main/CONTRIBUTING.md)
for updates. Report security vulnerabilities privately using the
[security policy](https://github.com/NVIDIA/warp-optix/blob/main/SECURITY.md).

## License

Warp integration uses [Apache-2.0](https://github.com/NVIDIA/warp-optix/blob/main/LICENSE.md);
supplemental bindings use [BSD-3-Clause](https://github.com/NVIDIA/warp-optix/blob/main/LICENSE.txt).
Bundled OptiX headers retain their
[NVIDIA SDK terms](https://github.com/NVIDIA/warp-optix/blob/main/OptiX_LICENSE.txt).
See [third-party notices](https://github.com/NVIDIA/warp-optix/blob/main/THIRD_PARTY_NOTICES.txt)
for component licenses and source provenance. License texts accompany the distribution.
