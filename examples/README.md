# Examples

Runnable programs for `warp_optix`. Run them from the repository root.

Install the example dependencies:

```bash
uv sync --extra pathtracing --extra ui --extra usd
# or: pip install -e ".[pathtracing,ui,usd]"
```

| Example | What it shows | Command |
| --- | --- | --- |
| `example_minimal.py` | Smallest pipeline: Warp kernels as raygen and miss programs | `uv run python examples/example_minimal.py` |
| `example_curves.py` | Built-in round-linear curves with varying width | `uv run python examples/example_curves.py` |
| `example_mixed_geometry.py` | Triangles, curves, and custom primitives under GPU-written instances | `uv run python examples/example_mixed_geometry.py` |
| `example_motion_blur.py` | Vertex motion blur with a two-key motion GAS | `uv run python examples/example_motion_blur.py` |
| `example_raytracing.py` | Cornell box ray tracer with an optional live viewer | `uv run python examples/example_raytracing.py --live` |
| `example_pathtracing.py` | Path-traced procedural chess set | `uv run python examples/example_pathtracing.py --scene chess` |
| | Hair ball made of cubic Bézier curves | `uv run python examples/example_pathtracing.py --scene hairball` |
| | Arrows built with `ArrowBatch` | `uv run python examples/example_pathtracing.py --scene arrowball` |
| | 100k changing OpenGL lines over a path-traced frame | `uv run python examples/example_pathtracing.py --scene contact_lines` |
| | Your own glTF or USD file | `uv run python examples/example_pathtracing.py --gltf scene.glb` or `--usd scene.usd` |
| `neural_texture/` | Neural texture compression and OptiX inference ([README](neural_texture/README.md)) | `uv run --extra training python examples/neural_texture/compress.py --output material.wnt` |

The curve, mixed-geometry, motion-blur, and ray tracing examples write a BMP
image (`--output`). The path tracing viewer takes `--width`, `--height`,
`--denoiser`, `--exposure`, `--max-frames N` (exit after N frames), and
`--screenshot out.png`; run it with `--help` for the per-scene options.
