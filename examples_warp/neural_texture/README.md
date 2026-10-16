# Neural textures

`neural_texture` is a small, self-contained inference-on-sample
pipeline inspired by NVIDIA's neural material texture research. It stores two
4-bit latent grids and a fixed per-material `32 -> 32 -> 32 -> 16` decoder.
Portable `.wnt` files contain memory-mappable packed latents, row-major FP16
biases and FP8-representable weights, checksums, channel mappings, and JSON
metadata. On upload, weights are converted once into OptiX's device-specific
FP8 E4M3 inference-optimal layout.

Only offline compression depends on `warp-nn`:

```bash
uv run --group training python examples_warp/neural_texture/compress.py \
  --output material.wnt --reconstruction material-decoded.npy
```

`compress.py` procedurally generates a material texture (or loads an HWC NumPy
array with `--input`), trains packed latents and the decoder, writes a
memory-mappable `.wnt` asset, and reports its stored size and FP8-projected
reconstruction PSNR.

The package is not installed; scripts and tests import it from this directory.
Run the tests from the repository root (training tests skip without warp-nn):

```bash
uv run pytest examples_warp/neural_texture
uv run --group training pytest examples_warp/neural_texture
```

The Python API is equally direct:

```python
import numpy as np
import warp_optix as wo
from neural_texture import compress_texture_set, load_asset, save_asset, upload_asset

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
context = wo.create_context("cuda:0")
runtime = upload_asset(asset, context, device="cuda:0")
```

All maps must have the same resolution and may contain one to four channels;
the set may contain at most 16 channels. Integer images are normalized
automatically, while floating-point images must already be in `[0, 1]`.
`color_spaces` records metadata and does not transform input values.
`compress_texture()` remains the short single-image helper.

Inference needs one device view and one Warp call. The view is an ordinary
launch parameter of any OptiX program:

```python
from neural_texture import NeuralTextureView, neural_texture_sample


@wp.kernel(enable_backward=False, module="unique")
def raygen(material: NeuralTextureView, image: wp.array2d(dtype=wp.vec3)):
    y, x = wp.tid()
    uv = wp.vec2(float(x) / float(image.shape[1]), float(y) / float(image.shape[0]))
    values = neural_texture_sample(material, uv)
    image[y, x] = wp.vec3(float(values[0]), float(values[1]), float(values[2]))


pipeline = wo.create_pipeline(context, raygen)
wo.launch(pipeline, dim=image.shape, inputs=[runtime.device_view(), image])
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

## Output-width benchmark

Run from the repository root with an existing 16-output `.wnt` asset:

```bash
uv run python examples_warp/neural_texture/benchmark_outputs.py material.wnt
```

This is an inference-only experiment; it needs no warp-nn. It truncates the
trained final matrix to 4 or 8 rows, without retraining or changing the public
asset format. Latents, hidden layers, meaningful output values, launch dimensions
and output traffic remain identical within each comparison. The separate
production16 case calls the actual public texel sampler.

Each variant is compiled in an independent OptiX module. An initial shared-module
experiment produced misleading ~2x slowdowns for narrower variants; separately
compiled variants eliminated that difference. Compiler context matters.

Timing excludes compilation, upload and correctness readback. It uses CUDA events,
100 warmup launches per case, then 15 randomized rounds of 200 launches per case.
Both coherent and pseudorandom texel access are tested. Each case writes exactly
three or seven float32 channels per pixel; seven channels from an RGB-trained
asset are only a synthetic arithmetic control, not a material-quality test.
The script reports all round timings, converted buffer sizes and output errors.

### Measured decision (2026-09-18)

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 13.1 reported by
Warp, Warp 1.18.0.dev20260824. Inputs were existing 2048x2048 photograph and
albedo/normal/roughness assets. Times below are median milliseconds for all
4,194,304 evaluations, including stores and (in random cases) coordinate RNG:

| Input / consumed channels | Access | 4 outputs | 8 outputs | 16 outputs | Production 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Photograph / RGB | Coherent | 0.164 | 0.166 | 0.165 | 0.165 |
| Photograph / RGB | Random | 0.231 | 0.232 | 0.234 | 0.233 |
| Material / 7 channels | Coherent | — | 0.199 | 0.196 | 0.196 |
| Material / 7 channels | Random | — | 0.278 | 0.279 | 0.278 |

All compared outputs were bit-identical in these runs. Repeat/control runs also
showed roughly tied performance; small differences do not establish a useful
speedup. This is a ray-generation microbenchmark, not a full renderer, and does
not establish results for other GPUs.

Converted final-matrix storage was 512 bytes for both 4 and 8 outputs, versus
1024 bytes for 16. Including biases, total GPU parameter savings are only
536 bytes (4 outputs) or 528 bytes (8 outputs); latent storage is unchanged.
Portable FP16 parameter payload savings would be 792 or 528 bytes, respectively.

Decision: keep one public 16-output decoder. The measured benefit does not
justify extra asset architectures, training paths and evaluation entry points.
The benchmark is retained so that this can be revisited on other hardware.
