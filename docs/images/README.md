# Images

All images are rendered with the warp-optix path tracer using sample accumulation
(no denoiser); isolated fireflies are removed with a 5×5 median filter.

## Banner

`banner.webp` (2400×960, transparent gaps):

- **Left:** the procedural chess set from `examples/scenes/chess.py` with glass and
  gold pieces, 8,192 spp.
- **Top right:** [Glass of Water](https://benedikt-bitterli.me/resources/) by aXel
  ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)), loaded from its
  Mitsuba scene. The area light is replaced by a sphere light; 16,384 spp.
- **Bottom right:** [The Breakfast Room](https://benedikt-bitterli.me/resources/) by
  Wig42 ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)), loaded from its
  Mitsuba scene and relit with the procedural sky and sun instead of its HDR
  environment; 8 bounces, 8,192 spp.

Both scenes are from Benedikt Bitterli's
[rendering resources](https://benedikt-bitterli.me/resources/). The scene files are
not part of this repository.

## Examples

The other images show the examples in [`examples/`](../../examples/README.md):
`chess`, `hairball` and `arrowball` are converged renders of
`example_pathtracing.py` scenes (4,096 spp); `raytracing`, `mixed_geometry`,
`curves` and `motion_blur` are the image outputs of the corresponding examples.
