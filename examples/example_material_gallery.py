# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browse the procedural material library with the OptiX path tracer."""

import argparse
import math
from pathlib import Path

import numpy as np
import warp as wp
from warp_optix.pathtracing import PathTracer

from warp_optix import procedural_textures as textures

COLUMNS = 4
ROWS = 2
PAGE_SIZE = COLUMNS * ROWS
PITCH_X = 2.65
PITCH_Y = 2.05
CAMERA_DISTANCE = 8.0
CONTROLS = "WASD: move   Q/E: down/up   Drag: look   Home: reset   Arrows: browse   V/R: variation/reference"


def _add_tile(api, x, y, material):
    # A slightly tilted flat swatch complements the curved UV sphere.
    corners = ((0.18, -0.48, 0.06), (1.14, -0.48, -0.08), (1.14, 0.48, -0.16), (0.18, 0.48, -0.02))
    vertices = np.array(
        [[x + dx, y + dy, dz] for dx, dy, dz in corners],
        dtype=np.float32,
    )
    indices = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
    uvs = np.array(((0, 1), (1, 1), (1, 0), (0, 0)), dtype=np.float32)
    mesh = api.create_mesh(vertices, indices, uvs=uvs, material_id=material)
    api.create_instance(mesh)


class Gallery:
    def __init__(self, api, texture_size, single=False, ids=None):
        self.api = api
        self.texture_size = texture_size
        self.ids = textures.reference_ids() if ids is None else tuple(ids)
        self.columns, self.rows = (1, 1) if single else (COLUMNS, ROWS)
        self.page_size = self.columns * self.rows
        self.page_count = math.ceil(len(self.ids) / self.page_size)
        self.page = 0
        self.seed = None
        self.visible_ids = ()
        self.half_height = 1.0
        api.initialize()
        api.viewer.accumulate_samples = True
        api.tonemap_exposure = 0.75
        api.set_use_procedural_sky(False)
        api.set_environment_color((0.28, 0.28, 0.28))
        self.resize(api.width, api.height)

    def slot_center(self, index):
        row, col = divmod(index, self.columns)
        return (col - (self.columns - 1) / 2) * PITCH_X, ((self.rows - 1) / 2 - row) * PITCH_Y

    def resize(self, width, height, *, reframe=True):
        if (width, height) != (self.api.width, self.api.height):
            self.api.resize(width, height)
        self.half_height = max(1.36 if self.page_size == 1 else 2.43, (self.columns * 1.32 + 0.22) * height / width)
        if reframe:
            self.api.set_camera_look_at((0, 0, CAMERA_DISTANCE), (0, 0, 0), fov=self.home_fov)

    @property
    def home_fov(self):
        return math.degrees(2 * math.atan(self.half_height / CAMERA_DISTANCE))

    def load(self, page, seed):
        page %= self.page_count
        ids = self.ids[page * self.page_size : (page + 1) * self.page_size]
        images = []
        materials = []
        for asset in ids:
            maps, kwargs = textures.bake_gltf(
                asset, (self.texture_size, self.texture_size), device="cuda", first_texture_index=len(images), seed=seed
            )
            images.extend(maps)
            materials.append(kwargs)

        self.api.clear_scene()
        scene = self.api.scene
        scene.set_gltf_textures(images)
        for slot, kwargs in enumerate(materials):
            material = scene.materials.add_gltf_material(**kwargs)
            x, y = self.slot_center(slot)
            self.api.add_sphere((x - 0.60, y, 0), 0.61, 48, material)
            _add_tile(self.api, x, y, material)
        scene.add_light_sphere((-6, 7, 9), 2.4, (1.0, 0.96, 0.91), 17.0)
        scene.add_light_sphere((6, 3, 6), 1.8, (0.91, 0.95, 1.0), 10.0)
        self.api.build_scene()
        self.page, self.seed, self.visible_ids = page, seed, ids

    def annotations(self):
        width, height = self.api.width, self.api.height
        variation = "reference fit" if self.seed is None else f"seed {self.seed}"
        yield f"Material gallery    Page {self.page + 1}/{self.page_count}    {variation}", width / 2, height - 24
        camera = self.api.viewer.camera
        view_projection = camera.get_view_matrix() @ camera.get_projection_matrix()
        for slot, asset in enumerate(self.visible_ids):
            x, y = self.slot_center(slot)
            clip = np.array((x, y - 0.80, 0.0, 1.0)) @ view_projection
            if clip[3] > 0.0:
                ndc_x, ndc_y = clip[:2] / clip[3]
                if abs(ndc_x) < 0.96 and abs(ndc_y) < 0.89:
                    yield asset, width * (0.5 + 0.5 * ndc_x), height * (0.5 - 0.5 * ndc_y)
        yield CONTROLS, width / 2, 19

    def save(self, path, frames):
        from PIL import Image, ImageDraw

        for _ in range(frames):
            self.api.render_frame()
        image = Image.fromarray(np.ascontiguousarray(self.api.get_frame_uint8()[::-1, :, :3]))
        draw = ImageDraw.Draw(image)
        for caption, x, y in self.annotations():
            draw.text((x, image.height - y), caption, anchor="mm", fill="white", stroke_width=1, stroke_fill="black")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)


def _show(gallery):
    from example_pathtracing import FreeCameraController, View, _pack_display_rgba8
    from warp_optix.gl import GLInteropViewer

    api = gallery.api
    frames_left = 32
    pending_size = None
    previous_time = 0.0

    def refresh():
        nonlocal frames_left
        frames_left = 32

    def resize(width, height):
        nonlocal pending_size
        pending_size = width, height

    viewer = GLInteropViewer(
        api.width,
        api.height,
        device="cuda",
        title="Warp OptiX Material Gallery",
        on_resize=resize,
        show_fps=False,
        fps=30,
    )
    viewer.window.set_minimum_size(320, 180)
    labels = [
        viewer.pyglet.text.Label(font_size=11, anchor_x="center", anchor_y="center", color=(255, 255, 255, 255))
        for _ in range(gallery.page_size + 2)
    ]
    key = viewer.pyglet.window.key

    def home_view():
        return View((0, 0, CAMERA_DISTANCE), 180.0, 0.0, gallery.home_fov, speed=2.0)

    controller = FreeCameraController(viewer, api, home_view(), use_arrows=False, on_change=refresh)

    def on_key_press(symbol, _modifiers):
        nonlocal frames_left
        if symbol in (key.LEFT, key.RIGHT):
            gallery.load(gallery.page + (1 if symbol == key.RIGHT else -1), gallery.seed)
        elif symbol == key.V:
            gallery.load(gallery.page, 0 if gallery.seed is None else gallery.seed + 1)
        elif symbol == key.R and gallery.seed is not None:
            gallery.load(gallery.page, None)
        elif symbol == key.HOME:
            controller.set_view(home_view())
            api.reset_temporal_history()
        else:
            return
        frames_left = 32

    def overlay():
        for label, (caption, x, y) in zip(labels, gallery.annotations(), strict=False):
            label.text, label.x, label.y = caption, x, y
            label.draw()

    def render(mapped_image, _frame, elapsed):
        nonlocal frames_left, pending_size, previous_time
        if pending_size is not None:
            gallery.resize(*pending_size, reframe=False)
            pending_size = None
            frames_left = 32
        controller.update(elapsed - previous_time)
        previous_time = elapsed
        if frames_left:
            api.render_frame()
            frames_left -= 1
        output = api.viewer.tonemapped_output
        if output.shape != (viewer.height, viewer.width):
            raise RuntimeError("The path tracer output size does not match the display buffer.")
        wp.launch(_pack_display_rgba8, dim=output.shape, inputs=[output, mapped_image], device="cuda")

    viewer.window.push_handlers(on_key_press=on_key_press)
    viewer.set_draw_overlay(overlay)
    try:
        viewer.run(render)
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, default=1, help="Initial page, starting at 1.")
    parser.add_argument("--single", action="store_true", help="Inspect one material per page at a larger scale.")
    parser.add_argument("--filter", default="", help="Show only material IDs containing this text, e.g. WoodFloor.")
    parser.add_argument("--seed", type=int, help="Initial material variation seed.")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--texture-size", type=int, help="Texture bake size (2048 for --single, 1024 for the grid).")
    parser.add_argument("--output", type=Path, help="Render one page to a PNG without opening a window.")
    parser.add_argument("--frames", type=int, default=16, help="Frames to accumulate for --output.")
    args = parser.parse_args()
    if args.texture_size is None:
        args.texture_size = 2048 if args.single else 1024
    if min(args.width, args.height, args.texture_size, args.frames) < 1:
        parser.error("width, height, texture size, and frames must be positive")
    ids = [asset for asset in textures.reference_ids() if args.filter.casefold() in asset.casefold()]
    if not ids:
        parser.error(f"no materials match {args.filter!r}")
    page_count = math.ceil(len(ids) / (1 if args.single else PAGE_SIZE))
    if not 1 <= args.page <= page_count:
        parser.error(f"page must be between 1 and {page_count}")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be non-negative")

    api = PathTracer(
        width=args.width, height=args.height, denoiser="none", samples_per_frame=4, enable_texture_mipmaps=True
    )
    try:
        gallery = Gallery(api, args.texture_size, single=args.single, ids=ids)
        gallery.load(args.page - 1, args.seed)
        if args.output is None:
            _show(gallery)
        else:
            gallery.save(args.output, args.frames)
    finally:
        api.close()


if __name__ == "__main__":
    main()
