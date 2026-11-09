# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example Path Tracing

Interactive `warp_optix.pathtracing` viewer for a procedural scene, a glTF
file, or a USD stage. The hairball and arrowball scenes use native OptiX
curves; contact_lines draws changing OpenGL lines over the path-traced frame.
glass_of_water and breakfast_room are downloaded on first use from Benedikt
Bitterli's rendering resources (CC0 and CC BY 3.0).

Runs:
- `python examples/example_pathtracing.py --scene chess`
- `python examples/example_pathtracing.py --scene glass_chess`
- `python examples/example_pathtracing.py --scene glass_of_water`
- `python examples/example_pathtracing.py --scene breakfast_room`
- `python examples/example_pathtracing.py --scene hairball`
- `python examples/example_pathtracing.py --scene arrowball`
- `python examples/example_pathtracing.py --scene contact_lines`
- `python examples/example_pathtracing.py --gltf scene.glb`
- `python examples/example_pathtracing.py --usd scene.usd`
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from scenes import arrowball, chess, hairball, style
from warp_optix.gl import GLInteropViewer
from warp_optix.pathtracing import DEFAULT_VIEWER_HEIGHT, DEFAULT_VIEWER_WIDTH, PathTracer


@dataclass
class View:
    position: tuple
    yaw: float
    pitch: float
    fov: float
    speed: float = 1.0
    exposure: float = 0.32


BALL_VIEW = View((0.0, 0.18, 3.55), 180.0, -3.0, 42.0)


@wp.kernel
def _pack_display_rgba8(src: wp.array2d(dtype=wp.vec4), dst: wp.array(dtype=wp.uint32)):
    y, x = wp.tid()
    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * src.shape[1] + x] = (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


class FreeCameraController:
    """Left-drag to look, WASD/arrows to move, Q/E down/up, Shift faster, wheel to zoom, 1-9 debug views."""

    def __init__(self, viewer, api: PathTracer, view: View):
        self.pyglet = viewer.pyglet
        self.api = api
        self.position = np.array(view.position, dtype=np.float32)
        self.yaw = float(view.yaw)
        self.pitch = float(view.pitch)
        self.fov = float(view.fov)
        self.speed = max(0.0, float(view.speed))
        self.keys_down: set[int] = set()
        self._apply()
        viewer.window.push_handlers(self)

    def _apply(self):
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def update(self, dt: float):
        key = self.pyglet.window.key
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        forward = np.array(
            [math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], dtype=np.float32
        )
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1.0e-6:
            right /= right_norm

        move = np.zeros(3, dtype=np.float32)
        for symbols, direction in (
            ((key.W, key.UP), forward),
            ((key.S, key.DOWN), -forward),
            ((key.A, key.LEFT), -right),
            ((key.D, key.RIGHT), right),
            ((key.Q,), -up),
            ((key.E,), up),
        ):
            if any(symbol in self.keys_down for symbol in symbols):
                move += direction
        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0e-6:
            speed = self.speed * (4.0 if {key.LSHIFT, key.RSHIFT} & self.keys_down else 1.0)
            self.position += move / move_norm * speed * max(0.0, min(dt, 0.1))
            self._apply()

    def on_key_press(self, symbol, _modifiers):
        # 1 final, 2 radiance, 3 depth, 4 motion, 5 normal, 6 roughness, 7 diffuse, 8 specular, 9 spec hit distance.
        key = self.pyglet.window.key
        debug_keys = (key._1, key._2, key._3, key._4, key._5, key._6, key._7, key._8, key._9)
        if symbol in debug_keys:
            self.api.set_debug_buffer_mode(debug_keys.index(symbol))
        self.keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self.keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if buttons & self.pyglet.window.mouse.LEFT:
            self.yaw -= float(dx) * 0.1
            self.pitch = max(-89.0, min(89.0, self.pitch + float(dy) * 0.1))
            self._apply()

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.fov = max(15.0, min(90.0, self.fov - float(scroll_y) * 2.0))
        self._apply()


def _add_ball_and_ground(api: PathTracer, core_material: int, ball_radius: float = 0.8):
    api.add_sphere((0.0, 0.0, 0.0), ball_radius * 1.005, 64, core_material)
    style.add_checker_ground(api, height=-1.38)


def setup_chess(api: PathTracer, args) -> View:
    chess.build_chess_scene(api)
    api.build_scene()
    position, yaw, pitch = style.camera_angles(
        chess.CAMERA_POSITION, np.subtract(chess.CAMERA_TARGET, chess.CAMERA_POSITION)
    )
    return View(position, yaw, pitch, chess.CAMERA_FOV, speed=2.0, exposure=0.68)


def setup_glass_chess(api: PathTracer, args) -> View:
    chess.build_chess_scene(api, pieces="glass")
    api.build_scene()
    position = (-8.4, 3.9, -7.6)
    position, yaw, pitch = style.camera_angles(position, np.subtract((-0.2, -0.3, 0.2), position))
    return View(position, yaw, pitch, 36.0, speed=2.0, exposure=0.68)


def _mitsuba_view(scene, speed: float, exposure: float) -> View:
    position, direction, fov = scene.camera()
    position, yaw, pitch = style.camera_angles(position, direction)
    return View(position, yaw, pitch, fov, speed=speed, exposure=exposure)


def setup_glass_of_water(api: PathTracer, args) -> View:
    from scenes import mitsuba

    # The scene's large area light is replaced by a sphere light, which the path tracer samples directly.
    scene = mitsuba.MitsubaScene(mitsuba.fetch("glass-of-water"), emitters=False)
    scene.build(api)
    api.scene.add_light_sphere((0.0, 10.5, 26.2), 3.0, (0.5, 0.85, 1.0), 90.0)
    api.build_scene()
    api.set_use_procedural_sky(False)
    api.set_environment_color((0.0, 0.0, 0.0))
    return _mitsuba_view(scene, speed=3.0, exposure=0.85)


def setup_breakfast_room(api: PathTracer, args) -> View:
    from scenes import mitsuba

    scene = mitsuba.MitsubaScene(mitsuba.fetch("dining-room"))
    scene.build(api)
    api.build_scene()
    # A low sun shines through the blinds instead of the scene's HDR environment.
    api.set_use_procedural_sky(True)
    api.set_sky_parameters((0.8, 0.5, 0.3), multiplier=1.5, haze=0.05, sun_disk_intensity=2.0, sun_glow_intensity=0.5)
    return _mitsuba_view(scene, speed=1.0, exposure=0.68)


def setup_hairball(api: PathTracer, args) -> View:
    segments = 5
    points, radii, segment_indices = hairball.generate_hair_ball(args.hair_count, segments)
    core_material = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    palette = style.create_rainbow_materials(api)
    _add_ball_and_ground(api, core_material)
    curve = api.create_curve(
        points,
        radii,
        segment_indices,
        material_id=int(palette[0]),
        material_ids=palette[hairball.rainbow_segment_slots(points, segments)],
        basis="cubic_bezier",
    )
    api.create_instance(curve)
    api.build_scene()
    style.configure_demo_sky(api)
    return BALL_VIEW


def setup_arrowball(api: PathTracer, args) -> View:
    starts, ends, slots = arrowball.generate_arrow_ball(args.arrow_count)
    core_material = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    palette = style.create_rainbow_materials(api)
    _add_ball_and_ground(api, core_material)
    arrows = api.create_arrow_batch(
        capacity=args.arrow_count,
        small_radius=0.011,
        large_radius=0.034,
        tip_length_ratio=0.28,
        material_id=int(palette[0]),
        material_ids=palette[slots],
    )
    api.update_arrow_batch(arrows, starts, ends)
    api.build_scene()
    style.configure_demo_sky(api)
    return BALL_VIEW


def setup_contact_lines(api: PathTracer, args) -> View:
    core_material = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    _add_ball_and_ground(api, core_material)
    api.build_scene()
    style.configure_demo_sky(api)
    return BALL_VIEW


def setup_gltf(api: PathTracer, args) -> View:
    path = args.gltf.expanduser().resolve()
    if not api.load_scene_from_gltf(str(path), build_scene=True):
        raise RuntimeError(f"Failed to load glTF scene: {path}")
    return View((-0.803, 0.340, 0.327), 115.2, -21.8, 45.0, speed=2.0, exposure=0.68)


def setup_usd(api: PathTracer, args) -> View:
    from scenes import usd

    path = args.usd.expanduser().resolve()
    if not api.load_scene_from_usd(str(path), max_texture_size=args.max_texture_size or None):
        raise RuntimeError(f"USD stage contained no supported render meshes: {path}")
    position, yaw, pitch, fov, radius, camera_path = usd.camera_for_scene(api, requested_path=args.usd_camera)
    print(f"[optix] camera: {camera_path or 'framed from geometry bounds'}")
    return View(position, yaw, pitch, fov, speed=usd.default_camera_speed(radius), exposure=0.68)


SCENES = {
    "chess": setup_chess,
    "glass_chess": setup_glass_chess,
    "glass_of_water": setup_glass_of_water,
    "breakfast_room": setup_breakfast_room,
    "hairball": setup_hairball,
    "arrowball": setup_arrowball,
    "contact_lines": setup_contact_lines,
}

# Glass needs deeper paths, and the downloaded scenes contain single-sided walls and glass.
SCENE_OPTIONS = {
    "glass_chess": {"max_bounces": 12},
    "glass_of_water": {"max_bounces": 16, "backface_culling": False},
    "breakfast_room": {"max_bounces": 8, "backface_culling": False},
}


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scene", choices=SCENES, default="chess", help="Procedural scene to render.")
    source.add_argument("--gltf", type=Path, help="Render a .gltf/.glb file instead of a procedural scene.")
    source.add_argument(
        "--usd", type=Path, help="Render a USD stage instead of a procedural scene (needs the usd extra)."
    )
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run until closed).")
    parser.add_argument("--screenshot", type=Path, help="Save the last frame as a PNG on exit.")
    parser.add_argument("--denoiser", choices=("auto", "optix", "none"), default="auto")
    parser.add_argument("--exposure", type=float, help="Display exposure (default depends on the scene).")
    parser.add_argument("--hair-count", type=int, default=4000, help="Number of strands (hairball).")
    parser.add_argument("--arrow-count", type=int, default=1800, help="Number of arrows (arrowball).")
    parser.add_argument("--contact-count", type=int, default=100_000, help="Maximum number of lines (contact_lines).")
    parser.add_argument("--usd-camera", help="USD camera prim path; defaults to the stage's bound camera.")
    parser.add_argument(
        "--max-texture-size", type=int, default=512, help="Maximum USD texture size per tile (0 = full size)."
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")

    wp.init()
    options = SCENE_OPTIONS.get(args.scene, {}) if args.usd is None and args.gltf is None else {}
    api = PathTracer(width=args.width, height=args.height, denoiser=args.denoiser, **options)
    api.initialize()

    if args.usd is not None:
        name, view = args.usd.name, setup_usd(api, args)
    elif args.gltf is not None:
        name, view = args.gltf.name, setup_gltf(api, args)
    else:
        name, view = args.scene, SCENES[args.scene](api, args)
    api.tonemap_exposure = view.exposure if args.exposure is None else args.exposure
    api.tonemap_contrast = 1.08
    api.tonemap_saturation = 1.1

    lines = None
    if args.usd is None and args.gltf is None and args.scene == "contact_lines":
        from scenes.contact_lines import ContactLines

        lines = ContactLines(api, args.contact_count)

    last_elapsed = 0.0

    def on_resize(width: int, height: int):
        nonlocal last_elapsed
        api.resize(int(width), int(height))
        if lines is not None:
            lines.resize()
        last_elapsed = 0.0

    viewer = GLInteropViewer(
        width=args.width,
        height=args.height,
        device="cuda",
        title=f"Path Tracing - {name}",
        on_resize=on_resize,
    )
    if lines is not None:
        lines.attach(viewer)
    controller = FreeCameraController(viewer, api, view)

    def render(mapped_image: wp.array, frame: int, elapsed: float):
        nonlocal last_elapsed
        controller.update(elapsed - last_elapsed)
        last_elapsed = elapsed
        if lines is not None:
            lines.update(frame, elapsed)
        api.render_frame()
        image = api.viewer.tonemapped_output
        wp.launch(_pack_display_rgba8, dim=image.shape, inputs=[image, mapped_image], device="cuda")

    print(f"[optix] scene: {name}, denoiser: {api.active_denoiser}")
    print(f"[optix] controls: {FreeCameraController.__doc__}")
    try:
        viewer.run(render, max_frames=args.max_frames)
    finally:
        if lines is not None:
            lines.close()

    if args.screenshot is not None:
        from PIL import Image

        screenshot = args.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        # The render buffer uses OpenGL's bottom-left origin.
        frame = np.flipud(np.clip(api.get_frame(), 0.0, 1.0))
        Image.fromarray((frame[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(screenshot)
        print(f"[optix] saved screenshot: {screenshot}")


if __name__ == "__main__":
    main()
