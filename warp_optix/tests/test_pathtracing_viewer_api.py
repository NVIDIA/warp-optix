# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from PIL import Image
from warp_optix.pathtracing import (
    PathTracerAPI,
    PathTracingRenderer,
    PathTracingViewerBackend,
)
from warp_optix.pathtracing import pathtracing_viewer as viewer_module
from warp_optix.pathtracing.scene import Scene


class _FakeScene:
    def __init__(self):
        self._meshes = []
        self._instances = []
        self._gltf_textures = []
        self.materials = SimpleNamespace(add_gltf_material=self._add_gltf_material)
        self.textured_materials = []
        self.uploaded_material_ids = None

    @property
    def texture_count(self):
        return len(self._gltf_textures)

    def set_gltf_textures(self, textures, srgb_texture_indices=None, append=False):
        del srgb_texture_indices
        if not append:
            self._gltf_textures.clear()
        self._gltf_textures.extend(textures)

    def _add_gltf_material(self, **kwargs):
        self.textured_materials.append(kwargs)
        return len(self.textured_materials) - 1

    def set_instance_material_ids_host(self, material_ids):
        self.uploaded_material_ids = np.asarray(material_ids).copy()


class _FakePathTracerAPI:
    def __init__(self):
        self.scene = _FakeScene()
        self.viewer = SimpleNamespace(tonemapped_output=None)
        self.materials = []
        self.sky = None
        self.camera = None
        self.build_count = 0
        self.refit_count = 0
        self.render_count = 0
        self.debug_mode = 0
        self.dlss_enabled = True
        self.tonemap_exposure = 1.0
        self.tonemap_contrast = 1.0
        self.tonemap_saturation = 1.0
        self.auto_exposure_enabled = False
        self.auto_exposure_config = None
        self.analytic_light_intensity = 1.0
        self.emissive_material_intensity = 1.0
        self.usd_scene = None
        self.usd_load = None

    def initialize(self):
        return True

    def set_use_procedural_sky(self, enabled):
        self.procedural_sky = enabled

    def set_sky_parameters(self, **kwargs):
        self.sky = kwargs

    def set_camera_look_at(self, position, target, up, fov):
        self.camera = (np.asarray(position), np.asarray(target), tuple(up), fov)

    def create_pbr_material(
        self,
        color,
        roughness,
        metallic,
        ior=1.5,
        specular=1.0,
        clearcoat=0.0,
        clearcoat_roughness=0.1,
        u_subdiv=0.0,
        v_subdiv=0.0,
        base_color_scale=0.75,
    ):
        self.materials.append(
            (
                tuple(color),
                roughness,
                metallic,
                ior,
                specular,
                clearcoat,
                clearcoat_roughness,
                u_subdiv,
                v_subdiv,
                base_color_scale,
            )
        )
        return len(self.materials) - 1

    def create_mesh(self, positions, indices, normals, uvs, material_id):
        self.scene._meshes.append(SimpleNamespace(material_id=material_id))
        return len(self.scene._meshes) - 1

    def create_instance(self, mesh_id):
        self.scene._instances.append(
            SimpleNamespace(
                mesh_index=mesh_id,
                material_id=None,
                visible=True,
                transform=np.eye(4, dtype=np.float32),
            )
        )
        return len(self.scene._instances) - 1

    def set_instance_visible(self, instance_id, visible):
        self.scene._instances[instance_id].visible = visible

    def set_instance_transform_matrix(self, instance_id, matrix):
        self.scene._instances[instance_id].transform = np.asarray(matrix).copy()

    def set_instance_transform_matrices(self, instance_ids, matrices):
        for instance_id, matrix in zip(instance_ids, matrices, strict=True):
            self.scene._instances[instance_id].transform = np.asarray(matrix).copy()

    def set_instances_visible(self, instance_ids, visible):
        for instance_id in instance_ids:
            self.scene._instances[instance_id].visible = bool(visible)

    def set_instance_material(self, instance_id, material_id):
        self.scene._instances[instance_id].material_id = material_id

    def build_scene(self):
        self.build_count += 1

    def rebuild_tlas(self):
        self.refit_count += 1

    def render_frame(self):
        self.render_count += 1

    def get_frame_uint8(self):
        return np.full((2, 3, 4), 127, dtype=np.uint8)

    def set_debug_buffer_mode(self, mode):
        self.debug_mode = int(mode)

    def set_environment_color(self, color):
        self.environment_color = tuple(color)

    def configure_auto_exposure(self, enabled, **kwargs):
        self.auto_exposure_enabled = bool(enabled)
        self.auto_exposure_config = kwargs

    def clear_scene(self):
        self.scene = _FakeScene()

    def load_scene_from_usd(self, path, **kwargs):
        self.usd_load = (path, kwargs)
        self.usd_scene = object()
        return True

    def close(self):
        pass


def _triangle():
    return (
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
        np.array(((0, 1, 2),), dtype=np.uint32),
    )


def test_scene_batches_instance_transforms_and_visibility():
    scene = Scene(None)
    instance_ids = [scene.add_instance(0) for _ in range(3)]
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    transforms[:, 0, 3] = (1.0, 2.0, 3.0)

    scene.set_instance_transforms_batch(instance_ids, transforms)
    scene.set_instances_visible_batch(instance_ids[1:], False)

    np.testing.assert_array_equal(scene._instance_transform_cache[:3], transforms)
    np.testing.assert_array_equal(scene._instance_visibility_cache[:3], (True, False, False))


def test_reference_sky_and_srgb_color_mapping():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    points, indices = _triangle()

    viewer.log_mesh(
        "triangle",
        points,
        indices,
        color=(0.8, 0.8, 0.8),
        roughness=0.25,
        metallic=0.75,
    )

    assert api.procedural_sky is True
    assert api.sky is None
    np.testing.assert_allclose(api.materials[0][0], (0.60382736, 0.60382736, 0.60382736), rtol=1.0e-6)
    assert api.materials[0][1:3] == (0.25, 0.75)
    assert api.materials[0][3:] == (1.5, 1.0, 0.0, 0.1, 0.0, 0.0, 0.75)

    viewer.set_sky_parameters(
        sun_direction=(0.0, 1.0, 0.0),
        ground_color=(0.5, 0.25, 0.75),
        night_color=(0.2, 0.4, 0.6),
    )
    viewer.set_environment_color((0.1, 0.5, 0.9))
    viewer.tonemap_saturation = 1.25
    viewer.tonemap_contrast = 1.1
    viewer.tonemap_exposure = 0.75
    viewer.analytic_light_intensity = 0.4
    viewer.emissive_material_intensity = 0.6
    assert viewer.tonemap_exposure == 0.75
    assert viewer.tonemap_saturation == 1.25
    assert viewer.tonemap_contrast == 1.1
    assert viewer.analytic_light_intensity == 0.4
    assert viewer.emissive_material_intensity == 0.6

    viewer.configure_auto_exposure(True, min_ev=-5.0, max_ev=7.0)
    assert viewer.auto_exposure_enabled
    assert api.auto_exposure_config == {
        "target_luminance": None,
        "min_ev": -5.0,
        "max_ev": 7.0,
        "brighten_speed": None,
        "darken_speed": None,
    }
    np.testing.assert_allclose(api.sky["ground_color"], (0.21404114, 0.05087609, 0.52252155), rtol=1.0e-6)
    np.testing.assert_allclose(api.sky["night_color"], (0.03310477, 0.13286832, 0.31854678), rtol=1.0e-6)
    np.testing.assert_allclose(api.environment_color, (0.01002283, 0.21404114, 0.78741229), rtol=1.0e-6)


def test_log_mesh_binds_and_reuses_base_color_texture():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    points, indices = _triangle()
    uvs = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    texture = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
        dtype=np.uint8,
    )

    viewer.log_mesh("first", points, indices, uvs=uvs, texture=texture)
    viewer.log_mesh("second", points, indices, uvs=uvs, texture=texture)
    viewer.log_instances(
        "batch",
        "first",
        np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.array(((0.5, 0.0, 0.0, 1.0),), dtype=np.float32),
    )

    assert len(api.scene._gltf_textures) == 1
    assert api.scene._gltf_textures[0].shape == (2, 2, 4)
    np.testing.assert_array_equal(api.scene._gltf_textures[0][0, 0, :3], (0, 0, 255))
    assert len(api.scene.textured_materials) == 2
    assert all(
        material["base_color_texture"] == {"index": 0, "texCoord": 0} for material in api.scene.textured_materials
    )
    assert api.scene._meshes[0].material_id == api.scene._meshes[1].material_id
    assert api.scene._instances[0].material_id == 1
    assert api.materials == []


def test_sky_parameters_normalize_sun_direction():
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace()
    api.initialize = lambda: True

    api.set_sky_parameters((0.0, 2.0, 1.0))
    assert api._viewer.sky_grayscale == pytest.approx(0.5)
    assert api._viewer.sky_ground_color == (0.4, 0.4, 0.4)
    api.set_sky_parameters((0.0, 2.0, 1.0), grayscale=True)

    assert api._viewer.sky_grayscale == 1.0
    api.set_sky_parameters((0.0, 2.0, 1.0), grayscale=0.35)
    assert api._viewer.sky_grayscale == pytest.approx(0.35)
    np.testing.assert_allclose(api._viewer.sky_sun_direction, (0.0, 0.8944272, 0.4472136), rtol=1.0e-6)
    with pytest.raises(ValueError, match="nonzero"):
        api.set_sky_parameters((0.0, 0.0, 0.0))


def test_night_preset_enables_moon_and_visible_sky():
    """Enable an antipodal moon disk and a nonblack night atmosphere."""
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)

    viewer.set_time_of_day("night")

    assert api.sky["sun_direction"][1] < 0.0
    assert api.sky["sun_disk_intensity"] == pytest.approx(1.0)
    assert api.sky["sun_disk_scale"] == pytest.approx(1.0)
    assert api.sky["sun_glow_intensity"] == pytest.approx(0.15)
    night_peak = max(api.sky["night_color"])
    assert night_peak > 0.0
    assert night_peak < 0.001


def test_light_intensity_controls_clamp_and_reset_history():
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(
        analytic_light_intensity=1.0,
        emissive_material_intensity=1.0,
        _dlss_reset_history=False,
    )

    api.analytic_light_intensity = -1.0
    assert api.analytic_light_intensity == 0.0
    assert api._viewer._dlss_reset_history is True

    api._viewer._dlss_reset_history = False
    api.emissive_material_intensity = 0.25
    assert api.emissive_material_intensity == 0.25
    assert api._viewer._dlss_reset_history is True


def test_load_scene_from_usd_enables_composed_environment_before_building():
    calls = []
    scene = SimpleNamespace(
        load_from_usd=lambda *args, **kwargs: calls.append(("load", kwargs["clear_existing"])) or True,
        usd_environment_path="environment.hdr",
    )
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    api.initialize = lambda: True
    api.build_scene = lambda: calls.append(("build",))
    api.set_environment_hdr = lambda path, scaling=1.0: calls.append((path, scaling))

    assert api.load_scene_from_usd("asset.usd", clear_existing=False)
    assert calls == [("load", False), ("build",)]
    calls.clear()
    assert api.load_scene_from_usd("asset.usd", load_usd_environment=True, usd_environment_scale=0.25)
    assert calls == [("load", True), ("environment.hdr", 0.25), ("build",)]


def test_resize_releases_old_dlss_resources_before_native_reallocation(monkeypatch):
    viewer = PathTracingRenderer.__new__(PathTracingRenderer)
    viewer.width = 800
    viewer.height = 600
    viewer._render_stream = object()
    viewer.frame_index = 9
    events = []
    viewer.camera = SimpleNamespace(set_aspect_ratio=lambda width, height: events.append(("aspect", width, height)))
    viewer._tonemapper = SimpleNamespace(resize=lambda width, height: events.append(("tonemap", width, height)))
    viewer._sync_prev_camera_matrices_to_current = lambda: events.append(("camera",))
    viewer._destroy_dlss_rr = lambda **kwargs: events.append(("destroy", kwargs))
    viewer._destroy_optix_denoiser = lambda: events.append(("destroy_optix",))
    viewer._init_denoiser = lambda: events.append(("init",))
    monkeypatch.setattr(viewer_module.wp, "synchronize_stream", lambda stream: events.append(("sync",)))

    viewer.resize(3840, 2160)

    assert viewer.width == 3840
    assert viewer.height == 2160
    assert events[:2] == [("sync",), ("destroy", {"restore_resolution": False})]
    assert events[2] == ("destroy_optix",)
    assert events[-2:] == [("init",), ("tonemap", 3840, 2160)]


def test_logged_instances_apply_up_axis_materials_and_frame_lifecycle():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, num_frames=1, up_axis="Z", api=api)
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)
    scales = np.array(((2.0, 3.0, 4.0),), dtype=np.float32)
    colors = np.array(((1.0, 0.0, 0.0),), dtype=np.float32)
    materials = np.array(((0.2, 0.6, 12.5, 7.25),), dtype=np.float32)

    viewer.log_instances("batch", "triangle", xforms, scales, colors, materials)
    instance = api.scene._instances[0]
    np.testing.assert_allclose(instance.transform[:3, 3], (1.0, 3.0, -2.0))
    np.testing.assert_allclose(instance.transform[:3, :3], ((2, 0, 0), (0, 0, 4), (0, -3, 0)))
    assert instance.visible is True
    assert api.materials[instance.material_id][1:3] == (0.2, 0.6)
    assert api.materials[instance.material_id][3:] == (
        1.5,
        1.0,
        0.0,
        0.1,
        12.5,
        7.25,
        0.75,
    )

    viewer.end_frame()

    assert api.build_count == 1
    assert api.render_count == 1
    assert not viewer.is_running()


def test_instance_visibility_and_cached_material_updates():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    viewer.log_instances("batch", "triangle", xforms, None, None, None)
    viewer.end_frame()
    viewer.update_instance_transforms("batch", xforms[:1])
    viewer.end_frame()

    assert api.build_count == 1
    assert api.refit_count == 1
    assert api.scene._instances[0].visible is True
    assert api.scene._instances[1].visible is False
    assert api.scene.uploaded_material_ids is not None


class _FakePicking:
    def __init__(self, model, pick_stiffness, pick_damping):
        self.model = model
        self.settings = (pick_stiffness, pick_damping)
        self.active = False
        self.applied = []
        self.picked = None
        self.updated = None

    def is_picking(self):
        return self.active

    def _apply_picking_force(self, state):
        self.applied.append(state)

    def pick(self, state, origin, direction):
        self.active = True
        self.picked = (state, tuple(origin), tuple(direction))

    def update(self, origin, direction):
        self.updated = (tuple(origin), tuple(direction))

    def release(self):
        self.active = False


def test_optional_picking_uses_physics_camera_ray_and_applies_forces():
    api = _FakePathTracerAPI()
    created = []

    def factory(*args, **kwargs):
        picking = _FakePicking(*args, **kwargs)
        created.append(picking)
        return picking

    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api, up_axis="Z", picking_factory=factory)
    model = SimpleNamespace(up_axis=2)
    viewer.set_model(model)
    picking = created[0]
    assert picking.settings == (50.0, 5.0)

    state = object()
    origin, direction = viewer._get_ray_from_mouse(viewer.width / 2, viewer.height / 2)
    np.testing.assert_allclose(origin, viewer._camera_position)
    np.testing.assert_allclose(direction, viewer._physics_camera_front(), atol=1.0e-6)
    picking.pick(state, origin, direction)
    viewer.log_state(state)
    viewer.apply_forces(state)
    assert picking.applied == [state]


def test_debug_and_bridge_transform_compatibility():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)

    viewer.set_debug_buffer_mode(7)
    assert api.debug_mode == 7
    key = SimpleNamespace(
        SPACE=10,
        ESCAPE=11,
        P=14,
        _0=20,
        _1=21,
        _2=22,
        _3=23,
        _4=24,
        _5=25,
        _6=26,
        _7=27,
        _8=28,
        BACKSPACE=29,
    )
    viewer._presenter = SimpleNamespace(pyglet=SimpleNamespace(window=SimpleNamespace(key=key)))
    viewer.on_key_press(key._1, None)
    assert api.debug_mode == 2
    viewer.on_key_press(key._1, None)
    assert api.debug_mode == 0
    viewer._presenter = None
    assert viewer.get_instance_transform_gl_buffer() == 0
    assert viewer.get_instance_transform_capacity() == 10000
    assert not viewer.is_gpu_transform_available()


def test_screenshot_exports_display_orientation(tmp_path):
    api = _FakePathTracerAPI()
    frame = np.zeros((2, 3, 4), dtype=np.uint8)
    frame[0, :, 0] = 255
    frame[1, :, 2] = 255
    api.get_frame_uint8 = lambda: frame
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)

    path = viewer.save_screenshot(tmp_path / "capture.png")
    image = np.asarray(Image.open(path))
    np.testing.assert_array_equal(image[0], frame[1, :, :3])
    np.testing.assert_array_equal(image[1], frame[0, :, :3])


def test_viewer_loads_usd_through_public_backend_api():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    viewer._mesh_ids["old"] = 1

    loaded = viewer.load_scene_from_usd(
        "scene.usd",
        max_texture_size=2048,
        load_usd_environment=True,
    )

    assert loaded
    assert api.usd_load == (
        "scene.usd",
        {
            "clear_existing": True,
            "apply_stage_units": True,
            "convert_up_axis": True,
            "max_texture_size": 2048,
            "max_texture_memory_bytes": None,
            "strict_sidedness": False,
            "enable_emissive_materials": True,
            "load_usd_lights": False,
            "usd_light_radius": None,
            "load_usd_environment": True,
            "usd_environment_scale": 1.0,
        },
    )
    assert viewer.usd_scene is api.usd_scene
    assert viewer._mesh_ids == {}


def test_renderer_space_look_at_preserves_level_z_up_camera():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, up_axis="Z", api=api)
    viewer._ensure_initialized()

    position = np.array((1.0, 2.0, 3.0), dtype=np.float32)
    target = np.array((4.0, 6.0, 3.0), dtype=np.float32)
    viewer.set_camera_look_at(position, target, fov=52.0, renderer_space=True)

    camera_position, camera_target, camera_up, camera_fov = api.camera
    np.testing.assert_allclose(camera_position, position)
    np.testing.assert_allclose(
        camera_target - camera_position,
        (target - position) / np.linalg.norm(target - position),
        atol=1.0e-6,
    )
    assert camera_up == (0.0, 1.0, 0.0)
    assert camera_fov == 52.0


def test_instance_capacity_is_enforced():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api, max_instances=1)
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    with pytest.raises(RuntimeError, match="instance capacity"):
        viewer.log_instances("batch", "triangle", xforms, None, None, None)


def test_volume_composite_launches_only_when_bound(monkeypatch):
    """Launch no volume work unless a volume is bound."""
    launches = []
    viewer = viewer_module.PathTracingViewer.__new__(viewer_module.PathTracingViewer)
    viewer.sample_index = 0
    viewer._render_width = 16
    viewer._render_height = 8
    viewer._dlss_enabled = True
    viewer._launch_params = object()
    viewer._volume_params = None
    viewer._render_stream = object()
    viewer._update_launch_params = lambda **kwargs: None
    viewer._launch_optix = lambda: None
    monkeypatch.setattr(
        viewer_module.wp,
        "launch",
        lambda kernel, **kwargs: launches.append(kernel),
    )
    matrix = np.eye(4, dtype=np.float32)

    viewer._volume = None
    viewer._launch_samples(
        1,
        False,
        view=matrix,
        proj=matrix,
        view_inv=matrix,
        proj_inv=matrix,
    )
    assert launches == []

    viewer._volume = SimpleNamespace(id=1)
    viewer._volume_min = (0.0, 0.0, 0.0)
    viewer._volume_max = (1.0, 1.0, 1.0)
    viewer._volume_density_scale = 1.0
    viewer._volume_step_size = 0.1
    viewer._volume_cool_colors = ((0.0, 0.0, 0.0),) * 3
    viewer._volume_warm_colors = ((0.0, 0.0, 0.0),) * 3
    viewer._volume_emission = 0.0
    viewer._volume_anisotropy = 0.0
    viewer._volume_transfer_table = None
    viewer._volume_density_feature = False
    viewer._launch_samples(
        1,
        False,
        view=matrix,
        proj=matrix,
        view_inv=matrix,
        proj_inv=matrix,
    )
    assert launches == [viewer_module.composite_volume]


def test_tlas_refit_keeps_but_scene_rebuild_resets_temporal_history():
    class _Scene:
        def __init__(self):
            self.refit_count = 0

        def rebuild_tlas(self):
            self.refit_count += 1

        def build(self, optix):
            self.optix = optix

    scene = _Scene()
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(
        _scene=scene,
        _optix=object(),
        _dlss_reset_history=False,
        _prev_instance_transforms_valid=True,
        sample_index=17,
        frame_index=9,
        _sync_prev_camera_matrices_to_current=lambda: None,
    )

    api.rebuild_tlas()
    assert scene.refit_count == 1
    assert (api._viewer.sample_index, api._viewer.frame_index) == (17, 9)

    api.build_scene()
    assert scene.optix is api._viewer._optix
    assert (api._viewer.sample_index, api._viewer.frame_index) == (0, 0)
    assert api._viewer._dlss_reset_history is True
    assert api._viewer._prev_instance_transforms_valid is False


def test_quality_modes_and_runtime_ray_budgets():
    assert PathTracingRenderer._normalize_dlss_quality("ultra-performance") == "ultra_performance"
    assert PathTracingRenderer._normalize_dlss_quality("native") == "native"
    with pytest.raises(ValueError, match="dlss_quality"):
        PathTracingRenderer._normalize_dlss_quality("cinematic")

    renderer = PathTracingRenderer.__new__(PathTracingRenderer)
    renderer._pipeline_max_bounces = 6
    renderer.max_bounces = 4
    renderer.direct_light_samples = 1
    renderer.russian_roulette_start_bounce = 3
    renderer.samples_per_frame = 1

    renderer.set_ray_budget(
        max_bounces=2,
        direct_light_samples=3,
        russian_roulette_start_bounce=5,
        samples_per_frame=4,
    )

    assert renderer.max_bounces == 2
    assert renderer.direct_light_samples == 3
    assert renderer.russian_roulette_start_bounce == 5
    assert renderer.samples_per_frame == 4
    with pytest.raises(ValueError, match="compiled limit"):
        renderer.set_ray_budget(max_bounces=7)
    with pytest.raises(ValueError, match="at least 1"):
        renderer.set_ray_budget(direct_light_samples=0)
    with pytest.raises(ValueError, match="at least 1"):
        renderer.set_ray_budget(russian_roulette_start_bounce=0)


def test_global_backface_culling_can_be_changed_at_runtime():
    renderer = PathTracingRenderer.__new__(PathTracingRenderer)
    renderer.backface_culling = True
    renderer.sample_index = 17
    renderer.frame_index = 9
    renderer._dlss_reset_history = False

    renderer.set_backface_culling(False)

    assert renderer.backface_culling is False
    assert renderer.sample_index == 0
    assert renderer.frame_index == 0
    assert renderer._dlss_reset_history is True

    renderer.sample_index = 4
    renderer.frame_index = 3
    renderer._dlss_reset_history = False
    renderer.set_backface_culling(False)
    assert renderer.sample_index == 4
    assert renderer.frame_index == 3
    assert renderer._dlss_reset_history is False

    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = renderer
    assert api.backface_culling is False
    api.set_backface_culling(True)
    assert api.backface_culling is True


@pytest.mark.gpu
@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_cuda_instance_batches_reuse_materials_after_count_changes():
    """Shrinking, hiding, and restoring batches must not rebuild materials."""
    api = _FakePathTracerAPI()
    material_updates = []
    api.set_instance_material_arrays = lambda *args: material_updates.append(args)
    viewer = PathTracingViewerBackend(device="cuda", headless=True, api=api)
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    colors = wp.array(np.ones((3, 3), dtype=np.float32), dtype=wp.vec3, device="cuda")
    materials = wp.array(np.ones((3, 4), dtype=np.float32), dtype=wp.vec4, device="cuda")
    xforms = np.tile((0, 0, 0, 0, 0, 0, 1), (3, 1)).astype(np.float32)
    viewer.log_instances("batch", "triangle", xforms, None, colors, materials)
    viewer._flush_scene()
    assert len(material_updates) == 1
    viewer.log_instances("batch", "triangle", xforms, None, None, None)
    assert len(material_updates) == 1
    material_count = len(api.materials)
    for count in (2, 1, 3):
        viewer.log_instances("batch", "triangle", xforms[:count], None, colors[:count], materials[:count])
        viewer._flush_scene()
        assert len(api.materials) == material_count
        assert api.build_count == 1
    viewer.log_instances("batch", "triangle", None, None, None, None, hidden=True)
    viewer._flush_scene()
    assert all(not instance.visible for instance in api.scene._instances)
    viewer.log_instances("batch", "triangle", xforms, None, colors, materials)
    viewer._flush_scene()
    assert all(instance.visible for instance in api.scene._instances)
    assert len(api.materials) == material_count
    assert api.build_count == 1
