# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Camera class for path tracing viewer.
Provides first-person and orbit camera controls with matrix generation.
"""

import math

import numpy as np


class Camera:
    """First-person/orbit camera with smooth movement and rotation."""

    def __init__(
        self,
        position: tuple = (0.0, 0.0, 6.0),
        target: tuple = (0.0, 0.0, 0.0),
        up: tuple = (0.0, 1.0, 0.0),
        fov: float = 45.0,
        aspect_ratio: float = 16.0 / 9.0,
        near: float = 0.1,
        far: float = 100.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        # Match reference movement speed factor.
        self._movement_speed = 1.0

    @property
    def forward(self) -> np.ndarray:
        """Get the forward direction vector."""
        direction = self.target - self.position
        return direction / np.linalg.norm(direction)

    @property
    def right(self) -> np.ndarray:
        """Get the right direction vector."""
        r = np.cross(self.forward, self.up)
        return r / np.linalg.norm(r)

    @property
    def distance(self) -> float:
        """Get the distance from position to target."""
        return np.linalg.norm(self.target - self.position)

    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix (look-at)."""
        return self._look_at(self.position, self.target, self.up)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the projection matrix.
        Uses RIGHT-HANDED perspective with 0-1 depth range (GLM perspectiveRH_ZO).
        Apply the Y-flip used by the renderer projection convention.
        """
        proj = self._perspective_rh_zo(math.radians(self.fov), self.aspect_ratio, self.near, self.far)
        proj[1, 1] *= -1.0
        return proj

    def get_view_inverse(self) -> np.ndarray:
        """Get the inverse view matrix (camera transform)."""
        return np.linalg.inv(self.get_view_matrix())

    def get_projection_inverse(self) -> np.ndarray:
        """Get the inverse projection matrix."""
        return np.linalg.inv(self.get_projection_matrix())

    def move_forward(self, distance: float):
        """Move the camera forward/backward along its view direction."""
        scaled = distance * self._movement_speed
        fwd = self.forward
        self.position += fwd * scaled
        self.target += fwd * scaled

    def move_right(self, distance: float):
        """Move the camera left/right."""
        scaled = distance * self._movement_speed
        right = self.right
        self.position += right * scaled
        self.target += right * scaled

    def move_up(self, distance: float):
        """Move the camera up/down."""
        scaled = distance * self._movement_speed
        up = self.up
        self.position += up * scaled
        self.target += up * scaled

    def rotate(self, yaw: float, pitch: float):
        """
        Rotate the camera view (first-person style).

        Args:
            yaw: Horizontal rotation in radians
            pitch: Vertical rotation in radians
        """
        # Match reference camera rotate behavior (spherical yaw/pitch around target).
        direction = self.target - self.position
        dist = np.linalg.norm(direction)
        if dist <= 1.0e-8:
            return

        # Convert to spherical coordinates.
        current_yaw = math.atan2(direction[0], direction[2])
        current_pitch = math.asin(direction[1] / dist)

        # Apply rotation with reference sign conventions.
        current_yaw -= yaw
        current_pitch += pitch

        # Clamp pitch to avoid gimbal lock.
        current_pitch = max(-math.pi * 0.49, min(math.pi * 0.49, current_pitch))

        # Convert back to Cartesian.
        direction[0] = dist * math.sin(current_yaw) * math.cos(current_pitch)
        direction[1] = dist * math.sin(current_pitch)
        direction[2] = dist * math.cos(current_yaw) * math.cos(current_pitch)

        self.target = self.position + direction

    def orbit(self, yaw: float, pitch: float):
        """
        Orbit the camera around the target point.

        Args:
            yaw: Horizontal rotation in radians
            pitch: Vertical rotation in radians
        """
        offset = self.position - self.target
        dist = np.linalg.norm(offset)

        # Convert to spherical coordinates
        current_yaw = math.atan2(offset[0], offset[2])
        current_pitch = math.asin(np.clip(offset[1] / dist, -1.0, 1.0))

        # Apply rotation
        current_yaw += yaw
        current_pitch += pitch

        # Clamp pitch
        current_pitch = max(-math.pi * 0.49, min(math.pi * 0.49, current_pitch))

        # Convert back to Cartesian
        offset[0] = dist * math.sin(current_yaw) * math.cos(current_pitch)
        offset[1] = dist * math.sin(current_pitch)
        offset[2] = dist * math.cos(current_yaw) * math.cos(current_pitch)

        self.position = self.target + offset

    def zoom(self, delta: float, min_distance: float = 0.5, max_distance: float = 50.0):
        """
        Zoom by changing the distance to target.

        Args:
            delta: Positive zooms in, negative zooms out
            min_distance: Minimum allowed distance
            max_distance: Maximum allowed distance
        """
        offset = self.position - self.target
        current_distance = np.linalg.norm(offset)
        new_distance = max(min_distance, min(max_distance, current_distance - delta))

        if current_distance > 0.001:
            self.position = self.target + (offset / current_distance) * new_distance

    def pan(self, delta_x: float, delta_y: float):
        """
        Pan the camera (moves target and position together) in screen space.

        Args:
            delta_x: Horizontal pan amount
            delta_y: Vertical pan amount
        """
        right = self.right
        up = np.cross(right, self.forward)

        offset = right * delta_x + up * delta_y
        self.position += offset
        self.target += offset

    @staticmethod
    def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at view matrix."""
        f = center - eye
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        result = np.eye(4, dtype=np.float32)
        result[0, 0] = s[0]
        result[1, 0] = s[1]
        result[2, 0] = s[2]
        result[0, 1] = u[0]
        result[1, 1] = u[1]
        result[2, 1] = u[2]
        result[0, 2] = -f[0]
        result[1, 2] = -f[1]
        result[2, 2] = -f[2]
        result[3, 0] = -np.dot(s, eye)
        result[3, 1] = -np.dot(u, eye)
        result[3, 2] = np.dot(f, eye)

        return result

    @staticmethod
    def _perspective_rh_zo(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
        """
        Create a right-handed perspective matrix with 0-1 depth range.
        Matches GLM's glm::perspectiveRH_ZO.
        """
        tan_half_fov = math.tan(fov_y * 0.5)

        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = 1.0 / (aspect * tan_half_fov)
        result[1, 1] = 1.0 / tan_half_fov
        result[2, 2] = far / (near - far)
        result[2, 3] = -1.0
        # Match reference perspective projection convention (RH_ZO, M43).
        result[3, 2] = -(near * far) / (near - far)

        return result

    def set_aspect_ratio(self, width: int, height: int):
        """Update aspect ratio from render dimensions."""
        self.aspect_ratio = width / height if height > 0 else 1.0
