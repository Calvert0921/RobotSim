from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from demos.utils import encode_language_descriptions


# ========= Shared Base Class (not registered) =========
class _MultiColorLiftCubeEnv(BaseEnv):
    """
    LiftCube environment with five cubes (red, blue, green, yellow, black).
    Only one cube is the target, others are distractors.
    Subclasses should override TARGET_COLOR to specify which cube is the target.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    cube_half_size = 0.02
    goal_height = 0.08
    min_dist = 0.06  # minimum distance between cube centers

    TARGET_COLOR = "red"  # to be overridden in subclasses

    # RGBA values for each cube color
    COLOR_RGBA = {
        "red":    [1, 0, 0, 1],
        "blue":   [0, 0, 1, 1],
        "green":  [0, 1, 0, 1],
        "yellow": [1, 1, 0, 1],
        "black":  [0, 0, 0, 1],
    }

    # Fixed order of cube colors for consistent naming
    COLOR_ORDER = ["red", "blue", "green", "yellow", "black"]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Base camera for observations
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # Render camera for visualization
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Spawn the robot at fixed position
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Create five cubes
        self.cubes = {}
        for color in self.COLOR_ORDER:
            self.cubes[color] = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.COLOR_RGBA[color],
                name=f"{color}_cube",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )

        # Reference to the target cube
        self.target_color = self.TARGET_COLOR
        self.target_cube = self.cubes[self.target_color]

    # ---------- Sampling utility with safe distance ----------
    def _sample_xy_for_all_cubes(
        self, b: int, min_dist: float, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Sample XY positions for all cubes such that all pairs
        are at least min_dist apart. Uses batch resampling.
        """
        def sample_xy(n):
            return torch.rand((n, 2), device=device) * 0.25 - 0.1

        xy = {c: sample_xy(b) for c in self.COLOR_ORDER}
        max_iters = 128
        for _ in range(max_iters):
            changed = False
            for i, ci in enumerate(self.COLOR_ORDER):
                others = [c for j, c in enumerate(self.COLOR_ORDER) if j != i]
                dmins = []
                for cj in others:
                    d = torch.linalg.norm(xy[ci] - xy[cj], dim=1)
                    dmins.append(d)
                dmin = torch.stack(dmins, dim=1).min(dim=1).values
                mask = dmin < min_dist
                if mask.any():
                    xy[ci][mask] = sample_xy(mask.sum().item())
                    changed = True
            if not changed:
                break
        return xy

    def _get_language_description(self):
        # Language instruction always refers to the target cube
        descriptions = [f"pick up the {self.target_color} cube" for _ in range(self.num_envs)]
        return encode_language_descriptions(descriptions)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Sample safe XY positions for all cubes
            xy_dict = self._sample_xy_for_all_cubes(b, self.min_dist, self.device)

            z_col = torch.full((b, 1), self.cube_half_size, device=self.device)

            # Set poses for each cube
            for color, cube in self.cubes.items():
                xyz = torch.cat([xy_dict[color], z_col], dim=1)
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Goal height and language instruction
            self.goal_z = torch.tensor([self.goal_height], device=self.device)
            self.language_description = self._get_language_description()

    # ----- Observation and reward (target cube only) -----
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_z=self.goal_z,
            language_description=self.language_description,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.target_cube.pose.raw_pose,
                tcp_to_obj_pos=self.target_cube.pose.p - self.agent.tcp.pose.p,
                obj_z_to_goal_z=self.goal_z - self.target_cube.pose.p[:, 2],
            )
        return obs

    def evaluate(self):
        is_above = (self.goal_z - self.target_cube.pose.p[:, 2] <= 0)
        is_grasped = self.agent.is_grasping(self.target_cube)
        is_static = self.agent.is_static(0.2)
        return {
            "success": is_above & is_grasped,
            "is_obj_above_goal_z": is_above,
            "is_robot_static": is_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        dist = torch.linalg.norm(self.target_cube.pose.p - self.agent.tcp.pose.p, axis=1)
        reach_r = 1 - torch.tanh(5 * dist)
        reward = reach_r + info["is_grasped"]

        z_diff = self.goal_z - self.target_cube.pose.p[:, 2]
        reward += -5 * z_diff * info["is_grasped"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


# ========= Five target-color subclasses =========
@register_env("LiftCube", max_episode_steps=600)
class LiftCubeEnv(_MultiColorLiftCubeEnv):
    TARGET_COLOR = "black"