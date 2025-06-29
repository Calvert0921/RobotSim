import numpy as np
import gymnasium as gym
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

@register_env("PickCubeMultiCam-v1", max_episode_steps=50)
class PickCubeMultiCamEnv(PickCubeEnv):
    @property
    def _default_sensor_configs(self):
        # start with the existing camera(s)
        configs = super()._default_sensor_configs

        # 1) Top-down camera, looking straight down at the table center
        top_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 2.0],        # 2m above the origin
            target=[0.0, 0.0, 0.1]      # looking at the cube area
        )
        configs.append(
            CameraConfig(
                uid="top_camera",
                pose=top_pose,
                width=256, height=256,
                fov=np.pi/3,
                near=0.01, far=100
            )
        )

        # 2) Side camera, a static view from the +X direction
        side_pose = sapien_utils.look_at(
            eye=[2.0, 0.0, 0.5],        # off to the robot’s right
            target=[0.0, 0.0, 0.1]
        )
        configs.append(
            CameraConfig(
                uid="side_camera",
                pose=side_pose,
                width=256, height=256,
                fov=np.pi/3,
                near=0.01, far=100
            )
        )

        # 3) Wrist-mounted camera, parented to the end-effector link
        #    Here we grab the last link of the robot; adjust if your link names differ.
        ee_link = self.agent.robot.get_links()[-1]
        wrist_pose = Pose.create_from_pq(
            p=[0.0, 0.0, 0.0], 
            q=[1.0, 0.0, 0.0, 0.0]
        )
        configs.append(
            CameraConfig(
                uid="wrist_camera",
                pose=wrist_pose,
                width=256, height=256,
                fov=np.pi/3,
                near=0.01, far=100,
                mount=ee_link
            )
        )

        return configs