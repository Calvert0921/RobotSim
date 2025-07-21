import numpy as np
import gymnasium as gym
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as R

@register_env("PickCubeMultiCam-v1", max_episode_steps=50)
class PickCubeMultiCamEnv(PickCubeEnv):
    @property
    def _default_sensor_configs(self):
        # start with the existing camera(s)
        configs = super()._default_sensor_configs
        
        # 0) Find the base_camera, resize it, and move it back+up for a full‐scene view
        for cfg in configs:
            if cfg.uid == "base_camera":
                # bump to 512×512
                cfg.width, cfg.height = 512, 512

                # reposition: e.g., 2m back on -Y, 1.5m up, looking at table center
                new_pose = sapien_utils.look_at(
                    eye    = [0.4,  0.3, 0.7],    # behind the robot along -Y
                    target = [0.0,  0.0, 0.1],    # table center
                    up     = [0.0,  0.0, 1.0]     # world Z up
                )
                cfg.pose = new_pose
                break

        # 1) Top-down camera, looking straight down at the table center
        top_pose = sapien_utils.look_at(
            eye=[0.0, 0.0, 1.5],        # 2m above the origin
            up= [1.0, 0.0, 0.0],        # ← use world +Y as the ‘up’ direction
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
            eye=[0.0, 1.0, 0.5],        # off to the robot’s right
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
        print(f"Joints: {self.agent.robot.active_joints.get_name()}")
        ee_link = next(l for l in self.agent.robot.get_links()
               if l.name == "panda_hand_tcp")

        # build a local mount‐frame pose:
        #    - move 5 cm forward along the hand’s +X
        #    - tilt down 30° about the hand’s +Y so "forward" points into the scene
        translation = [0.05, 0.0, 0.0]  
        # a 30° pitch down is a rotation of +30° about the hand’s Y axis:
        pitch = np.deg2rad(30)
        # Create an (x,y,z,w) quaternion from XYZ‐Euler
        quat = R.from_euler("xyz", [0.0, pitch, 0.0]).as_quat()

        wrist_cam_pose = Pose.create_from_pq(p=translation, q=quat)

        # append the camera config
        configs.append(
            CameraConfig(
                uid="wrist_camera",
                mount=ee_link,
                pose=wrist_cam_pose,    # local to the TCP link frame
                width=256, height=256,
                fov=np.pi/3,
                near=0.01, far=100
            )
        )

        return configs