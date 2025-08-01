import numpy as np
import gymnasium as gym
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
import mani_skill.envs.utils.randomization as randomization
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
        # top_pose = sapien_utils.look_at(
        #     eye=[0.0, 0.0, 1.5],        # 2m above the origin
        #     up= [1.0, 0.0, 0.0],        # ← use world +Y as the ‘up’ direction
        #     target=[0.0, 0.0, 0.1]      # looking at the cube area
        # )
        # configs.append(
        #     CameraConfig(
        #         uid="top_camera",
        #         pose=top_pose,
        #         width=256, height=256,
        #         fov=np.pi/3,
        #         near=0.01, far=100
        #     )
        # )
        
        top_pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.4],        # 2m above the origin
            target=[-0.1, 0, 0.1]      # looking at the cube area
        )
        configs.append(
            CameraConfig(
                uid="top_camera",
                pose=top_pose,
                width=256, height=256,
                fov=np.pi/2,
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

        return configs

@register_env("PickMultiCubeMultiCam-v1", max_episode_steps=50)
class PickMultiCubeMultiCamEnv(PickCubeEnv):
    #----------------------------------
    # Scene Loading (add blue cube)
    #----------------------------------
    def _load_scene(self, options: dict):
        # call base to build table and red cube
        super()._load_scene(options)
        # add a second, blue cube
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],  # blue
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0.1, 0, self.cube_half_size])
        )

    #----------------------------------
    # Episode Initialization (randomize both cubes)
    #----------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 1) do all the normal red‐cube initialization
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)

        # 2) read out red‐cube positions (shape: [b,3])
        #    .pose.p is a (b,3) tensor of current cube centers
        red_pos = self.cube.pose.p  

        # 3) set your minimum clearance:
        #    two diameters plus a little margin, e.g. 2*0.02*2 + 0.02 = 0.06m
        margin = 0.05
        d_min = 2 * self.cube_half_size + margin  

        # 4) sample blue‐cube XY until it’s >= d_min from the corresponding red XY
        xyz_blue = torch.zeros((b, 3), device=self.device)
        for i in range(b):
            while True:
                candidate_xy = torch.rand(2, device=self.device) * 0.25 - 0.1
                if (candidate_xy - red_pos[i, :2]).norm() >= d_min:
                    xyz_blue[i, :2] = candidate_xy
                    break
            xyz_blue[i, 2] = self.cube_half_size

        # 5) randomize orientation as before
        qs_blue = randomization.random_quaternions(b, lock_x=True, lock_y=True)

        # 6) finally place the blue cube
        self.blue_cube.set_pose(Pose.create_from_pq(xyz_blue, qs_blue))

    #----------------------------------
    # Observations (include blue cube info)
    #----------------------------------
    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            blue_cube_pose=self.blue_cube.pose.raw_pose,
            tcp_to_blue=self.blue_cube.pose.p - self.agent.tcp.pose.p
        )
        return obs
    #----------------------------------
    # Add extra cameras
    #----------------------------------
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
        
        top_pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.4],        # 2m above the origin
            target=[-0.1, 0, 0.1]      # looking at the cube area
        )
        configs.append(
            CameraConfig(
                uid="top_camera",
                pose=top_pose,
                width=256, height=256,
                fov=np.pi/2,
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

        return configs
    
@register_env("PegInsertionSideMultiCam-v1", max_episode_steps=50)
class PegInsertionSideMultiCamEnv(PegInsertionSideEnv):
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
        # define the same “in‑box” target and eye position:
        eye    = [0.35, -0.3, 0.2]           # same as your base example
        target = [0, -0.1, 0.05]          # in box direction

        # build the look‑at pose (no need to pass ‘up’ here unless you want a non‑Z‑up orientation)
        top_pose = sapien_utils.look_at(eye, target)

        # append with the identical intrinsics/clip to the base camera
        configs.append(
            CameraConfig(
                uid="top_camera",         # keep your uid
                pose=top_pose,            # new look‑at
                width=256, height=256,    # same as base
                fov=np.pi/2,              # 90° FOV
                near=0.01, far=100         # identical clip planes
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
        return configs