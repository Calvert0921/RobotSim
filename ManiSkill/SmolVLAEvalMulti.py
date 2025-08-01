# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path
import os
import sys

import gymnasium as gym
import mani_skill.envs
import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import OBS_STATE
from CamHandler import *
from utils import *

def eval_once():
    # Reset the policy and environments to prepare for rollout
    policy.reset()
    raw_observation, info = env.reset()

    # Open image writer
    video_path = output_dir / f"base_camera_{iter+1}.mp4"
    writer = imageio.get_writer(str(video_path), fps=fps)
    
    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    step = 0
    done = False
    while not done:
        # Prepare observation for the policy running in Pytorch
        state = raw_observation["agent"]["qpos"].unsqueeze(0)
        # print(f"{step=} {state=}")
        
        base_image = raw_observation["sensor_data"]["base_camera"]["rgb"]      # Shape: [1, 256, 256, 3]
        top_image = raw_observation["sensor_data"]["top_camera"]["rgb"]
        side_image = raw_observation["sensor_data"]["side_camera"]["rgb"]
        wrist_image = raw_observation["sensor_data"]["hand_camera"]["rgb"]

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        base_image = (base_image.to(torch.float32)
                    .permute(0, 3, 1, 2)
                    /255.0
                    )
        top_image = (top_image.to(torch.float32)
                    .permute(0, 3, 1, 2)
                    /255.0
                    )
        side_image = (side_image.to(torch.float32)
                    .permute(0, 3, 1, 2) 
                    /255.0
                    )
        wrist_image = (wrist_image.to(torch.float32)
                    .permute(0, 3, 1, 2) 
                    /255.0
                    )

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        base_image = base_image.to(device, non_blocking=True)
        top_image = top_image.to(device, non_blocking=True)
        side_image = side_image.to(device, non_blocking=True)
        wrist_image = wrist_image.to(device, non_blocking=True)

        # Create the policy input dictionary
        if task == "pickcube" or task == "pickmulticube":
            batch = {
                OBS_STATE: state,
                "observation.images.up": top_image,
                "observation.images.wrist": wrist_image,
                "task": "pick up the red cube."
            }
        elif task == "peginsertion":
            batch = {
                OBS_STATE: state,
                "observation.images.up": top_image,
                "observation.images.wrist": wrist_image,
                "task": "insert the peg in the hole."
            }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(batch)
            
        # print(f"{step=} {action=}")

        # Step through the environment and receive a new observation
        raw_observation, reward, terminated, truncated, info = env.step(action)
        # print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        
        # Append each camera frame to its writer
        rgba = raw_observation["sensor_data"]["base_camera"]["rgb"]
        rgb = rgba[0, ..., :3].cpu().numpy().astype(np.uint8)
        writer.append_data(rgb)

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1
        
    # Close writer
    writer.close()
    
    # Count number of successful round  
    if rewards[-1] >= 0.2:
        print(f"Iteration: {iter+1}, Success: True")
        return True
    else:
        print(f"Iteration: {iter+1}, Success: False")
        return False

        
if __name__ == "__main__":
    # Select task
    tasks = ["pickcube", "peginsertion", "pickmulticube"]
    task = tasks[0]

    # Select your device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda"
    
    # Disable parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load Policy
    if task == "pickcube" or task == "pickmulticube":
        pretrained_policy_path = "Calvert0921/smolvla_franka_liftcube_2000"
        dataset = LeRobotDataset("Calvert0921/SmolVLA_LiftCube_Franka_2000", download_videos=False)
    else:
        pretrained_policy_path = "Calvert0921/smolvla_franka_peginsertion_200"
        dataset = LeRobotDataset("Calvert0921/SmolVLA_PegInsertion_Franka", download_videos=False)
    policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path, dataset_stats=dataset.meta.stats)
    
    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    if task == "pickcube":
        env = gym.make(
            "PickCubeMultiCam-v1",
            robot_uids="panda_wristcam",
            num_envs=1,
            obs_mode="state_dict+rgb", 
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            max_episode_steps=200,
        )
    elif task == "pickcube":
        env = gym.make(
            "PegInsertionSideMultiCam-v1",
            robot_uids="panda_wristcam",
            num_envs=1,
            obs_mode="state_dict+rgb", 
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            max_episode_steps=1000,
        )
    else:
        env = gym.make(
            "PickMultiCubeMultiCam-v1",
            robot_uids="panda_wristcam",
            num_envs=1,
            obs_mode="state_dict+rgb", 
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            max_episode_steps=200,
        )
    
    # Number of iterations
    iterations = 100
    success = 0
    
    # Create output directory for videos
    output_dir = Path(f"./videos_smolvla_{iterations}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare imageio writers for base camera
    fps = env.metadata.get("render_fps", 30)
    
    # === BEGIN PRINT HISTORY REDIRECTION ===
    log_path = output_dir / "print_history.txt"
    log_file = open(log_path, "w")

    class Tee(object):
        def __init__(self, *writers):
            self.writers = writers
        def write(self, data):
            for w in self.writers:
                w.write(data)
        def flush(self):
            for w in self.writers:
                w.flush()

    # every print() from here on goes to both console and log_file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)  # optional: also capture errors
    # === END PRINT HISTORY REDIRECTION ===
    
    for iter in range(iterations):
        if eval_once():
            success += 1
    print(f"Success rate: {success / iterations}")