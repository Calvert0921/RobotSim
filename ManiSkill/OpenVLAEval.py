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

import gymnasium as gym
import mani_skill.envs
import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from CamHandler import PickCubeMultiCamEnv
from utils import *
from transformers import AutoModelForVision2Seq, AutoProcessor
from torchvision.transforms import ToPILImage

# Select your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"

# ====== Load Model ======
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

prompt = "In: What action should the robot take to {<pick up the red cube>}?\nOut:"

env = gym.make(
    "PickCubeMultiCam-v1",
    num_envs=1,
    obs_mode="state_dict+rgb", 
    # control_mode="pd_ee_delta_pose",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    max_episode_steps=100
)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
# print(f"Normalization mapping: {policy.config.normalization_mapping}")
# print(policy.config.input_features)
print(f"obs: {env.observation_space}\n")

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
# print(policy.config.output_features)
print(f"action: {env.action_space}")

# Reset the policy and environments to prepare for rollout
raw_observation, info = env.reset(seed=42)
# print(f"Observation: {raw_observation}")

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Create output directory for videos
output_dir = Path("./videos_openvla")
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare imageio writers for each camera
fps = env.metadata.get("render_fps", 30)
writers = {}
camera_uids = list(raw_observation["sensor_data"].keys())
for uid in camera_uids:
    video_path = output_dir / f"{uid}.mp4"
    writers[uid] = imageio.get_writer(str(video_path), fps=fps)
    print(f"Opened writer for {uid}: {video_path}")


step = 0
done = False
to_pil = ToPILImage()
while not done:
    # Prepare image
    img = raw_observation["sensor_data"]["base_camera"]["rgb"].squeeze().cpu().permute(2, 0, 1)      # Shape: [3, 256, 256]
    img = to_pil(img)

    # Predict the next action with respect to the current observation
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    action[0:6] *= 10 
        
    # Prepare the action for the environment
    # action = action.squeeze(0).astype(np.float32)
    print(f"{step=} {action=}")

    # Step through the environment and receive a new observation
    raw_observation, reward, terminated, truncated, info = env.step(action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    # Append each camera frame to its writer
    for uid, data in raw_observation["sensor_data"].items():
        rgba = data["rgb"]           # tensor [1,H,W,4]
        rgb = rgba[0, ..., :3].cpu().numpy().astype(np.uint8)
        writers[uid].append_data(rgb)

    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Close all writers
for uid, writer in writers.items():
    writer.close()
    print(f"Saved video for camera '{uid}' at: {output_dir/ (uid + '.mp4')}" )
    
# === Explicit cleanup to avoid segfault on exit ===
env.close()                            # close physics & rendering contexts
