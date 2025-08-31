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
from ManiSkill.CamHandler import *
from ManiSkill.EvalTasks.utils import *

COLORS = ["red", "blue", "green", "yellow", "black"]

def eval_once(target_colour):
    # Reset the policy and environments to prepare for rollout
    policy.reset()
    raw_observation, info = env.reset()
    
    # record the cube's initial pos
    init_pos = {c: get_xyz_from_obs(raw_observation, c) for c in
            ["red", "blue", "green", "yellow", "black"]}
    
    init_xy = init_pos[target_colour][:2]
    init_z = float(init_pos[target_colour][2].item())

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
        batch = {
            OBS_STATE: state,
            "observation.images.up": top_image,
            "observation.images.wrist": wrist_image,
            "task": f"pick up the {target_colour} cube"
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
        
        # Allocate buffers once per episode (put these near the top of eval_once)
        if step == 0:
            tcp_traj = []
            cubes_traj = {c: [] for c in COLORS}

        # Each step: collect positions
        tcp_traj.append(tcp_xyz(raw_observation).cpu().numpy())
        for c in COLORS:
            cubes_traj[c].append(cube_xyz(raw_observation, c).cpu().numpy())

        # check the cube's current Z
        curr_z = float(raw_observation["extra"][f"{target_colour}_cube_pose"][0, 2].item())
        
        if curr_z - init_z >= 0.1:
            print(f"Iteration: {iter+1}, Success: True")
            writer.close()
            return True, None, None
        
        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1
        
    # Close writer
    writer.close()
    
    # Count number of successful round  
    if curr_z - init_z >= 0.04:
        print(f"Iteration: {iter+1}, Success: True")
        return True, None, None
    else:
        print(f"Iteration: {iter+1}, Success: False")
        # After the episode finishes (and we determined it's a failure):
        # Check miss but contact
        cur_xy = raw_observation["extra"][f"{target_colour}_cube_pose"][0, :2]
        if not torch.allclose(init_xy, cur_xy, atol=1e-4):
            miss_but_contact = True
            print(f"[FAIL] contacted but fail to lift")
            return False, miss_but_contact, None
        
        # Check correct colour detected
        scores = intent_scores_from_traj(tcp_traj, cubes_traj, alpha=8.0)

        probs = softmax_probs(scores)
        pred_color = max(scores, key=scores.get)
        intended_correct = (pred_color == target_colour)

        print(f"[FAIL] predicted={pred_color}, target={target_colour}, probs={probs}")

        return False, None, intended_correct

        
if __name__ == "__main__":
    # Select your device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda"
    
    # Disable parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load Policy
    pretrained_policy_path = "Calvert0921/smolvla_franka_liftblackcube5_1000"
    dataset = LeRobotDataset("Calvert0921/SmolVLA_LiftBlackCube5_Franka_1000", download_videos=False)

    policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path, dataset_stats=dataset.meta.stats)
    
    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "PickFiveCubeMultiCam-v1",
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="state_dict+rgb", 
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=200,
    )
    
    # Number of iterations
    iterations = 200
    success = 0
    intend_true = 0
    contact_true = 0
    
    # Create output directory for videos
    output_dir = Path(f"/home/zhizhou/SawyerSim/Tests/liftcube_colour_{iterations}")
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
        target_colour = "black"
        ok, contacted, intended = eval_once(target_colour)
        if ok:
            success += 1
        else:
            if contacted:
                contact_true += 1
            elif intended:
                intend_true += 1
                
    print(f"Success rate: {success / iterations}")
    print(f"Contacted on failures: {(contact_true / max(1, (iterations - success))):.2f}")
    print(f"Intended-correct on failures: {(intend_true / max(1, (iterations - success - contact_true))):.2f}")