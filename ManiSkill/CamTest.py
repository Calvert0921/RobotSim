#!/usr/bin/env python3
"""
CamTest.py

Run the multi-camera PickCube environment with random actions,
capture each camera stream directly into per-camera MP4s (no PNGs saved).
"""
from pathlib import Path
import os
import gymnasium as gym
import imageio
import numpy as np

# Ensure ManiSkill envs are registered
import mani_skill.envs

# Import your custom multi-camera env
from CamHandler import PickCubeMultiCamEnv


def main():
    # Create output directory for videos
    output_dir = Path("./videos_random")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create multi-camera environment
    env = gym.make(
        "PickCubeMultiCam-v1",
        num_envs=1,
        obs_mode="rgb",  # raw sensor outputs including all cameras
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        max_episode_steps=300,
    )

    # Unwrap to base env to access sensor_data keys
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    # Determine camera UIDs from sensor_data mapping
    obs, _ = env.reset(seed=42)
    camera_uids = list(obs["sensor_data"].keys())
    print(f"Capturing video streams for cameras: {camera_uids}")

    # Prepare imageio writers for each camera
    fps = env.metadata.get("render_fps", 30)
    writers = {}
    for uid in camera_uids:
        video_path = output_dir / f"{uid}.mp4"
        writers[uid] = imageio.get_writer(str(video_path), fps=fps)
        print(f"Opened writer for {uid}: {video_path}")

    # Begin rollout with random actions
    max_steps = 300
    obs, info = env.reset(seed=42)
    for step in range(1, max_steps + 1):
        # Sample random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Append each camera frame to its writer
        for uid, data in obs["sensor_data"].items():
            rgba = data["rgb"]           # tensor [1,H,W,4]
            rgb = rgba[0, ..., :3].cpu().numpy().astype(np.uint8)
            writers[uid].append_data(rgb)

        if terminated or truncated:
            print(f"Episode finished after {step} steps, success={terminated}")
            break

    # Close all writers
    for uid, writer in writers.items():
        writer.close()
        print(f"Saved video for camera '{uid}' at: {output_dir/ (uid + '.mp4')}" )


if __name__ == "__main__":
    main()
