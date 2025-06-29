import os
import gymnasium as gym
import mani_skill.envs
import imageio
import numpy as np
import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# video output path
video_path = "./videos/pickcube_manual.mp4"
fps = 15

env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    render_mode="rgb_array"
)

obs, _ = env.reset(seed=0)
frames = []

done = False
while not done:
    # 随机动作（或你自己的动作）
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render() 返回 torch.Tensor 或 numpy.ndarray
    frame = env.render()
    # 如果是 batched：Tensor shape (1,H,W,3) 或 ndarray shape (1,H,W,3)
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    # 取第 0 个环境
    if isinstance(frame, np.ndarray) and frame.ndim == 4:
        frame = frame[0]
    # 确保 dtype 是 uint8
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    frames.append(frame)

env.close()

# 写入 MP4
with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
    for f in frames:
        writer.append_data(f)

print(f"recording finished, saved in: {os.path.abspath(video_path)}")
