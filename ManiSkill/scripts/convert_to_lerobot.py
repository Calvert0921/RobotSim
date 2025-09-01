import shutil
import os
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from demos.utils import decode_language_descriptions
import h5py
import tyro

# Change to your Hugging Face username and desired dataset name
REPO_NAME = "Calvert0921/SmolVLA_LiftBlackCube5_Franka_500"


def main(data_path: str, *, push_to_hub: bool = False):
    # Clean up any existing data at the target location
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Define the exact feature schema matching SmolVLA's expectations
    features = {
        # Robot actions: 8 DoF
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "panda_joint1.pos",
                "panda_joint2.pos",
                "panda_joint3.pos",
                "panda_joint4.pos",
                "panda_joint5.pos",
                "panda_joint6.pos",
                "panda_joint7.pos",
                "gripper.pos",
            ],
        },
        # Proprioceptive state: 9 DoF
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "panda_joint1.pos",
                "panda_joint2.pos",
                "panda_joint3.pos",
                "panda_joint4.pos",
                "panda_joint5.pos",
                "panda_joint6.pos",
                "panda_joint7.pos",
                "finger1.pos",
                "finger2.pos",
            ],
        },
        # Upward-facing camera video
        "observation.images.up": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        # Side-facing camera video
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
    }

    # Create the LeRobot dataset for Franka follower at 30 fps
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="franka",
        fps=30,
        features=features,
        # use_videos=True,
        # video_backend="libaom-av1",
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Iterate through your raw data and add frames
    with h5py.File(data_path, "r") as raw:
        for episode in raw.values():
            actions = episode["actions"]  # shape: (T, 6)
            n = actions.shape[0]
            state = episode["obs"]["qpos"]  # shape: (T, 6)
            up_cam = episode["obs"]["image"][..., :3]
            wrist_cam = episode["obs"]["wrist_image"][..., :3]
            task_desc = decode_language_descriptions(
                episode["obs"]["language_description"][0]
            )[0]

            for i in range(n):
                dataset.add_frame({
                    "action": actions[i],
                    "observation.state": state[i],
                    "observation.images.up": up_cam[i],
                    "observation.images.wrist": wrist_cam[i],
                },
                task=task_desc,
                )

            # finalize this episode with the task label
            dataset.save_episode()

    # Push to HF Hub including videos if desired
    if push_to_hub:
        dataset.push_to_hub(
            tags=["smolvla", "franka", "custom"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            max_shard_size="100MB",               # keep each file under 100 MB
            chunk_size=2 * 1024 * 1024,           # 2 MiB per upload chunk
            num_threads=4                         # reduce concurrency
        )


if __name__ == "__main__":
    tyro.cli(main)
