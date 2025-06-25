import sapien.core as sapien
from sapien.core import Pose
import numpy as np
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaWristCam
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose as MSPose

# Step 1: Create SAPIEN engine and renderer
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)

# Step 2: Create simulation scene
scene_config = sapien.SceneConfig()
scene_config.solver_position_iteration_count = 8
scene_config.solver_velocity_iteration_count = 1
scene = engine.create_scene(scene_config)
scene.set_renderer(renderer)
scene.set_timestep(1 / 240)

# Step 3: Add lighting
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

# Step 4: Create GUI window
window = renderer.create_window(title="Panda + Cube Scene", width=1280, height=720)
camera = scene.add_camera("main_camera", width=1280, height=720, fovy=1.57, near=0.1, far=100)
camera.set_local_pose(Pose([1.0, 0.5, 0.6], euler2quat(-0.5, 0.5, -1.57)))
window.set_camera(camera)

# Step 5: Add table scene and load built-in Panda robot
table_scene = TableSceneBuilder(scene, robot_init_qpos_noise=0.02)
table_scene.build()

# Load the PandaWristCam robot using internal ManiSkill assets
robot = PandaWristCam(scene)
robot.load()
robot.set_root_pose(Pose([-0.615, 0, 0]))  # Position it to match default table scene
robot.reset()

# Step 6: Add a red cube to the scene
cube_half_size = 0.02  # Cube of size 4cm
material = scene.create_physical_material(static_friction=0.8, dynamic_friction=0.5, restitution=0.1)

cube_builder = scene.create_actor_builder()
cube_builder.add_box_collision(half_size=[cube_half_size]*3, material=material)
cube_builder.add_box_visual(half_size=[cube_half_size]*3, color=[1.0, 0.0, 0.0, 1.0])
cube = cube_builder.build(name="cube")
cube.set_pose(Pose([0.0, 0.0, cube_half_size + 0.001]))  # Place it on the table

# Step 7: Run GUI simulation
while not window.should_close():
    scene.step()
    scene.update_render()
    window.render()

# Cleanup (optional)
renderer.close()
