import sim
import cv2
import torch
import time
import numpy as np
import sys
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
from scipy.spatial.transform import Rotation as R

def build_target_frame(x, y, z, roll, pitch, yaw):
    # Build rotation matirx
    r = R.from_euler('xyz', [roll, pitch, yaw])
    rot_matrix = r.as_matrix()  # shape: (3,3)

    # Build 4x4 transformation matrix
    target_frame = np.eye(4)
    target_frame[:3, :3] = rot_matrix
    target_frame[:3, 3] = [x, y, z]

    return target_frame

# ====== Connect Server ======
clientID = sim.simxStart('127.0.0.1', 19995, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')
    
sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

# ====== Get Handle ======
cam_handle = sim.simxGetObjectHandle(clientID, 'visionSensor', sim.simx_opmode_blocking)[1]
joint_handles = [sim.simxGetObjectHandle(clientID, f'right_j{i}', sim.simx_opmode_blocking)[1] for i in range(6, 13)]

# Initialize vision streaming
sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_streaming)
time.sleep(0.5)

# ====== Construct Sawyer Arm Structure ======
print(f"Constructing Sawyer Structure...")
sawyer_chain = Chain(name='sawyer', links=[

    # Base（OriginLink 固定）
    OriginLink(),

    # Joint 0: right_j6
    URDFLink(
        name='right_j6',
        origin_translation=[0.000123, 0.000119, 0.039506],
        origin_orientation=[-3.141593,  3.141593, -3.141593],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-3.0502998828888,  3.0502998828888)
    ),

    # Joint 1: right_j7
    URDFLink(
        name='right_j7',
        origin_translation=[0.081000,  0.061000,  0.230500],
        origin_orientation=[-1.570796,  1.570796,  0.000000],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-3.8183000087738,  2.2823998928070)
    ),

    # Joint 2: right_j8
    URDFLink(
        name='right_j8',
        origin_translation=[0.000002, -0.124498,  0.131502],
        origin_orientation=[-1.570796,  3.141593,  3.141247],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-3.0513999462128,  3.0513999462128)
    ),

    # Joint 3: right_j9
    URDFLink(
        name='right_j9',
        origin_translation=[0.000095, -0.047483,  0.275502],
        origin_orientation=[ 1.570796,  3.141247,  3.141593],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-3.0513999462128,  3.0513999462128)
    ),

    # Joint 4: right_j10
    URDFLink(
        name='right_j10',
        origin_translation=[0.000000, -0.113058, -0.120993],
        origin_orientation=[-1.570796,  3.141593,  3.141247],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-2.9842000007629,  2.9842000007629)
    ),

    # Joint 5: right_j11
    URDFLink(
        name='right_j11',
        origin_translation=[0.000100,  0.040504,  0.286949],
        origin_orientation=[ 1.570796,  3.141247,  3.141593],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-2.9842000007629,  2.9842000007629)
    ),

    # Joint 6: right_j12
    URDFLink(
        name='right_j12',
        origin_translation=[0.000001, -0.073999,  0.095801],
        origin_orientation=[-1.570834,  3.141630, -3.141593],
        rotation=[0.0, 0.0, 1.0],
        bounds=(-4.7104001045227,  4.7104001045227)
    ),

    # End-effector: BaxterGripper（fixed）
    URDFLink(
        name='BaxterGripper',
        origin_translation=[-0.000014, -0.000018,  0.074397],
        origin_orientation=[ 3.141586,  3.141604, -1.570715],
        joint_type='fixed'
    ),

])

sawyer_chain.active_links_mask = [False] + [True]*7 + [False]

# ====== Load Model ======
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b", 
#     # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True, 
#     trust_remote_code=True
# ).to("cuda:0")

# prompt = "In: What action should the robot take to {<pick up the cuboid>}?\nOut:"

# ====== Main Loop ======
while True:
    # Get Image
    err, res, img = sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_buffer)
    if err != 0:
        continue

    img = np.array(img, dtype=np.int8).astype(np.uint8)
    img = img.reshape((res[1], res[0], 3))
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    # inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    # action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    x, y, z, roll, pitch, yaw, gripper = [0.55, 0, 0.05, 0, 0, 0, 1]
    target_frame = build_target_frame(x, y, z, roll, pitch, yaw)
    q_init = [0.0] * len(sawyer_chain.links)
    
    # Solve IK
    print(f"Solving IK...")
    ik_solution = sawyer_chain.inverse_kinematics_frame(
        target=target_frame,
        initial_position=q_init
    )   
    print(ik_solution)
    
    # Set Joint Angles
    print(f"Acting...")
    for handle, angle in zip(joint_handles, ik_solution[1:8]):
        print(handle, angle)
        sim.simxSetJointTargetPosition(clientID, handle, float(angle), sim.simx_opmode_oneshot)

    # 任务完成判断（可选）
    # if condition_met:
    #     break
    break
