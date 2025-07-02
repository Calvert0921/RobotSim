import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_7_to_6(state7):
    # split position + quaternion
    pos, quat = state7[0][:3], state7[0][3:]

    # quaternion → Euler (roll, pitch, yaw) in radians
    r = R.from_quat(quat)
    roll, pitch, yaw = r.as_euler("xyz", degrees=False)
    
    # concatenate into a (6,) vector
    state6 = np.concatenate([pos, [roll, pitch, yaw]], axis=0)  # → shape (6,)
    
    return state6

def convert_6_to_7(action6):
    # 1) Get your 6-dim SmolVLA action:
    #    action6 is a torch.Tensor of shape (1, 6) = [dx, dy, dz, roll, pitch, yaw]

    # 2) Split into position deltas and Euler angles
    action6_np = action6.cpu().numpy()     # → shape (1,6)
    pos_delta = action6_np[:, :3]          # → shape (1,3)
    eulers   = action6_np[:, 3:]           # → shape (1,3), in radians

    # 3) Convert each (roll, pitch, yaw) to a quaternion (x,y,z,w)
    #    Note: roll = rotation about X, pitch about Y, yaw about Z
    r = R.from_euler('XYZ', eulers, degrees=False)  
    # as_quat() returns [..., x, y, z, w]
    quats = r.as_quat()                     # → shape (1,4)

    # 4) Stack back into a (1,7) array
    action7 = np.concatenate([pos_delta, quats], axis=1)  # → shape (1,7)
    
    return action7