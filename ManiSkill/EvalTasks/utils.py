import numpy as np

def get_xyz_from_obs(raw_obs, color: str):
    pose = raw_obs["extra"][f"{color}_cube_pose"]  # [B,7] torch tensor
    return pose[0, :3]  # xyz of the first (and only) env

def tcp_xyz(raw_obs):
    # [B,7]: x y z qx qy qz qw
    return raw_obs["extra"]["tcp_pose"][0, :3]

def cube_xyz(raw_obs, color):
    return raw_obs["extra"][f"{color}_cube_pose"][0, :3]

def intent_scores_from_traj(tcp_traj, cubes_traj, alpha=8.0):
    """
    tcp_traj: list of np.array shape (3,)
    cubes_traj: dict color -> list of np.array shape (3,)
    Returns: dict color -> float score
    """
    eps = 1e-8
    scores = {c: 0.0 for c in cubes_traj.keys()}
    for t in range(1, len(tcp_traj)):
        v = tcp_traj[t] - tcp_traj[t-1]                       # (3,)
        v_norm = np.linalg.norm(v) + eps
        for c in cubes_traj.keys():
            r = cubes_traj[c][t] - tcp_traj[t]                # vector from tcp to cube
            r_norm = np.linalg.norm(r) + eps
            cos = float(np.dot(v, r) / (v_norm * r_norm))
            cos = max(0.0, cos)                               # only motion TOWARD the cube
            d = r_norm
            w = np.exp(-alpha * d)                            # distance weight
            scores[c] += w * cos
    return scores

def softmax_probs(scores):
    xs = np.array(list(scores.values()), dtype=np.float64)
    xs = xs - xs.max()
    ex = np.exp(xs)
    p = ex / (ex.sum() + 1e-12)
    return {c: float(p[i]) for i, c in enumerate(scores.keys())}
