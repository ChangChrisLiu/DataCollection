# gello/utils/transform_utils.py
"""SE(3) coordinate transformation utilities for UR robot skill execution.

Uses scipy.spatial.transform.Rotation for rotation vector <-> matrix <-> quaternion
conversions. UR robots natively use rotation vectors [rx, ry, rz] (axis-angle where
the magnitude encodes the angle in radians).

Key operation for skill replay:
    T_target^base = T_current^base @ T_step^local

where T_step^local is a relative waypoint recorded in the gripper frame.
"""

from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector [rx, ry, rz] to a 3x3 rotation matrix.

    Args:
        rotvec: (3,) rotation vector (axis-angle, magnitude = angle in rad).

    Returns:
        (3, 3) rotation matrix.
    """
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector [rx, ry, rz].

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) rotation vector.
    """
    return Rotation.from_matrix(R).as_rotvec()


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a quaternion [qx, qy, qz, qw].

    Args:
        rotvec: (3,) rotation vector.

    Returns:
        (4,) quaternion in [qx, qy, qz, qw] order (scipy convention).
    """
    return Rotation.from_rotvec(rotvec).as_quat()


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [qx, qy, qz, qw] to a rotation vector.

    Args:
        quat: (4,) quaternion in [qx, qy, qz, qw] order.

    Returns:
        (3,) rotation vector.
    """
    return Rotation.from_quat(quat).as_rotvec()


def pose6d_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    """Convert a UR-style 6D pose [x, y, z, rx, ry, rz] to a 4x4 SE(3) matrix.

    Args:
        pose: (6,) array [x, y, z, rx, ry, rz] where rx/ry/rz is a rotation vector.

    Returns:
        (4, 4) homogeneous transformation matrix.
    """
    pose = np.asarray(pose, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def homogeneous_to_pose6d(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 SE(3) matrix to a UR-style 6D pose [x, y, z, rx, ry, rz].

    Args:
        T: (4, 4) homogeneous transformation matrix.

    Returns:
        (6,) array [x, y, z, rx, ry, rz].
    """
    pos = T[:3, 3]
    rotvec = matrix_to_rotvec(T[:3, :3])
    return np.concatenate([pos, rotvec])


def align_rotvec(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Ensure a rotation vector is on the same side of the pi boundary as a reference.

    Near rotation angle pi, scipy as_rotvec() can flip the axis direction,
    producing a rotation vector that represents the same orientation but causes
    UR moveL path blending to interpolate the long way around.

    If the target and reference point in opposite directions (dot product < 0),
    flip the target to the equivalent representation: angle' = 2*pi - angle,
    axis' = -axis.

    Args:
        target: (3,) rotation vector to align.
        reference: (3,) reference rotation vector.

    Returns:
        (3,) aligned rotation vector (same rotation, consistent direction).
    """
    if np.dot(target, reference) < 0:
        angle = np.linalg.norm(target)
        if angle < 1e-10:
            return target
        axis = target / angle
        return -axis * (2 * np.pi - angle)
    return target


def align_pose6d_rotvecs(
    poses: List[np.ndarray],
    reference_rotvec: np.ndarray,
) -> List[np.ndarray]:
    """Align rotation vectors in a sequence of 6D poses to a reference direction.

    Prevents UR moveL path blending issues caused by rotation vector flips
    near the pi boundary.

    Args:
        poses: List of (6,) poses [x, y, z, rx, ry, rz].
        reference_rotvec: (3,) reference rotation vector direction.

    Returns:
        List of (6,) poses with aligned rotation vectors.
    """
    aligned = []
    for pose in poses:
        rv = align_rotvec(pose[3:6], reference_rotvec)
        aligned.append(np.concatenate([pose[:3], rv]))
    return aligned


def pose6d_to_pos_quat(pose: np.ndarray) -> np.ndarray:
    """Convert a UR 6D pose to [x, y, z, qx, qy, qz, qw].

    Args:
        pose: (6,) array [x, y, z, rx, ry, rz].

    Returns:
        (7,) array [x, y, z, qx, qy, qz, qw].
    """
    pos = pose[:3]
    quat = rotvec_to_quat(pose[3:6])
    return np.concatenate([pos, quat])


def compute_relative_waypoints(
    poses_abs: List[np.ndarray],
) -> List[np.ndarray]:
    """Convert a sequence of absolute 6D poses to relative 4x4 transforms.

    For consecutive poses P_i and P_{i+1}:
        T_step_local = inv(T_i) @ T_{i+1}

    This gives the relative motion in the frame of the previous pose,
    suitable for replaying from any starting pose.

    Args:
        poses_abs: List of (6,) absolute poses [x, y, z, rx, ry, rz].

    Returns:
        List of (4, 4) relative homogeneous transforms. Length = len(poses_abs) - 1.
    """
    if len(poses_abs) < 2:
        return []

    relative_waypoints = []
    T_prev = pose6d_to_homogeneous(poses_abs[0])

    for i in range(1, len(poses_abs)):
        T_curr = pose6d_to_homogeneous(poses_abs[i])
        T_step = np.linalg.inv(T_prev) @ T_curr
        relative_waypoints.append(T_step)
        T_prev = T_curr

    return relative_waypoints


def transform_waypoints_to_base(
    waypoints_local: List[np.ndarray],
    T_current_base: np.ndarray,
) -> List[np.ndarray]:
    """Transform relative (local-frame) waypoints to base-frame 6D poses.

    For each relative waypoint T_step_local:
        T_target_base = T_current_base @ T_step_local

    Then T_current_base is updated to T_target_base for the next step.

    Args:
        waypoints_local: List of (4, 4) relative transforms in gripper-local frame.
        T_current_base: (4, 4) current TCP pose in base frame.

    Returns:
        List of (6,) target poses [x, y, z, rx, ry, rz] in base frame.
    """
    target_poses = []
    T_current = T_current_base.copy()

    for T_step in waypoints_local:
        T_target = T_current @ T_step
        target_poses.append(homogeneous_to_pose6d(T_target))
        T_current = T_target

    return target_poses
