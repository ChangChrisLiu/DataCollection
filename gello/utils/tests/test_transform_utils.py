# gello/utils/tests/test_transform_utils.py
"""Tests for SE(3) coordinate transformation utilities."""

import numpy as np
from scipy.spatial.transform import Rotation

from gello.utils.transform_utils import (
    compute_relative_waypoints,
    homogeneous_to_pose6d,
    matrix_to_rotvec,
    pose6d_to_homogeneous,
    pose6d_to_pos_quat,
    quat_to_rotvec,
    rotvec_to_matrix,
    rotvec_to_quat,
    transform_waypoints_to_base,
)


class TestRotvecConversions:
    def test_rotvec_to_matrix_identity(self):
        R = rotvec_to_matrix(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotvec_to_matrix_90deg_z(self):
        rotvec = np.array([0, 0, np.pi / 2])
        R = rotvec_to_matrix(rotvec)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotvec_roundtrip(self):
        rotvec = np.array([0.3, -0.5, 1.2])
        R = rotvec_to_matrix(rotvec)
        recovered = matrix_to_rotvec(R)
        np.testing.assert_allclose(recovered, rotvec, atol=1e-10)

    def test_rotvec_to_quat_identity(self):
        quat = rotvec_to_quat(np.zeros(3))
        # Identity quaternion: [0, 0, 0, 1]
        np.testing.assert_allclose(quat, [0, 0, 0, 1], atol=1e-10)

    def test_quat_rotvec_roundtrip(self):
        rotvec = np.array([0.1, -0.2, 0.7])
        quat = rotvec_to_quat(rotvec)
        recovered = quat_to_rotvec(quat)
        np.testing.assert_allclose(recovered, rotvec, atol=1e-10)


class TestHomogeneousConversions:
    def test_identity_pose(self):
        pose = np.zeros(6)
        T = pose6d_to_homogeneous(pose)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_translation_only(self):
        pose = np.array([1.0, 2.0, 3.0, 0, 0, 0])
        T = pose6d_to_homogeneous(pose)
        assert T[0, 3] == 1.0
        assert T[1, 3] == 2.0
        assert T[2, 3] == 3.0
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)

    def test_roundtrip(self):
        pose = np.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.5])
        T = pose6d_to_homogeneous(pose)
        recovered = homogeneous_to_pose6d(T)
        np.testing.assert_allclose(recovered, pose, atol=1e-10)

    def test_homogeneous_is_valid_se3(self):
        pose = np.array([1.0, 2.0, 3.0, 0.5, 0.3, -0.7])
        T = pose6d_to_homogeneous(pose)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)
        # R should be orthogonal: R @ R^T = I
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # det(R) = 1
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestPoseToQuat:
    def test_identity(self):
        pose = np.zeros(6)
        pos_quat = pose6d_to_pos_quat(pose)
        assert pos_quat.shape == (7,)
        np.testing.assert_allclose(pos_quat[:3], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(pos_quat[3:], [0, 0, 0, 1], atol=1e-10)

    def test_with_translation_and_rotation(self):
        pose = np.array([1.0, 2.0, 3.0, 0, 0, np.pi / 2])
        pos_quat = pose6d_to_pos_quat(pose)
        assert pos_quat.shape == (7,)
        np.testing.assert_allclose(pos_quat[:3], [1.0, 2.0, 3.0], atol=1e-10)
        # 90deg around Z: quat should be [0, 0, sin(45), cos(45)]
        expected_quat = Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat()
        np.testing.assert_allclose(pos_quat[3:], expected_quat, atol=1e-10)


class TestRelativeWaypoints:
    def test_empty_input(self):
        result = compute_relative_waypoints([])
        assert result == []

    def test_single_pose(self):
        result = compute_relative_waypoints([np.zeros(6)])
        assert result == []

    def test_identity_relative(self):
        """Two identical poses should give identity relative transform."""
        pose = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        result = compute_relative_waypoints([pose, pose])
        assert len(result) == 1
        np.testing.assert_allclose(result[0], np.eye(4), atol=1e-10)

    def test_pure_translation(self):
        """Relative waypoint for pure translation along X."""
        p1 = np.array([0, 0, 0, 0, 0, 0])
        p2 = np.array([1, 0, 0, 0, 0, 0])
        result = compute_relative_waypoints([p1, p2])
        assert len(result) == 1
        # Should encode 1m translation along X with no rotation
        np.testing.assert_allclose(result[0][:3, 3], [1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(result[0][:3, :3], np.eye(3), atol=1e-10)

    def test_roundtrip_with_transform_to_base(self):
        """Record absolute poses -> relative -> replay from same start = same result."""
        poses = [
            np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0]),
            np.array([0.6, 0.3, 0.1, 0.0, 0.0, 0.1]),
            np.array([0.7, 0.4, 0.2, 0.1, 0.0, 0.2]),
        ]
        relative = compute_relative_waypoints(poses)
        T_start = pose6d_to_homogeneous(poses[0])
        replayed = transform_waypoints_to_base(relative, T_start)

        assert len(replayed) == 2
        for replayed_pose, original_pose in zip(replayed, poses[1:]):
            np.testing.assert_allclose(replayed_pose, original_pose, atol=1e-10)

    def test_replay_from_different_start(self):
        """Relative waypoints replayed from different start produce valid SE(3)."""
        poses = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        relative = compute_relative_waypoints(poses)

        # Replay from a different starting pose
        new_start = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1])
        T_new_start = pose6d_to_homogeneous(new_start)
        replayed = transform_waypoints_to_base(relative, T_new_start)

        assert len(replayed) == 1
        # The replayed pose should be a valid 6D pose
        T_check = pose6d_to_homogeneous(replayed[0])
        R = T_check[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
