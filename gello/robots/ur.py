import socket
import threading
import time
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from gello.robots.robot import Robot

GRIPPER_PORT = 63352


class GripperHandler:
    """Robotiq 2F-85 gripper via direct socket (port 63352).

    Matches joysticktst.py GripperHandler — simple SET/GET commands,
    thread-safe with lock, local position tracking.
    """

    def __init__(self):
        self.socket = None
        self.lock = threading.Lock()
        self.connected = False
        self.current_pos = 0

    def connect(self, ip, port=GRIPPER_PORT):
        print(f"[GRIPPER] Connecting to {ip}:{port}...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((ip, port))
            self.connected = True
            try:
                self.socket.recv(1024)
            except Exception:
                pass
            print("[GRIPPER] Connected successfully.")
            return True
        except Exception as e:
            print(f"[GRIPPER] Failed to connect: {e}.")
            self.connected = False
            return False

    def _drain_response(self):
        """Read and discard any pending data from the gripper socket."""
        if self.socket is None:
            return
        try:
            self.socket.setblocking(False)
            while True:
                try:
                    self.socket.recv(1024)
                except BlockingIOError:
                    break
        except Exception:
            pass
        finally:
            self.socket.setblocking(True)

    def _send_cmd(self, cmd):
        if not self.connected or self.socket is None:
            return None
        try:
            with self.lock:
                self._drain_response()
                data = (cmd + "\n").encode("ascii")
                self.socket.sendall(data)
                time.sleep(0.01)
                self.socket.settimeout(0.1)
                try:
                    resp = self.socket.recv(1024).decode("ascii").strip()
                except (socket.timeout, OSError):
                    resp = None
                self.socket.settimeout(2.0)
                return resp
        except Exception as e:
            print(f"[GRIPPER] Comm error: {e}")
            return None

    def get_actual_pos(self):
        """Read the actual gripper position from the robot (0-255)."""
        if not self.connected:
            return self.current_pos
        resp = self._send_cmd("GET POS")
        if resp and resp.startswith("POS"):
            try:
                return int(resp.split()[1])
            except (IndexError, ValueError):
                pass
        return self.current_pos

    def activate(self):
        if not self.connected:
            return
        print("[GRIPPER] Activating...")
        self._send_cmd("SET ACT 0")
        time.sleep(0.5)
        self._send_cmd("SET ACT 1")
        time.sleep(2.0)
        self._send_cmd("SET GTO 1")
        time.sleep(0.3)
        self._send_cmd("SET SPE 255")
        self._send_cmd("SET FOR 255")
        self._send_cmd("GET POS")
        time.sleep(0.1)
        self._drain_response()
        print("[GRIPPER] Activation complete.")

    def set_speed(self, speed):
        """Set gripper finger speed (0-255). Lower = slower."""
        if not self.connected:
            return
        speed = max(0, min(255, int(speed)))
        self._send_cmd(f"SET SPE {speed}")

    def move(self, pos):
        if not self.connected:
            return
        self.current_pos = max(0, min(255, int(pos)))
        self._send_cmd(f"SET POS {self.current_pos}")

    def move_and_wait(self, pos, tolerance=5):
        """Move gripper and block until it reaches target or stalls on object."""
        if not self.connected:
            return
        target = max(0, min(255, int(pos)))
        self.move(target)
        last_pos = -1
        stall_count = 0
        while True:
            actual = self.get_actual_pos()
            if abs(actual - target) <= tolerance:
                break
            if actual == last_pos:
                stall_count += 1
                if stall_count >= 3:
                    break
            else:
                stall_count = 0
            last_pos = actual
            time.sleep(0.1)

    def stop(self):
        if self.socket:
            self.socket.close()


class URRobot(Robot):
    """UR5e robot with integrated Robotiq 2F-85 gripper.

    Uses the same direct-socket gripper control as joysticktst.py
    (GripperHandler) instead of the GELLO RobotiqGripper class.
    """

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        print(f"[URRobot] Connecting to {robot_ip}...")
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self._use_gripper = not no_gripper
        self._gripper_pos = 0  # Tracked locally (0-255)
        if self._use_gripper:
            self.gripper = GripperHandler()
            self.gripper.connect(robot_ip, GRIPPER_PORT)
            self.gripper.activate()

        self._free_drive = False
        self.robot.endFreedriveMode()
        print("[URRobot] Ready.")

    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        """Get gripper position as normalized 0-1 from hardware reading."""
        if self._use_gripper:
            return self.gripper.get_actual_pos() / 255.0
        return self._gripper_pos / 255.0

    def get_tcp_pose_raw(self) -> np.ndarray:
        """Get the current TCP pose as [x, y, z, rx, ry, rz] (UR rotation vector)."""
        return np.array(self.r_inter.getActualTCPPose())

    def move_joints(
        self,
        joints: list,
        speed: float = 0.5,
        accel: float = 0.3,
    ) -> None:
        """Move to joint positions via moveJ (blocking)."""
        self.robot.moveJ(list(joints[:6]), speed, accel)

    def move_linear(
        self,
        pose: np.ndarray,
        speed: float = 0.1,
        accel: float = 0.5,
        asynchronous: bool = False,
    ) -> None:
        """Move the TCP linearly to a target pose via moveL."""
        self.robot.moveL(list(pose), speed, accel, asynchronous)

    def move_linear_path(
        self,
        path: list,
        asynchronous: bool = False,
    ) -> None:
        """Move the TCP through waypoints with blending.

        Each element: [x, y, z, rx, ry, rz, speed, accel, blend_radius]
        """
        self.robot.moveL(path, asynchronous)

    def get_joint_state(self) -> np.ndarray:
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            pos = np.append(robot_joints, self._get_gripper_pos())
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the robot via servoJ (for GELLO/SpaceMouse agents)."""
        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(robot_joints, 0.5, 0.5, 1.0 / 500, 0.2, 100)
        if self._use_gripper:
            gripper_pos = int(joint_state[-1] * 255)
            self._gripper_pos = max(0, min(255, gripper_pos))
            self.gripper.move(self._gripper_pos)
        self.robot.waitPeriod(t_start)

    def command_cartesian_velocity(
        self,
        velocity: np.ndarray,
        acceleration: float = 0.5,
        time_running: float = 0.1,
        gripper_vel: float = 0.0,
    ) -> None:
        """Command TCP velocity via speedL.

        Gripper position tracked locally, matching joysticktst.py.
        """
        assert len(velocity) == 6, f"Expected 6D velocity, got {len(velocity)}"
        self.robot.speedL(list(velocity), acceleration, time=time_running)
        if self._use_gripper and abs(gripper_vel) > 0.001:
            self._gripper_pos = max(
                0, min(255, self._gripper_pos + int(gripper_vel * 255))
            )
            self.gripper.move(self._gripper_pos)

    def set_gripper(self, pos: int) -> None:
        """Set gripper position (0-255) directly via socket, no RTDE."""
        if self._use_gripper:
            self._gripper_pos = max(0, min(255, int(pos)))
            self.gripper.move(self._gripper_pos)

    def set_gripper_speed(self, speed: int) -> None:
        """Set gripper finger speed (0-255). Lower = slower."""
        if self._use_gripper:
            self.gripper.set_speed(max(0, min(255, int(speed))))

    def get_actual_gripper_pos(self) -> int:
        """Read actual gripper position (0-255) from hardware via GET POS.

        Unlike _get_gripper_pos() which returns the locally tracked value,
        this reads the real position from the gripper. Use for grasp
        verification where commanded vs actual position matters.
        """
        if self._use_gripper:
            return self.gripper.get_actual_pos()
        return self._gripper_pos

    def speed_stop(self) -> None:
        """Stop speedL motion immediately."""
        self.robot.speedStop()

    def stop_linear(self) -> None:
        """Stop moveL motion (clears async moveL state)."""
        self.robot.stopL()

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        gripper_pos = np.array([joints[-1]])

        tcp_raw = self.r_inter.getActualTCPPose()
        pos = np.array(tcp_raw[:3])
        quat = Rotation.from_rotvec(tcp_raw[3:6]).as_quat()
        pos_quat = np.concatenate([pos, quat])

        joint_vel = np.array(self.r_inter.getActualQd())

        return {
            "joint_positions": joints,
            "joint_velocities": joint_vel,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot_ip = "10.125.144.209"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
