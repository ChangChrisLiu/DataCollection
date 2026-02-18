import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq

from gello.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        print(f"Robot Server Binding to {addr}, Robot: {robot}")
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "command_cartesian_velocity":
                    result = self._robot.command_cartesian_velocity(**args)
                elif method == "speed_stop":
                    result = self._robot.speed_stop()
                elif method == "stop_linear":
                    result = self._robot.stop_linear()
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "get_tcp_pose_raw":
                    result = self._robot.get_tcp_pose_raw()
                elif method == "move_joints":
                    result = self._robot.move_joints(**args)
                elif method == "move_linear":
                    result = self._robot.move_linear(**args)
                elif method == "move_linear_path":
                    result = self._robot.move_linear_path(**args)
                elif method == "set_gripper":
                    result = self._robot.set_gripper(**args)
                elif method == "set_gripper_speed":
                    result = self._robot.set_gripper_speed(**args)
                elif method == "get_actual_gripper_pos":
                    result = self._robot.get_actual_gripper_pos()
                elif method == "set_freedrive_mode":
                    result = self._robot.set_freedrive_mode(**args)
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                # Timeout occurred - don't spam the console
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQObsServerRobot:
    """Read-only ZMQ server for observation polling during skill execution.

    This server wraps the same robot instance as ZMQServerRobot but only
    dispatches read-only methods. It runs on a separate port (e.g. 6002)
    so that observation queries remain responsive even when the primary
    server is blocked on a moveL call.

    Thread safety: This server only calls RTDEReceiveInterface methods
    (get_observations, get_joint_state, get_tcp_pose_raw) while the
    primary server calls RTDEControlInterface methods (moveL, servoJ).
    These are independent C++ objects with separate TCP connections,
    so no locking is needed.
    """

    def __init__(
        self,
        robot: Robot,
        port: int = 6002,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        print(f"Obs Server (read-only) Binding to {addr}")
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve read-only robot observations over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                method = request.get("method")
                result: Any

                # Only allow read-only methods
                if method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "get_tcp_pose_raw":
                    result = self._robot.get_tcp_pose_raw()
                elif method == "get_actual_gripper_pos":
                    result = self._robot.get_actual_gripper_pos()
                elif method == "num_dofs":
                    result = self._robot.num_dofs()
                else:
                    result = {
                        "error": f"Method '{method}' not allowed "
                        f"on read-only obs server"
                    }

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def command_cartesian_velocity(
        self,
        velocity: np.ndarray,
        acceleration: float = 0.5,
        time_running: float = 0.1,
        gripper_vel: float = 0.0,
    ) -> None:
        """Command TCP velocity via speedL through ZMQ."""
        request = {
            "method": "command_cartesian_velocity",
            "args": {
                "velocity": velocity,
                "acceleration": acceleration,
                "time_running": time_running,
                "gripper_vel": gripper_vel,
            },
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def speed_stop(self) -> None:
        """Stop speedL motion through ZMQ."""
        request = {"method": "speed_stop"}
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def stop_linear(self) -> None:
        """Stop moveL motion (clears async moveL state) through ZMQ."""
        request = {"method": "stop_linear"}
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def get_tcp_pose_raw(self) -> np.ndarray:
        """Get current TCP pose [x,y,z,rx,ry,rz] through ZMQ.

        Returns:
            np.ndarray: (6,) TCP pose in base frame (UR rotation vector format).
        """
        request = {"method": "get_tcp_pose_raw"}
        self._socket.send(pickle.dumps(request))
        result = pickle.loads(self._socket.recv())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def move_joints(
        self,
        joints: list,
        speed: float = 0.5,
        accel: float = 0.3,
    ) -> None:
        """Move to joint positions via moveJ (blocking) through ZMQ."""
        request = {
            "method": "move_joints",
            "args": {"joints": joints, "speed": speed, "accel": accel},
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def move_linear(
        self,
        pose: np.ndarray,
        speed: float = 0.1,
        accel: float = 0.5,
        asynchronous: bool = False,
    ) -> None:
        """Move TCP linearly to target pose through ZMQ.

        When asynchronous=False (default), blocks until moveL completes.
        When asynchronous=True, returns immediately — the robot moves in
        the background. Use speed_stop() to halt the motion.

        Args:
            pose: (6,) target [x,y,z,rx,ry,rz] in base frame.
            speed: TCP speed in m/s.
            accel: TCP acceleration in m/s^2.
            asynchronous: If True, return immediately (non-blocking).
        """
        request = {
            "method": "move_linear",
            "args": {
                "pose": pose,
                "speed": speed,
                "accel": accel,
                "asynchronous": asynchronous,
            },
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def move_linear_path(
        self,
        path: list,
        asynchronous: bool = False,
    ) -> None:
        """Move TCP through blended waypoints via ZMQ.

        Each element in path is [x,y,z,rx,ry,rz, speed, accel, blend_radius].

        Args:
            path: List of 9-element waypoint lists.
            asynchronous: If True, return immediately (non-blocking).
        """
        request = {
            "method": "move_linear_path",
            "args": {
                "path": path,
                "asynchronous": asynchronous,
            },
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def set_gripper(self, pos: int) -> None:
        """Set gripper position (0-255) directly via socket, no RTDE."""
        request = {
            "method": "set_gripper",
            "args": {"pos": pos},
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def set_gripper_speed(self, speed: int) -> None:
        """Set gripper finger speed (0-255) via ZMQ."""
        request = {
            "method": "set_gripper_speed",
            "args": {"speed": speed},
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def get_actual_gripper_pos(self) -> int:
        """Read actual gripper position (0-255) from hardware via GET POS."""
        request = {"method": "get_actual_gripper_pos"}
        self._socket.send(pickle.dumps(request))
        result = pickle.loads(self._socket.recv())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def set_freedrive_mode(self, enable: bool) -> None:
        """Enable or disable freedrive mode through ZMQ.

        Args:
            enable: True to enable freedrive, False to disable.
        """
        request = {
            "method": "set_freedrive_mode",
            "args": {"enable": enable},
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        self._socket.close()
        self._context.term()
