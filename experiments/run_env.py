# experiments/run_env.py
"""Main control loop (T4) for UR5e teleoperation with data collection.

Supports agents: gello, joystick, spacemouse, dummy/none
Connects to: T1 (robot), T2 (cameras), T3 (agent server, gello only)
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    try:
        import termcolor

        if len(args) > 0:
            args = tuple(
                termcolor.colored(arg, color=color, attrs=attrs) for arg in args
            )
    except ImportError:
        pass
    print(*args, **kwargs)


def angdiff(a, b):
    """Compute angular difference wrapped to [-pi, pi)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Args:
    agent: str = "gello"
    robot_port: int = 6001
    gello_server_port: int = 6000  # T3 agent server port (gello only)
    wrist_camera_port: int = 5000  # T2 camera PUB port (RealSense)
    base_camera_port: int = 5001  # T2 camera PUB port (OAK-D)
    hostname: str = "127.0.0.1"
    robot_type: str = "ur5"  # for spacemouse IK model
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    verbose: bool = False
    no_cameras: bool = False  # skip camera connections

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args):
    # -----------------------------------------------------------------
    # 1. Connect to robot and cameras
    # -----------------------------------------------------------------
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

        if args.no_cameras:
            camera_clients = {}
        else:
            # Both cameras at 1280x720
            camera_clients = {
                "wrist": ZMQClientCamera(
                    port=args.wrist_camera_port,
                    host=args.hostname,
                    camera_name="wrist",
                    dummy_shape_rgb=(720, 1280, 3),
                    dummy_shape_depth=(720, 1280, 1),
                ),
                "base": ZMQClientCamera(
                    port=args.base_camera_port,
                    host=args.hostname,
                    camera_name="base",
                    dummy_shape_rgb=(720, 1280, 3),
                    dummy_shape_depth=(720, 1280, 1),
                ),
            }

    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
    )

    # -----------------------------------------------------------------
    # 2. Create agent
    # -----------------------------------------------------------------
    if args.agent == "gello":
        # Gello runs as a remote ZMQ server (T3)
        agent_cfg = {
            "_target_": "gello.agents.zmq_agent.ZMQAgent",
            "host": args.hostname,
            "port": args.gello_server_port,
            "num_dofs": 7,  # UR5e: 6 joints + 1 gripper
        }
    elif args.agent == "joystick":
        agent_cfg = {
            "_target_": "gello.agents.joystick_agent.JoystickAgent",
            "robot_type": args.robot_type,
            "num_dofs": 7,
            "verbose": args.verbose,
        }
    elif args.agent == "spacemouse":
        agent_cfg = {
            "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
            "robot_type": args.robot_type,
            "verbose": args.verbose,
        }
    elif args.agent in ("dummy", "none"):
        agent_cfg = {
            "_target_": "gello.agents.agent.DummyAgent",
            "num_dofs": robot_client.num_dofs(),
        }
    else:
        raise ValueError(
            f"Unknown agent '{args.agent}'. "
            f"Choose from: gello, joystick, spacemouse, dummy, none"
        )

    agent = instantiate_from_dict(agent_cfg)

    # -----------------------------------------------------------------
    # 3. Move to start position (only for gello -- others start in-place)
    # -----------------------------------------------------------------
    if args.agent == "gello":
        # Default UR5e reset pose (6 joints + 1 gripper)
        if args.start_joints is None:
            reset_joints = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        else:
            reset_joints = np.array(args.start_joints)

        curr_joints = env.get_obs()["joint_positions"]

        if reset_joints.shape == curr_joints.shape:
            max_delta = np.abs(curr_joints - reset_joints).max()
            steps = min(int(max_delta / 0.01), 100)
            if steps > 0:
                print("Moving to start position...")
                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
                print("Done.")
        else:
            print(
                f"ERROR: start_joints shape {reset_joints.shape} != "
                f"robot joint shape {curr_joints.shape}. "
                f"Check --start-joints matches your robot DOF."
            )
            return

        # Align gello leader to follower
        print("Aligning gello to robot...")
        start_pos = agent.act(env.get_obs())
        obs = env.get_obs()
        joints = obs["joint_positions"]

        abs_deltas = np.abs(angdiff(start_pos, joints))
        id_max = np.argmax(abs_deltas)

        if abs_deltas[id_max] > 0.8:
            print("Large delta detected, soft-aligning...")
            current = joints.copy()
            target = start_pos
            steps = min(
                int(np.abs(angdiff(current, target)).max() / 0.01),
                300,
            )
            for jnt in np.linspace(current, target, steps):
                env.step(jnt)
                time.sleep(0.002)

        assert len(start_pos) == len(joints), (
            f"Agent output dim = {len(start_pos)}, " f"but env dim = {len(joints)}"
        )

        # Fine alignment
        max_delta = 0.05
        for _ in range(25):
            obs = env.get_obs()
            command_joints = agent.act(obs)
            current_joints = obs["joint_positions"]
            delta = angdiff(command_joints, current_joints)
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            env.step(current_joints + delta)
        print("Alignment complete. Gello control active.")

        # Safety check
        obs = env.get_obs()
        joints = obs["joint_positions"]
        action = agent.act(obs)
        if (action - joints > 0.5).any():
            print("WARNING: Action jump too large after alignment!")
            joint_index = np.where(action - joints > 0.8)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, "
                    f"follower: {joints[j]}, "
                    f"diff: {action[j] - joints[j]}"
                )
            exit()

    # -----------------------------------------------------------------
    # 4. Run control loop
    # -----------------------------------------------------------------
    from gello.utils.control_utils import (
        SaveInterface,
        run_control_loop,
    )

    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=args.data_dir,
            agent_name=args.agent,
            expand_user=True,
        )

    run_control_loop(env, agent, save_interface, use_colors=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
