# experiments/launch_nodes.py
"""Robot server (T1) — launches ZMQ servers wrapping a robot driver.

Launches two servers on real robots:
  - Port 6001: Full control server (servoJ, speedL, moveL, speed_stop, etc.)
  - Port 6002: Read-only observation server (get_observations, get_tcp_pose_raw)

The dual-port architecture allows observation polling during blocking moveL
calls for skill execution. Both servers wrap the same robot instance but
access different RTDE interfaces (Control vs Receive), which is thread-safe.

Supported robots: ur, sim_ur, panda, sim_panda, bimanual_ur, print/none
"""

import threading
from dataclasses import dataclass
from pathlib import Path

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQObsServerRobot, ZMQServerRobot


@dataclass
class Args:
    robot: str = "ur"
    robot_port: int = 6001
    obs_port: int = 6002  # Read-only observation server port
    hostname: str = "127.0.0.1"
    robot_ip: str = "10.125.144.209"


def launch_robot_server(args: Args):
    port = args.robot_port

    # ------------------------------------------------------------------
    # Simulation robots (MuJoCo)
    # ------------------------------------------------------------------
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()

    elif args.robot == "sim_panda":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=None, port=port, host=args.hostname
        )
        server.serve()

    # ------------------------------------------------------------------
    # Real robots
    # ------------------------------------------------------------------
    else:
        if args.robot == "ur":
            from gello.robots.ur import URRobot

            robot = URRobot(robot_ip=args.robot_ip)

        elif args.robot == "panda":
            from gello.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=args.robot_ip)

        elif args.robot == "bimanual_ur":
            from gello.robots.ur import URRobot

            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)

        elif args.robot in ("none", "print"):
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot '{args.robot}' not implemented. "
                f"Choose from: sim_ur, sim_panda, ur, panda, bimanual_ur, none"
            )

        # Launch primary control server (port 6001)
        server = ZMQServerRobot(robot, port=port, host=args.hostname)

        # Launch read-only observation server (port 6002)
        obs_server = ZMQObsServerRobot(robot, port=args.obs_port, host=args.hostname)
        obs_thread = threading.Thread(target=obs_server.serve, daemon=True)
        obs_thread.start()
        print(
            f"Observation server started on port {args.obs_port} "
            f"(read-only, for skill execution)"
        )

        print(f"Starting robot control server on port {port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
