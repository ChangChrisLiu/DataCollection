# experiments/launch_nodes.py
"""Robot server (T1) — launches a ZMQ server wrapping a robot driver.

Supported robots: ur, sim_ur, panda, sim_panda, bimanual_ur, print/none
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "ur"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "10.125.145.89"


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

        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
