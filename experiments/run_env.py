# experiments/run_env.py (最终、完整、7-DOF 异步版)
import glob
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.robot_node import ZMQClientRobot
# [修改] 导入我们新的异步 SUB 客户端
from gello.zmq_core.camera_node import ZMQClientCamera
# [新] 导入我们新的 Gello REQ 客户端
from gello.agents.zmq_agent import ZMQAgent


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def angdiff(a, b):
    import numpy as np
    d = a - b
    # 把差值映射到 [-pi, pi)
    return (d + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    gello_server_port: int = 6000  # <--- [新] T3 Gello 服务器的端口
    wrist_camera_port: int = 5000  # <--- T2 相机服务器
    base_camera_port: int = 5001   # <--- T2 相机服务器
    hostname: str = "127.0.0.1"    # <--- 确保所有终端都使用 127.0.0.1
    robot_type: str = None
    hz: int = 100                  # <--- 保持 100Hz!
    start_joints: Optional[Tuple[float, ...]] = None

    # gello_port: Optional[str] = None # <--- [删除] 不再需要，由 T3 处理
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        # [关键修复]
        # RealSense RGB = 424x240, RealSense Depth = 424x240 (对齐后)
        rs_rgb_shape = (240, 424, 3) 
        rs_dep_shape = (240, 424, 1) # <--- [修改] 
        # OAK-D RGB = 416x240, OAK-D Depth = 416x240 (对齐后)
        oak_rgb_shape = (240, 416, 3) # <--- [修改] 
        oak_dep_shape = (240, 416, 1) # <--- [修改] 


        camera_clients = {
            "wrist": ZMQClientCamera(
                port=args.wrist_camera_port, host=args.hostname, camera_name="wrist",
                dummy_shape_rgb=rs_rgb_shape, # 424
                dummy_shape_depth=rs_dep_shape # 424
            ),
            "base": ZMQClientCamera(
                port=args.base_camera_port, host=args.hostname, camera_name="base",
                dummy_shape_rgb=oak_rgb_shape, # 416
                dummy_shape_depth=oak_dep_shape # 416
            ),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)



    agent_cfg = {}
    if args.bimanual:
        # (Bimanual 逻辑保持不变)
        if args.agent == "gello":
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": left,
                },
                "agent_right": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": right,
                },
            }
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "l",
                },
                "agent_right": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "r",
                },
            }
        elif args.agent == "spacemouse":
            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": left_path,
                    "verbose": args.verbose,
                },
                "agent_right": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": right_path,
                    "verbose": args.verbose,
                    "invert_button": True,
                },
            }
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        # <--- [重大修改] ---
        if args.agent == "gello":
            # Gello 现在是一个 ZMQ 客户端
            agent_cfg = {
                "_target_": "gello.agents.zmq_agent.ZMQAgent",
                "host": args.hostname,
                "port": args.gello_server_port,
                "num_dofs": 7 # <--- [修复] 必须是 7-DOF
            }
            
            if args.start_joints is None:
                # [修复] 默认的 reset_joints 也是 7-DOF
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0] 
                )
            else:
                reset_joints = np.array(args.start_joints)

            # 这里的 get_obs() 是 100Hz 非阻塞的
            curr_joints = env.get_obs()["joint_positions"]
            
            # [修复] 现在 shape 应该匹配了 (7 == 7)
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)
                print("正在执行启动程序：移动到 start_joints...")
                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
                print("✅ 移动完成。")
            else:
                print(f"❌ 致命错误: start_joints 形状 ({reset_joints.shape}) 与机器人 ({curr_joints.shape}) 不匹配。")
                print(f"   请检查你的 --start-joints 命令是否为 7 个值 (你的是 {len(args.start_joints)})。")
                print(f"   并确保你的 T1 机器人 ({args.robot}) 确实是 7-DOF。")
                return # 退出
        # <--- [修改结束] ---
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                "robot_type": args.robot_type,
                "which_hand": "l",
            }
        elif args.agent == "spacemouse":
            agent_cfg = {
                "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                "robot_type": args.robot_type,
                "verbose": args.verbose,
            }
        elif args.agent == "dummy" or args.agent == "none":
            agent_cfg = {
                "_target_": "gello.agents.agent.DummyAgent",
                "num_dofs": robot_client.num_dofs(),
            }
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    agent = instantiate_from_dict(agent_cfg)
    
    # (自动对齐阶段 2 和 3 保持不变)
    print("正在对齐 Gello 和机器人...")
    start_pos = agent.act(env.get_obs()) # 第一次调用 T3
    obs = env.get_obs()
    joints = obs["joint_positions"]
    print("Debug current joints", joints)
    print("Debug start joints", start_pos)

    abs_deltas = np.abs(angdiff(start_pos, joints))
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        print("\n检测到巨大差异；正在软对齐到 Gello 初始位置...")
        current = joints.copy()
        target  = start_pos
        steps = min(int(np.abs(angdiff(current, target)).max() / 0.01), 300)
        for jnt in np.linspace(current, target, steps):
            env.step(jnt)
            time.sleep(0.002)

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

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
    print("✅ 对齐完成。Gello 控制已激活。")

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    from gello.utils.control_utils import SaveInterface, run_control_loop

    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=args.data_dir, agent_name=args.agent, expand_user=True
        )

    run_control_loop(env, agent, save_interface, use_colors=True)


if __name__ == "__main__":
    main(tyro.cli(Args))


