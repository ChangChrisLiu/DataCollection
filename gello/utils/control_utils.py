# gello/utils/control_utils.py (最终版 - 带失败标记)
"""Shared utilities for robot control loops."""

import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional
import pickle # <--- 需要导入 pickle
import os # <--- 需要导入 os (虽然 pathlib.rename 更好)

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv # (确保你的 env.py 是我们修改过的最终版)
# [新] 导入 KBReset
from gello.data_utils.keyboard_interface import KBReset
# [删除] 不再需要导入 save_frame


DEFAULT_MAX_JOINT_DELTA = 1.0


# (move_to_start_position 函数保持不变)
def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually."""
    print("Going to start position")
    start_pos = agent.act(env.get_obs()) 
    obs = env.get_obs()
    joints = obs["joint_positions"]
    try: 
        from gello.utils.math_utils import angdiff
        abs_deltas = np.abs(angdiff(start_pos, joints))
    except ImportError:
        print("警告: 未找到 gello.utils.math_utils.angdiff, 使用简单差值。")
        abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)
    max_joint_delta = DEFAULT_MAX_JOINT_DELTA 
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print("\n错误: 目标起始位置与当前位置相差过大！")
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(ids, abs_deltas[id_mask], start_pos[id_mask], joints[id_mask]):
            print(f"joint[{i}]: \t delta: {delta:4.3f} > {max_joint_delta:.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}")
        return False 
    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(joints), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        try:
            from gello.utils.math_utils import angdiff
            delta = angdiff(command_joints, current_joints)
        except ImportError: delta = command_joints - current_joints
        max_joint_delta_step = np.abs(delta).max()
        if max_joint_delta_step > max_delta: delta = delta / max_joint_delta_step * max_delta
        env.step(current_joints + delta)
    return True 


class SaveInterface:
    """
    [修改] 处理键盘数据保存接口。
    只在检测到新的相机帧时才保存 (大约 30Hz)。
    停止录制后询问用户是否标记为失败。
    """
    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        """Initialize save interface."""
        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None
        self.last_saved_wrist_ts = 0.0
        self.last_saved_base_ts = 0.0
        self.frame_count = 0

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording (saves at ~30Hz on new camera frames)")
        print("  Q: Stop recording (then prompts G/N)") # <--- 更新提示

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """
        [修改] 更新保存接口。
        停止时询问用户并可能重命名文件夹。
        """
        dt = datetime.datetime.now() # 当前时间戳 (用于文件名)
        state = self.kb_interface.update()

        if state == "start":
            if self.save_path is not None:
                print("\n警告：在未停止上一次录制的情况下按下了 'S'。旧数据可能未被标记。")
                self.save_path = None # 强制重置
                
            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"\n开始录制，保存至: {self.save_path}") # 使用 \n 换行
            self.last_saved_wrist_ts = 0.0
            self.last_saved_base_ts = 0.0
            self.frame_count = 0
            
        elif state == "save":
            if self.save_path is not None:
                current_wrist_ts = obs.get('wrist_timestamp', 0.0)
                current_base_ts = obs.get('base_timestamp', 0.0)
                has_new_wrist = current_wrist_ts > 0 and current_wrist_ts > self.last_saved_wrist_ts
                has_new_base = current_base_ts > 0 and current_base_ts > self.last_saved_base_ts
                has_new_frame = has_new_wrist or has_new_base
                
                if has_new_frame:
                    filename = f"frame_{self.frame_count:04d}_{dt.strftime('%Y%m%d_%H%M%S_%f')}.pkl"
                    filepath = self.save_path / filename
                    data_to_save = {'obs': obs, 'action': action}
                    try: 
                        with open(filepath, 'wb') as f: pickle.dump(data_to_save, f)
                        if has_new_wrist: self.last_saved_wrist_ts = current_wrist_ts
                        if has_new_base: self.last_saved_base_ts = current_base_ts
                        self.frame_count += 1
                    except Exception as e: print(f"\n❌ 保存帧失败: {filepath}, 错误: {e}")

        elif state == "normal":
            if self.save_path is not None:
                finished_path = self.save_path
                finished_frame_count = self.frame_count
                
                print(f"\n停止录制。总共保存了 {finished_frame_count} 帧到 {finished_path}")
                
                # --- 询问用户 ---
                while True:
                    # 使用 Pygame 获取键盘事件可能更可靠，但 input() 简单直接
                    # (确保运行 run_env.py 的终端是活动的)
                    user_input = input("  刚刚的演示是否成功？(g = Good / n = Not Good/Failed): ").strip().lower()
                    if user_input == 'n':
                        try:
                            failed_path = finished_path.with_name(finished_path.name + "_Failed")
                            finished_path.rename(failed_path)
                            print(f"  ✅ 文件夹已标记为失败: {failed_path.name}")
                        except Exception as e:
                            print(f"  ❌ 重命名文件夹失败: {e}")
                        break 
                    elif user_input == 'g':
                        print(f"  ✅ 数据已保留: {finished_path.name}")
                        break 
                    else:
                        print("  无效输入，请输入 'g' 或 'n'。")
                # ---
                
                self.save_path = None # 在询问结束后重置
            
        elif state == "quit":
             if self.save_path is not None:
                 finished_path = self.save_path
                 finished_frame_count = self.frame_count
                 print(f"\n程序退出时仍在录制。总共保存了 {finished_frame_count} 帧到 {finished_path}")
                 while True:
                     user_input = input("  标记此未完成的演示？(g = Good / n = Not Good/Failed): ").strip().lower()
                     if user_input == 'n':
                         try:
                             failed_path = finished_path.with_name(finished_path.name + "_Failed")
                             finished_path.rename(failed_path)
                             print(f"  ✅ 文件夹已标记为失败: {failed_path.name}")
                         except Exception as e: print(f"  ❌ 重命名文件夹失败: {e}")
                         break 
                     elif user_input == 'g':
                         print(f"  ✅ 数据已保留: {finished_path.name}")
                         break 
                     else: print("  无效输入，请输入 'g' 或 'n'。")
                 self.save_path = None
                 
             print("\nExiting.")
             return "quit"
            
        else:
            raise ValueError(f"Invalid state {state}")

        return None


# (run_control_loop - 移除了详细计时，但核心逻辑不变)
def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True, 
    use_colors: bool = False,
) -> None:
    """运行主控制循环 (无详细计时)。"""
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored
            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError: start_msg = "\nStart 🚀🚀🚀"
    else: start_msg = "\nStart 🚀🚀🚀"
    print(start_msg)

    start_time = time.time(); last_print_time = time.time()
    loop_count = 0; actual_hz = 0.0

    try:
        obs = env.get_obs()
    except Exception as e: print(f"\n❌ 获取初始观测时出错: {e}"); return

    while True:
        try:
            action = agent.act(obs) 

            if save_interface is not None:
                result = save_interface.update(obs, action)
                if result == "quit": break # 如果 update 返回 "quit"，退出

            obs = env.step(action) 
            
            loop_count += 1
            
            # 打印基本频率
            current_time = time.time()
            if print_timing and (current_time - last_print_time >= 1.0):
                elapsed_time = current_time - start_time
                time_since_last_print = current_time - last_print_time
                if time_since_last_print > 0: 
                    actual_hz = loop_count / time_since_last_print
                message = f"\rT:{elapsed_time:.1f}s | Hz:{actual_hz:.1f}          " # 简化输出
                
                if colors_available: print(colored(message, color="white", attrs=["bold"]), end="", flush=True)
                else: print(message, end="", flush=True)
                
                last_print_time = current_time
                loop_count = 0

        except KeyboardInterrupt:
            print("\n🛑 检测到中断信号 (Ctrl+C)...")
            # 确保在退出前调用一次 update 来触发可能的询问
            if save_interface is not None:
                 save_interface.kb_interface.state = 'quit' # 尝试设置状态 (依赖 KBReset 实现)
                 try: 
                     # 尝试传递最新状态进行询问
                     _obs = obs if 'obs' in locals() else {}
                     _action = action if 'action' in locals() else np.array([])
                     save_interface.update(_obs, _action) 
                 except Exception as e_qiut:
                      print(f"退出时处理 SaveInterface 出错: {e_qiut}") # 避免这里再次崩溃
            break
        except Exception as e:
            print(f"\n❌ 控制循环出错: {e}")
            import traceback; traceback.print_exc(); break
            
    print("\n控制循环结束。")


