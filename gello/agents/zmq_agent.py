# gello/agents/zmq_agent.py (最终 7-DOF 版)
import zmq, pickle, numpy as np
from typing import Dict, Any
from gello.agents.agent import Agent

class ZMQAgent(Agent):
    """一个 同步 (REQ/REP) 客户端，用于连接到 Gello 硬件服务器。"""
    
    # [关键修复] 默认自由度改回 7
    def __init__(self, port: int, host: str, num_dofs: int = 7):
        print(f"ZMQAgent: 正在连接 (REQ) 到 Gello Server [tcp://{host}:{port}]")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 2000) # 2 秒超时
        self._socket.connect(f"tcp://{host}:{port}")
        self.num_dofs = num_dofs
        print(f"✅ ZMQAgent: 已连接 (期望 {num_dofs}-DOF)。")

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """将 obs 发送到服务器并获取 Gello 动作。"""
        try:
            self._socket.send(pickle.dumps(obs))
            action = pickle.loads(self._socket.recv())
            if action is None: raise RuntimeError("Gello 服务器返回了 None")
            
            # 确保我们收到的
            if len(action) != self.num_dofs:
                 print(f"⚠️ ZMQAgent 警告: 期望 {self.num_dofs}-DOF, 但收到了 {len(action)}-DOF!")
                 # (尝试填充或截断，但这不应该发生)
                 new_action = np.zeros(self.num_dofs)
                 l = min(self.num_dofs, len(action))
                 new_action[:l] = action[:l]
                 return new_action

            return action
        except Exception as e:
            print(f"❌ ZMQAgent 错误: {e}. 返回安全的 '无操作'。")
            return obs.get("joint_positions", np.zeros(self.num_dofs))


