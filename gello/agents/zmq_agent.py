# gello/agents/zmq_agent.py
import pickle
from typing import Any, Dict

import numpy as np
import zmq

from gello.agents.agent import Agent


class ZMQAgent(Agent):
    """Synchronous REQ/REP client for the Gello hardware server."""

    def __init__(
        self, port: int, host: str, num_dofs: int = 7
    ):
        print(
            f"ZMQAgent: Connecting (REQ) to Gello Server "
            f"[tcp://{host}:{port}]"
        )
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
        self._socket.connect(f"tcp://{host}:{port}")
        self.num_dofs = num_dofs
        print(f"ZMQAgent: Connected (expecting {num_dofs}-DOF).")

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Send obs to server and get Gello action."""
        try:
            self._socket.send(pickle.dumps(obs))
            action = pickle.loads(self._socket.recv())
            if action is None:
                raise RuntimeError("Gello server returned None")

            if len(action) != self.num_dofs:
                print(
                    f"ZMQAgent warning: expected {self.num_dofs}-DOF, "
                    f"got {len(action)}-DOF!"
                )
                new_action = np.zeros(self.num_dofs)
                length = min(self.num_dofs, len(action))
                new_action[:length] = action[:length]
                return new_action

            return action
        except Exception as e:
            print(f"ZMQAgent error: {e}. Returning safe no-op.")
            return obs.get(
                "joint_positions", np.zeros(self.num_dofs)
            )
