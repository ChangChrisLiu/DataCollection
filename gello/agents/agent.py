from typing import Any, Dict, Protocol, Union

import numpy as np

# Joint-position array or Cartesian velocity/skill command dict.
Action = Union[np.ndarray, Dict[str, Any]]


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> Action:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: joint-position array (GELLO/SpaceMouse) or
                command dict (JoystickAgent velocity/skill).
        """
        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> Action:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> Action:
        left_obs: Dict[str, Any] = {}
        right_obs: Dict[str, Any] = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        left_action = self.agent_left.act(left_obs)
        right_action = self.agent_right.act(right_obs)
        assert isinstance(left_action, np.ndarray) and isinstance(
            right_action, np.ndarray
        ), "BimanualAgent requires both sub-agents to return np.ndarray"
        return np.concatenate([left_action, right_action])
