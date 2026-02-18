"""ur5e_vla_planner dataset: VLA planner (teleop + stop signal).

Includes only teleop phase frames, with 3 stop-signal frames appended
at the end (gripper=255). Teaches the model WHEN to call a skill.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlds_builder_base import Ur5eVlaBuilderBase


class Ur5eVlaPlanner(Ur5eVlaBuilderBase):
    """VLA planner dataset — teleop only + stop signal."""

    PHASE_FILTER = {"teleop"}
    APPEND_STOP_SIGNAL = True
