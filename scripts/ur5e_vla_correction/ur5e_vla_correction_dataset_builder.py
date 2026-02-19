"""ur5e_vla_correction dataset: VLA correction (recovery after grasp failure).

Includes only correction phase frames, with 3 stop-signal frames appended
at the end (gripper=255). Teaches the model recovery behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlds_builder_base import Ur5eVlaBuilderBase


class Ur5eVlaCorrection(Ur5eVlaBuilderBase):
    """VLA correction dataset — correction phase only + stop signal."""

    PHASE_FILTER = {"correction"}
    APPEND_STOP_SIGNAL = True
    AMPLIFY_TRIGGER = True
