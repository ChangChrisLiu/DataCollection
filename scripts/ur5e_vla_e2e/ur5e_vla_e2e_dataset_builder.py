"""ur5e_vla_e2e dataset: Full end-to-end VLA trajectory.

Includes all phases (teleop + skill + correction + skill_resume).
No stop signals appended.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlds_builder_base import Ur5eVlaBuilderBase


class Ur5eVlaE2e(Ur5eVlaBuilderBase):
    """End-to-end VLA dataset — all phases, no stop signal."""

    PHASE_FILTER = {"teleop", "skill", "correction", "skill_resume"}
    APPEND_STOP_SIGNAL = False
