"""UR5e policy transforms for OpenPI fine-tuning.

UR5e dataset schema (from LeRobot v2.1):
    base_rgb      (256,256,3)   image    OAK-D Pro third-person camera
    wrist_rgb     (256,256,3)   image    RealSense D435i wrist camera
    state         (7,)          float32  [q0-q5, gripper/255]
    action        (7,)          float32  [q0_next..q5_next, gripper_next/255]
    task          string                 Language instruction (CPU or RAM pick-and-place)

Actions are absolute next-step joint positions. DeltaActions transform (applied in
config.py) converts to delta for training; AbsoluteActions converts back at inference.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5e policy."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "pick up the cpu",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    """Convert UR5e inputs to model-expected format.

    After RepackTransform, dataset keys are mapped to:
        observation/image       <- base_rgb
        observation/wrist_image <- wrist_rgb
        observation/state       <- state (7D: 6 joints + gripper)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    """Parse model output actions for UR5e (7D: 6 joints + gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
