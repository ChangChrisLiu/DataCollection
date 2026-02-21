"""UR5e configuration section for OpenPI training config.

This file is a REFERENCE COPY of the UR5e-specific code added to:
    src/openpi/training/config.py

To deploy on a new OpenPI installation:
1. Add the import at the top of config.py:
       import openpi.policies.ur5e_policy as ur5e_policy
2. Add LeRobotUR5eDataConfig class (after LeRobotLiberoDataConfig)
3. Add all TrainConfig entries to the _CONFIGS list

Total: 52 configs (4 original backward-compat + 48 FPS-variant)
"""

# ==============================================================================
# LeRobotUR5eDataConfig class — add after LeRobotLiberoDataConfig
# ==============================================================================

# @dataclasses.dataclass(frozen=True)
# class LeRobotUR5eDataConfig(DataConfigFactory):
#     """Config for UR5e LeRobot datasets (planner, e2e, correction).
#
#     Dataset features: base_rgb, wrist_rgb, state (7D), action (7D absolute next-step).
#     Actions are absolute joint positions — DeltaActions converts to delta for training.
#     """
#
#     @override
#     def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
#         repack_transform = _transforms.Group(
#             inputs=[
#                 _transforms.RepackTransform(
#                     {
#                         "observation/image": "base_rgb",
#                         "observation/wrist_image": "wrist_rgb",
#                         "observation/state": "state",
#                         "actions": "action",
#                         "prompt": "prompt",
#                     }
#                 )
#             ]
#         )
#
#         data_transforms = _transforms.Group(
#             inputs=[ur5e_policy.UR5eInputs(model_type=model_config.model_type)],
#             outputs=[ur5e_policy.UR5eOutputs()],
#         )
#
#         delta_action_mask = _transforms.make_bool_mask(6, -1)
#         data_transforms = data_transforms.push(
#             inputs=[_transforms.DeltaActions(delta_action_mask)],
#             outputs=[_transforms.AbsoluteActions(delta_action_mask)],
#         )
#
#         model_transforms = ModelTransformFactory()(model_config)
#
#         return dataclasses.replace(
#             self.create_base_config(assets_dirs, model_config),
#             repack_transforms=repack_transform,
#             data_transforms=data_transforms,
#             model_transforms=model_transforms,
#             action_sequence_keys=("action",),
#         )


# ==============================================================================
# Config Matrix (48 new + 4 original = 52 total)
# ==============================================================================
#
# Naming: {model}_ur5e_{target}[_lora]_{fps}hz
#
# | Model        | Checkpoint   | LoRA batch | Full batch | lr_schedule        |
# |--------------|-------------|------------|------------|-------------------|
# | pi0          | pi0_base     | default    | default    | default            |
# | pi0_fast     | pi0_fast_base| default    | default    | default            |
# | pi05         | pi05_base    | 32         | 256        | CosineDecay 5e-5   |
# | pi05_droid   | pi05_droid   | 32         | 256        | CosineDecay 5e-5   |
#
# Datasets (6):
#   ChangChrisLiu/ur5e_planner_10hz    ChangChrisLiu/ur5e_planner_30hz
#   ChangChrisLiu/ur5e_e2e_10hz        ChangChrisLiu/ur5e_e2e_30hz
#   ChangChrisLiu/ur5e_correction_10hz ChangChrisLiu/ur5e_correction_30hz
#
# Original backward-compat configs (all point to ur5e_planner_30hz):
#   pi0_ur5e, pi0_ur5e_lora, pi0_fast_ur5e, pi0_fast_ur5e_lora
#
# See the deployed config.py for the full list of TrainConfig entries.
