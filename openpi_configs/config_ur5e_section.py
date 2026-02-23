"""UR5e configuration section for OpenPI training config.

This file is a REFERENCE COPY of the UR5e-specific code added to:
    src/openpi/training/config.py

To deploy on a new OpenPI installation:
1. Add the import at the top of config.py (after other policy imports, ~line 23):
       import openpi.policies.ur5e_policy as ur5e_policy
2. Add LeRobotUR5eDataConfig class (after LeRobotLiberoDataConfig, ~line 360)
3. Add all TrainConfig entries to the _CONFIGS list (after existing configs, ~line 815)

Total: 52 configs (4 original backward-compat + 48 FPS-variant)

Config naming: {model}_ur5e_{target}[_lora]_{fps}hz
  Models:  pi0, pi0_fast, pi05 (base checkpoint), pi05_droid (DROID checkpoint)
  Targets: planner, e2e, correction
  FPS:     10hz, 30hz
  Variant: _lora = LoRA fine-tune, no suffix = full fine-tune

Config matrix:
  | Model        | Checkpoint    | LoRA batch | Full batch | lr_schedule       | Norm Type |
  |--------------|---------------|------------|------------|-------------------|-----------|
  | pi0          | pi0_base      | default    | default    | default           | z-score   |
  | pi0_fast     | pi0_fast_base | default    | default    | default           | quantile  |
  | pi05         | pi05_base     | 32         | 256        | CosineDecay 5e-5  | quantile  |
  | pi05_droid   | pi05_droid    | 32         | 256        | CosineDecay 5e-5  | quantile  |

Datasets (6):
  ChangChrisLiu/ur5e_planner_10hz    ChangChrisLiu/ur5e_planner_30hz
  ChangChrisLiu/ur5e_e2e_10hz        ChangChrisLiu/ur5e_e2e_30hz
  ChangChrisLiu/ur5e_correction_10hz ChangChrisLiu/ur5e_correction_30hz

Norm stats sharing (same dataset + same norm type = identical stats):
  Pi0 (z-score):          6 unique, must compute all
  Pi0-FAST (quantile):    6 unique, compute these as representative
  Pi0.5-base (quantile):  symlink to Pi0-FAST
  Pi0.5-DROID (quantile): symlink to Pi0-FAST
  Full finetune:          symlink to LoRA (same norm type)
  Total: 12 computed + 37 symlinked + 3 pre-uploaded = 52
"""

# ==============================================================================
# STEP 1: Import (add at top of config.py, ~line 23)
# ==============================================================================

# import openpi.policies.ur5e_policy as ur5e_policy


# ==============================================================================
# STEP 2: LeRobotUR5eDataConfig class (add after LeRobotLiberoDataConfig)
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
#         # Remap dataset keys to match the inference pipeline keys.
#         # Dataset columns: base_rgb, wrist_rgb, state, action (singular), prompt (from task)
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
#         # Convert absolute joint actions to delta (gripper stays absolute).
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
#             # Dataset column is "action" (singular), not "actions".
#             action_sequence_keys=("action",),
#         )


# ==============================================================================
# STEP 3: TrainConfig entries (add to _CONFIGS list)
# ==============================================================================
# All entries below go inside the _CONFIGS = [...] list.

_UR5E_CONFIGS = [
    #
    # Fine-tuning UR5e configs.
    #
    # UR5e e-waste disassembly with dual cameras (base + wrist).
    # Datasets at ~/lerobot_datasets/ChangChrisLiu/ — set HF_LEROBOT_HOME=~/lerobot_datasets.
    # Actions are absolute joint positions; DeltaActions converts to delta during training.

    # ---- Original backward-compat configs (all use ur5e_planner_30hz) ----
    # TrainConfig(
    #     name="pi0_ur5e",
    #     model=pi0_config.Pi0Config(),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_30hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=30_000,
    # ),
    # TrainConfig(
    #     name="pi0_ur5e_lora",
    #     model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_30hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_config.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),
    # TrainConfig(
    #     name="pi0_fast_ur5e",
    #     model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_30hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    #     num_train_steps=30_000,
    # ),
    # TrainConfig(
    #     name="pi0_fast_ur5e_lora",
    #     model=pi0_fast.Pi0FASTConfig(
    #         action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
    #     ),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_30hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_fast.Pi0FASTConfig(
    #         action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),

    # ================================================================
    # UR5e FPS-variant configs — full matrix of model x target x variant x FPS.
    # 4 models x 3 targets x 2 variants (LoRA/full) x 2 FPS = 48 configs.
    # Naming: {model}_ur5e_{target}[_lora]_{fps}hz
    # ================================================================

    # ---- Pi0 LoRA ----
    # TrainConfig(
    #     name="pi0_ur5e_planner_lora_10hz",
    #     model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_config.Pi0Config(
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),
    # ... (planner_30hz, e2e_10hz, e2e_30hz, correction_10hz, correction_30hz — same pattern)

    # ---- Pi0 full finetune ----
    # TrainConfig(
    #     name="pi0_ur5e_planner_10hz",
    #     model=pi0_config.Pi0Config(),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     num_train_steps=30_000,
    # ),
    # ... (planner_30hz, e2e_10hz, e2e_30hz, correction_10hz, correction_30hz — same pattern)

    # ---- Pi0-FAST LoRA ----
    # TrainConfig(
    #     name="pi0_fast_ur5e_planner_lora_10hz",
    #     model=pi0_fast.Pi0FASTConfig(
    #         action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora",
    #     ),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_fast.Pi0FASTConfig(
    #         action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora",
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),
    # ... (same pattern for all 6 datasets)

    # ---- Pi0-FAST full finetune ----
    # TrainConfig(
    #     name="pi0_fast_ur5e_planner_10hz",
    #     model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
    #     num_train_steps=30_000,
    # ),
    # ... (same pattern for all 6 datasets)

    # ---- Pi0.5 (base checkpoint) LoRA ----
    # TrainConfig(
    #     name="pi05_ur5e_planner_lora_10hz",
    #     model=pi0_config.Pi0Config(
    #         pi05=True, action_horizon=10, discrete_state_input=False,
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    #     ),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=32,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_config.Pi0Config(
    #         pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),
    # ... (same pattern for all 6 datasets)

    # ---- Pi0.5 (base checkpoint) full finetune ----
    # TrainConfig(
    #     name="pi05_ur5e_planner_10hz",
    #     model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=256,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
    #     ),
    #     ema_decay=0.999,
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    #     num_train_steps=30_000,
    # ),
    # ... (same pattern for all 6 datasets)

    # ---- Pi0.5 (DROID checkpoint) LoRA ----
    # TrainConfig(
    #     name="pi05_droid_ur5e_planner_lora_10hz",
    #     model=pi0_config.Pi0Config(
    #         pi05=True, action_horizon=10, discrete_state_input=False,
    #         paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    #     ),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=32,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
    #     ),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
    #     num_train_steps=30_000,
    #     freeze_filter=pi0_config.Pi0Config(
    #         pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    #     ).get_freeze_filter(),
    #     ema_decay=None,
    # ),
    # ... (same pattern for all 6 datasets)

    # ---- Pi0.5 (DROID checkpoint) full finetune ----
    # TrainConfig(
    #     name="pi05_droid_ur5e_planner_10hz",
    #     model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    #     data=LeRobotUR5eDataConfig(
    #         repo_id="ChangChrisLiu/ur5e_planner_10hz",
    #         base_config=DataConfig(prompt_from_task=True),
    #     ),
    #     batch_size=256,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
    #     ),
    #     ema_decay=0.999,
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
    #     num_train_steps=30_000,
    # ),
    # ... (same pattern for all 6 datasets)
]

# ==============================================================================
# Complete config name list (all 52)
# ==============================================================================
#
# Backward-compat (4) — all use ur5e_planner_30hz:
#   pi0_ur5e, pi0_ur5e_lora, pi0_fast_ur5e, pi0_fast_ur5e_lora
#
# Pi0 LoRA (6):
#   pi0_ur5e_planner_lora_10hz, pi0_ur5e_planner_lora_30hz
#   pi0_ur5e_e2e_lora_10hz, pi0_ur5e_e2e_lora_30hz
#   pi0_ur5e_correction_lora_10hz, pi0_ur5e_correction_lora_30hz
#
# Pi0 full (6):
#   pi0_ur5e_planner_10hz, pi0_ur5e_planner_30hz
#   pi0_ur5e_e2e_10hz, pi0_ur5e_e2e_30hz
#   pi0_ur5e_correction_10hz, pi0_ur5e_correction_30hz
#
# Pi0-FAST LoRA (6):
#   pi0_fast_ur5e_planner_lora_10hz, pi0_fast_ur5e_planner_lora_30hz
#   pi0_fast_ur5e_e2e_lora_10hz, pi0_fast_ur5e_e2e_lora_30hz
#   pi0_fast_ur5e_correction_lora_10hz, pi0_fast_ur5e_correction_lora_30hz
#
# Pi0-FAST full (6):
#   pi0_fast_ur5e_planner_10hz, pi0_fast_ur5e_planner_30hz
#   pi0_fast_ur5e_e2e_10hz, pi0_fast_ur5e_e2e_30hz
#   pi0_fast_ur5e_correction_10hz, pi0_fast_ur5e_correction_30hz
#
# Pi0.5-base LoRA (6):
#   pi05_ur5e_planner_lora_10hz, pi05_ur5e_planner_lora_30hz
#   pi05_ur5e_e2e_lora_10hz, pi05_ur5e_e2e_lora_30hz
#   pi05_ur5e_correction_lora_10hz, pi05_ur5e_correction_lora_30hz
#
# Pi0.5-base full (6):
#   pi05_ur5e_planner_10hz, pi05_ur5e_planner_30hz
#   pi05_ur5e_e2e_10hz, pi05_ur5e_e2e_30hz
#   pi05_ur5e_correction_10hz, pi05_ur5e_correction_30hz
#
# Pi0.5-DROID LoRA (6):
#   pi05_droid_ur5e_planner_lora_10hz, pi05_droid_ur5e_planner_lora_30hz
#   pi05_droid_ur5e_e2e_lora_10hz, pi05_droid_ur5e_e2e_lora_30hz
#   pi05_droid_ur5e_correction_lora_10hz, pi05_droid_ur5e_correction_lora_30hz
#
# Pi0.5-DROID full (6):
#   pi05_droid_ur5e_planner_10hz, pi05_droid_ur5e_planner_30hz
#   pi05_droid_ur5e_e2e_10hz, pi05_droid_ur5e_e2e_30hz
#   pi05_droid_ur5e_correction_10hz, pi05_droid_ur5e_correction_30hz
#
# The authoritative source is the deployed config.py at:
#   /home/chris/openpi/src/openpi/training/config.py (local)
#   $SCRATCH/openpi/src/openpi/training/config.py (GRACE)
