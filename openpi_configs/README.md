# OpenPI UR5e Configuration Files

Reference copies of the custom UR5e config files deployed in the OpenPI codebase (`third_party/openpi/`). These files are the **only** modifications needed on top of upstream OpenPI.

## Files

| File | Deployed Location in OpenPI | Purpose |
|------|----------------------------|---------|
| `ur5e_policy.py` | `src/openpi/policies/ur5e_policy.py` | Input/output transforms mapping UR5e observations to model format |
| `config_ur5e_section.py` | Inserted into `src/openpi/training/config.py` | 52 TrainConfig entries (4 old + 48 new FPS-variant configs) |
| `compute_all_ur5e_norm_stats.sh` | `scripts/compute_all_ur5e_norm_stats.sh` | Batch norm stats computation for all configs |

## Quick Setup on a New Machine

```bash
# 1. Clone OpenPI
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync

# 2. Copy UR5e policy
cp /path/to/DataCollection/openpi_configs/ur5e_policy.py src/openpi/policies/

# 3. Add UR5e import to config.py (after other policy imports)
#    import openpi.policies.ur5e_policy as ur5e_policy

# 4. Add LeRobotUR5eDataConfig class and TrainConfig entries from config_ur5e_section.py
#    into src/openpi/training/config.py

# 5. Copy norm stats script
cp /path/to/DataCollection/openpi_configs/compute_all_ur5e_norm_stats.sh scripts/

# 6. Change default cache dir (optional)
#    In src/openpi/shared/download.py: DEFAULT_CACHE_DIR = "~/openpi_data"

# 7. Compute norm stats
export HF_LEROBOT_HOME=~/lerobot_datasets
bash scripts/compute_all_ur5e_norm_stats.sh

# 8. Train
HF_LEROBOT_HOME=~/lerobot_datasets XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz --exp-name test
```

## Config Naming Convention

```
{model}_ur5e_{target}[_lora]_{fps}hz
```

- **Models**: `pi0`, `pi0_fast`, `pi05`, `pi05_droid`
- **Targets**: `planner`, `e2e`, `correction`
- **Variants**: `_lora` suffix = LoRA fine-tune, no suffix = full fine-tune
- **FPS**: `_10hz` or `_30hz`

The 4 original configs (`pi0_ur5e`, `pi0_ur5e_lora`, `pi0_fast_ur5e`, `pi0_fast_ur5e_lora`) are kept for backward compatibility and point to `ur5e_planner_30hz`.
