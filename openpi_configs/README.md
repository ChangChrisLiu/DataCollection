# OpenPI UR5e Configuration Files

Reference copies of the custom UR5e config files deployed in the OpenPI codebase. These files are the **only** modifications needed on top of upstream OpenPI to support UR5e fine-tuning.

## Files

| File | Deployed Location in OpenPI | Type | Purpose |
|------|----------------------------|------|---------|
| `ur5e_policy.py` | `src/openpi/policies/ur5e_policy.py` | NEW | Input/output transforms mapping UR5e observations to model format |
| `config_ur5e_section.py` | Inserted into `src/openpi/training/config.py` | REFERENCE | LeRobotUR5eDataConfig class + 52 TrainConfig entries (patterns + full name list) |
| `compute_all_ur5e_norm_stats.sh` | `scripts/compute_all_ur5e_norm_stats.sh` | NEW | Batch norm stats computation (12 unique configs, ~3.5 hrs total) |
| `SERVER_SETUP_HPRC.md` | N/A (docs only) | GUIDE | Complete HPRC GRACE server deployment (Steps 0-8) |
| `TRAINING_RUN_1.md` | N/A (docs only) | GUIDE | Pi0.5-DROID LoRA x 3 targets @ 10hz training guide |

### Additional Modification (not a separate file)

**`src/openpi/shared/download.py`** — One-line change to default cache directory:
```python
# Line 19, change:
DEFAULT_CACHE_DIR = "~/.cache/openpi"
# To:
DEFAULT_CACHE_DIR = "~/openpi_data"
```

This moves the model checkpoint cache from a hidden directory to a visible location. The `OPENPI_DATA_HOME` env var can also override this at runtime.

---

## Quick Setup on a New Machine

```bash
# 1. Clone OpenPI and install
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 1b. Fix numpy version conflict (openpi pins <2.0, but rerun-sdk needs >=2)
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' pyproject.toml
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' packages/openpi-client/pyproject.toml

GIT_LFS_SKIP_SMUDGE=1 uv sync

# 2. Copy UR5e policy file
cp /path/to/DataCollection/openpi_configs/ur5e_policy.py src/openpi/policies/

# 3. Add UR5e import to config.py (after other policy imports, ~line 23)
#    import openpi.policies.ur5e_policy as ur5e_policy

# 4. Add LeRobotUR5eDataConfig class and 52 TrainConfig entries
#    from config_ur5e_section.py into src/openpi/training/config.py
#    (see config_ur5e_section.py for exact insertion points)

# 5. Copy norm stats script
cp /path/to/DataCollection/openpi_configs/compute_all_ur5e_norm_stats.sh scripts/

# 6. Change default cache dir in src/openpi/shared/download.py
#    DEFAULT_CACHE_DIR = "~/openpi_data"

# 7. Set environment variables in ~/.bashrc
export HF_LEROBOT_HOME=~/lerobot_datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 8. Compute norm stats (12 unique, ~18 min each)
bash scripts/compute_all_ur5e_norm_stats.sh

# 9. Create symlinks for remaining configs (see SERVER_SETUP_HPRC.md Step 4)
cd assets
ln -sf pi0_ur5e_planner_lora_10hz pi0_ur5e_planner_10hz     # full → LoRA
ln -sf pi0_fast_ur5e_planner_lora_10hz pi05_ur5e_planner_lora_10hz  # Pi0.5 → Pi0-FAST
# ... (see SERVER_SETUP_HPRC.md Step 4 for complete symlink commands)

# 10. Verify configs load
uv run python -c "
from openpi.training.config import _CONFIGS
ur5e = [c for c in _CONFIGS if 'ur5e' in c.name]
print(f'Found {len(ur5e)} UR5e configs')
"
# Expected: Found 52 UR5e configs

# 11. Train
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name test --project-name ur5e-finetuning
```

---

## Config Naming Convention

```
{model}_ur5e_{target}[_lora]_{fps}hz
```

- **Models**: `pi0`, `pi0_fast`, `pi05` (base checkpoint), `pi05_droid` (DROID checkpoint)
- **Targets**: `planner`, `e2e`, `correction`
- **Variants**: `_lora` suffix = LoRA fine-tune, no suffix = full fine-tune
- **FPS**: `_10hz` or `_30hz`

The 4 original configs (`pi0_ur5e`, `pi0_ur5e_lora`, `pi0_fast_ur5e`, `pi0_fast_ur5e_lora`) are kept for backward compatibility and point to `ur5e_planner_30hz`.

### All 52 Config Names

```
# Original backward-compat (4):
pi0_ur5e, pi0_ur5e_lora, pi0_fast_ur5e, pi0_fast_ur5e_lora

# Pi0 LoRA:           pi0_ur5e_{planner,e2e,correction}_lora_{10,30}hz     (6)
# Pi0 full:           pi0_ur5e_{planner,e2e,correction}_{10,30}hz          (6)
# Pi0-FAST LoRA:      pi0_fast_ur5e_{planner,e2e,correction}_lora_{10,30}hz (6)
# Pi0-FAST full:      pi0_fast_ur5e_{planner,e2e,correction}_{10,30}hz     (6)
# Pi0.5-base LoRA:    pi05_ur5e_{planner,e2e,correction}_lora_{10,30}hz    (6)
# Pi0.5-base full:    pi05_ur5e_{planner,e2e,correction}_{10,30}hz         (6)
# Pi0.5-DROID LoRA:   pi05_droid_ur5e_{planner,e2e,correction}_lora_{10,30}hz (6)
# Pi0.5-DROID full:   pi05_droid_ur5e_{planner,e2e,correction}_{10,30}hz   (6)
```

---

## Norm Stats Sharing

Configs sharing the same dataset + normalization type produce identical stats. Only 12 unique computations are needed:

| Norm Type | Models | Unique Configs | Others |
|-----------|--------|---------------|--------|
| z-score | Pi0 | 6 (compute) | Pi0 full → symlink to Pi0 LoRA |
| quantile | Pi0-FAST, Pi0.5-base, Pi0.5-DROID | 6 (compute Pi0-FAST) | All others → symlink to Pi0-FAST LoRA |

Total: 12 computed + 37 symlinked + 3 pre-uploaded = 52 configs covered.

---

## GRACE Server Deployment

For HPRC GRACE cluster setup, see:
- **[`SERVER_SETUP_HPRC.md`](SERVER_SETUP_HPRC.md)** — Complete 9-step deployment guide
- **[`TRAINING_RUN_1.md`](TRAINING_RUN_1.md)** — Pi0.5-DROID LoRA training (3 targets @ 10hz)

Key GRACE-specific issues:
- `gcsfs`/`aiohttp` ignores `http_proxy` — must pre-download model checkpoint on login node (Step 5)
- SLURM non-interactive bash skips `~/.bashrc` — all env vars must be in SLURM scripts
- `UV_FROZEN=1` required to skip uv dependency re-resolution
