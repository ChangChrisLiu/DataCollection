#!/bin/bash
# Compute norm stats for ALL UR5e configs.
# Only 6 unique computations needed (~18 min each, ~2 hours total).
#
# compute_norm_stats.py computes mean, std, q01, q99 from raw dataset values.
# Stats are IDENTICAL across all model types for the same dataset — the script
# does not use model_type. Different models just read different fields at
# training time (Pi0 uses mean/std for z-score; Pi0-FAST/Pi0.5 use q01/q99
# for quantile normalization).
#
# We compute using Pi0 LoRA config names (one per dataset), then symlink
# all 46 other config names to those 6 computed dirs.
#
# After this script, run the symlink commands from SERVER_SETUP_HPRC.md Step 4.

set -e
cd "$(dirname "$0")/.."
export HF_LEROBOT_HOME=~/lerobot_datasets

CONFIGS=(
    # One per dataset — stats are model-agnostic, so any config name works.
    # Using Pi0 LoRA names as the canonical computed dirs.
    pi0_ur5e_planner_lora_10hz
    pi0_ur5e_planner_lora_30hz
    pi0_ur5e_e2e_lora_10hz
    pi0_ur5e_e2e_lora_30hz
    pi0_ur5e_correction_lora_10hz
    pi0_ur5e_correction_lora_30hz
)

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Computing norm stats for: $config"
    echo "========================================"
    uv run scripts/compute_norm_stats.py --config-name "$config"
done

echo ""
echo "Done! Now create symlinks for all 46 other configs:"
echo "  See SERVER_SETUP_HPRC.md Step 4 for the complete symlink script."
