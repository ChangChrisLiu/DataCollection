#!/bin/bash
# Compute norm stats for ALL UR5e configs.
# Run on the training server (takes ~18 min each, ~15 hours total for all 45).
# Configs sharing the same dataset + normalization type produce identical stats,
# so you can symlink after computing one representative per group.
#
# Groups (same dataset + same norm type = identical stats):
#   Pi0 (z-score):       6 datasets -> 6 unique computations
#   Pi0-FAST (quantile): 6 datasets -> 6 unique computations
#   Pi0.5-base (quantile): 6 datasets -> same as Pi0-FAST (can symlink)
#   Pi0.5-DROID (quantile): 6 datasets -> same as Pi0-FAST (can symlink)
#
# Minimum unique computations: 12 (6 z-score + 6 quantile)
# With symlinks, only ~4 hours instead of ~15 hours.

set -e
cd "$(dirname "$0")/.."
export HF_LEROBOT_HOME=~/lerobot_datasets

CONFIGS=(
    # Pi0 LoRA (z-score) -- must compute all 6
    pi0_ur5e_planner_lora_10hz pi0_ur5e_planner_lora_30hz
    pi0_ur5e_e2e_lora_10hz pi0_ur5e_e2e_lora_30hz
    pi0_ur5e_correction_lora_10hz pi0_ur5e_correction_lora_30hz
    # Pi0-FAST LoRA (quantile) -- compute these, symlink pi05 variants
    pi0_fast_ur5e_planner_lora_10hz pi0_fast_ur5e_planner_lora_30hz
    pi0_fast_ur5e_e2e_lora_10hz pi0_fast_ur5e_e2e_lora_30hz
    pi0_fast_ur5e_correction_lora_10hz pi0_fast_ur5e_correction_lora_30hz
    # Pi0.5-base LoRA (quantile) -- same stats as Pi0-FAST for same dataset
    pi05_ur5e_planner_lora_10hz pi05_ur5e_planner_lora_30hz
    pi05_ur5e_e2e_lora_10hz pi05_ur5e_e2e_lora_30hz
    pi05_ur5e_correction_lora_10hz pi05_ur5e_correction_lora_30hz
    # Pi0.5-DROID LoRA (quantile) -- 3 already computed locally
    # pi05_droid_ur5e_planner_lora_10hz  # already done
    # pi05_droid_ur5e_e2e_lora_10hz      # already done
    # pi05_droid_ur5e_correction_lora_10hz  # already done
    pi05_droid_ur5e_planner_lora_30hz
    pi05_droid_ur5e_e2e_lora_30hz
    pi05_droid_ur5e_correction_lora_30hz
)

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Computing norm stats for: $config"
    echo "========================================"
    uv run scripts/compute_norm_stats.py --config-name "$config"
done

echo ""
echo "Done! Now symlink identical stats for full-finetune configs:"
echo "  Full finetune configs share stats with their LoRA counterpart."
echo "  e.g., ln -s ../pi0_ur5e_planner_lora_10hz assets/pi0_ur5e_planner_10hz"
