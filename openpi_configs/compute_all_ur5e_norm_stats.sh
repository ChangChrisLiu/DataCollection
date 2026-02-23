#!/bin/bash
# Compute norm stats for ALL UR5e configs.
# Computes 12 unique norm stat configs (~18 min each, ~3.5 hours total).
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
    # Pi0 LoRA (z-score normalization) — 6 unique, must compute all
    pi0_ur5e_planner_lora_10hz pi0_ur5e_planner_lora_30hz
    pi0_ur5e_e2e_lora_10hz pi0_ur5e_e2e_lora_30hz
    pi0_ur5e_correction_lora_10hz pi0_ur5e_correction_lora_30hz

    # Pi0-FAST LoRA (quantile normalization) — 6 unique, compute these
    # Pi0.5-base and Pi0.5-DROID use the SAME quantile stats → symlink after
    pi0_fast_ur5e_planner_lora_10hz pi0_fast_ur5e_planner_lora_30hz
    pi0_fast_ur5e_e2e_lora_10hz pi0_fast_ur5e_e2e_lora_30hz
    pi0_fast_ur5e_correction_lora_10hz pi0_fast_ur5e_correction_lora_30hz

    # Total: 12 configs computed. All others get symlinks.
    # After this script finishes, run the symlink commands from SERVER_SETUP_HPRC.md Step 4.
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
echo "  e.g., cd assets && ln -sf pi0_ur5e_planner_lora_10hz pi0_ur5e_planner_10hz"
