# Training Run 1: Pi0.5-DROID LoRA x 3 Targets @ 10hz

**Cluster**: TAMU HPRC GRACE (1 GPU)
**Model**: Pi0.5-DROID (pre-trained on DROID single-arm manipulation data)
**Variant**: LoRA fine-tuning (single GPU)
**Steps**: 30,000 per run (3 runs sequential)
**Estimated total time**: ~36-45 hours for all 3 runs

## Three Training Runs (Sequential)

| Order | Config Name | Dataset | Frames | What It Learns |
|-------|------------|---------|--------|----------------|
| 1 | `pi05_droid_ur5e_planner_lora_10hz` | `ur5e_planner_10hz` | 87,426 | Teleop approach — when to hand off to skill |
| 2 | `pi05_droid_ur5e_e2e_lora_10hz` | `ur5e_e2e_10hz` | 222,016 | Full task — all 4 phases autonomously |
| 3 | `pi05_droid_ur5e_correction_lora_10hz` | `ur5e_correction_10hz` | 29,109 | Grasp recovery — correct after failed grasp |

All use 529 episodes, 10hz (every 3rd frame from 30hz raw data), batch_size=32.

---

## Prerequisites Checklist

Run on the Grace login node before submitting jobs:

```bash
ssh changliu.chris@grace.hprc.tamu.edu
cd $SCRATCH/openpi

# 1. Files uploaded
ls src/openpi/policies/ur5e_policy.py

# 2. Environment variables (should be in ~/.bashrc)
echo $HF_LEROBOT_HOME       # /scratch/user/changliu.chris
echo $OPENPI_DATA_HOME       # /scratch/user/changliu.chris/openpi_data

# 3. Datasets accessible
ls $HF_LEROBOT_HOME/ChangChrisLiu/ur5e_planner_10hz/
ls $HF_LEROBOT_HOME/ChangChrisLiu/ur5e_e2e_10hz/
ls $HF_LEROBOT_HOME/ChangChrisLiu/ur5e_correction_10hz/

# 4. Norm stats uploaded
ls assets/pi05_droid_ur5e_planner_lora_10hz/
ls assets/pi05_droid_ur5e_e2e_lora_10hz/
ls assets/pi05_droid_ur5e_correction_lora_10hz/

# 5. Config import test
uv run --frozen python -c "
from openpi.training.config import _CONFIGS
for name in ['pi05_droid_ur5e_planner_lora_10hz',
             'pi05_droid_ur5e_e2e_lora_10hz',
             'pi05_droid_ur5e_correction_lora_10hz']:
    match = [c for c in _CONFIGS if c.name == name]
    print(f'{name}: {\"OK\" if match else \"MISSING\"}')"
```

All 5 checks must pass before proceeding.

---

## Step 1: Quick Validation (5-step test)

```bash
# Request interactive GPU session
srun --gres=gpu:1 --mem=64G --time=00:30:00 --partition=gpu --pty bash

# Run 5-step test
cd $SCRATCH/openpi
export UV_FROZEN=1

uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name validate \
    --num-train-steps 5 \
    --no-wandb-enabled \
    --overwrite

# Expected: loss ~1.4-1.5, completes in ~2-3 min
# If successful, exit the interactive session:
exit
```

---

## Step 2: Create the SLURM Job Script

One script runs all 3 targets sequentially:

```bash
cat > $SCRATCH/openpi/train_all_3.slurm << 'SLURM'
#!/bin/bash
#SBATCH --job-name=pi05d_ur5e_3x
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=train_all_3_%j.log

## Environment
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_FROZEN=1
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

cd /scratch/user/changliu.chris/openpi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

## Run 1: Planner
echo ""
echo "========================================"
echo "RUN 1/3: PLANNER — $(date)"
echo "========================================"
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name planner_v1 \
    --project-name ur5e-finetuning \
    --num-train-steps 30000 \
    --overwrite
echo "Planner finished at $(date)"

## Run 2: End-to-End
echo ""
echo "========================================"
echo "RUN 2/3: E2E — $(date)"
echo "========================================"
uv run scripts/train.py pi05_droid_ur5e_e2e_lora_10hz \
    --exp-name e2e_v1 \
    --project-name ur5e-finetuning \
    --num-train-steps 30000 \
    --overwrite
echo "E2E finished at $(date)"

## Run 3: Correction
echo ""
echo "========================================"
echo "RUN 3/3: CORRECTION — $(date)"
echo "========================================"
uv run scripts/train.py pi05_droid_ur5e_correction_lora_10hz \
    --exp-name correction_v1 \
    --project-name ur5e-finetuning \
    --num-train-steps 30000 \
    --overwrite
echo "Correction finished at $(date)"

echo ""
echo "========================================"
echo "ALL 3 RUNS COMPLETE — $(date)"
echo "========================================"
SLURM
```

**Important**: Replace `<YOUR_WANDB_API_KEY>` with your actual key before submitting.

---

## Step 3: Submit

```bash
cd $SCRATCH/openpi
sbatch train_all_3.slurm
```

---

## Step 4: Monitor

```bash
# Check job status
squeue -u changliu.chris

# Watch live output (find the log file)
ls -t train_all_3_*.log | head -1
tail -f train_all_3_*.log

# Wandb dashboard — all 3 runs appear here:
# https://wandb.ai/losttemplor-texas-a-m-university/ur5e-finetuning
```

**What to look for in wandb:**
- `loss` should decrease over time (starts ~1.4, should drop to ~0.5-1.0)
- `grad_norm` should be stable (not exploding)
- Camera sample images at step 0 (sanity check that data loaded correctly)
- Checkpoints saved every 1,000 steps

---

## Step 5: Download Checkpoints to Local Machine

After the job completes, download the final checkpoints for local inference:

```bash
# Run from LOCAL machine (not server):
SERVER=changliu.chris@grace.hprc.tamu.edu
REMOTE=$SERVER:/scratch/user/changliu.chris/openpi/checkpoints

# Planner
scp -r $REMOTE/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000

# E2E
scp -r $REMOTE/pi05_droid_ur5e_e2e_lora_10hz/e2e_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_e2e_lora_10hz/e2e_v1/30000

# Correction
scp -r $REMOTE/pi05_droid_ur5e_correction_lora_10hz/correction_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_correction_lora_10hz/correction_v1/30000
```

---

## Step 6: Serve Policy Locally for Robot Testing

```bash
cd ~/openpi

# Serve one policy at a time (port 8000):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_droid_ur5e_planner_lora_10hz \
    --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000
```

**Client inference:**
```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer({
    "observation/image": base_rgb,           # (256,256,3) uint8
    "observation/wrist_image": wrist_rgb,    # (256,256,3) uint8
    "observation/state": state_7d,           # [q0-q5, gripper/255] float32
    "prompt": "Pick up the CPU",
})
actions = result["actions"]  # (10, 7) — 10-step action chunk, absolute joint positions
```

---

## Timing Budget (1 GPU, Sequential)

| Steps/Run | Per Run | 3 Runs Total | Fits in 6 Days (144h)? |
|-----------|---------|--------------|------------------------|
| **30,000** | ~12-15 hrs | **~36-45 hrs** | **Yes** (99 hrs margin) |
| 50,000 | ~20-25 hrs | ~60-75 hrs | Yes (69 hrs margin) |
| 100,000 | ~40-50 hrs | ~120-150 hrs | Tight (maybe) |

30k steps leaves plenty of room. If the loss curves look promising, resume to 50k+ (see below).

## Resume / Extend Training

To continue training beyond 30k steps without starting over:

```bash
# In the SLURM script, change ONLY these:
#   --num-train-steps 50000   (new TOTAL, not additional steps)
#   Remove --overwrite        (resumes from latest checkpoint)

uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name planner_v1 \
    --project-name ur5e-finetuning \
    --num-train-steps 50000
# This resumes from step 30000 and trains to 50000
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Job stuck in `PENDING` | GPU queue is full. Wait, or check `squeue -u changliu.chris`. |
| `Unable to initialize backend 'cuda'` | Not on a GPU node. Must use `srun --gres=gpu:1` or `sbatch`. |
| `uv` dependency resolution error | Set `export UV_FROZEN=1` in the SLURM script. |
| OOM | Add `--batch-size 16` (or lower) to the train command. |
| Wandb not showing runs | Check `WANDB_API_KEY` is set. Check internet access from GPU node. |
| Job killed (time limit) | The script requests 72h. If not enough, increase `--time`. Resume without `--overwrite`. |
| Loss stuck / not decreasing | Check norm stats. Try lower learning rate: `--lr-schedule.peak-lr 1e-5`. |
| Run 2 or 3 fails | The script continues to next run even if one fails. Check logs. Resubmit just the failed config. |
| Slow first run (~20 min startup) | First run downloads ~10 GB base checkpoint from GCS. Subsequent runs reuse cache. |
