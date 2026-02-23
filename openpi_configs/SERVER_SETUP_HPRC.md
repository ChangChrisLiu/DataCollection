# HPRC GRACE Server Setup for OpenPI UR5e Fine-Tuning

Complete step-by-step instructions for deploying UR5e configs on the TAMU HPRC GRACE cluster.

**Grace cluster specs**: 48 cores / 384 GB RAM per node, up to 2 GPUs per node.
**Scratch storage**: 1 TB / 250k files at `$SCRATCH` (`/scratch/user/changliu.chris`).

## Prerequisites (Already Done)

- OpenPI cloned and installed at `$SCRATCH/openpi`
- `uv sync` completed, numpy compatibility resolved
- 6 LeRobot datasets uploaded to `$SCRATCH/ChangChrisLiu/`
- UV cache set: `export UV_CACHE_DIR=$SCRATCH/.cache/uv`

## Step 0: Update Server OpenPI to Latest Upstream

The server's OpenPI clone may be older than your local copy. Our custom `config.py` is built on the latest upstream and imports modules (e.g. `polaris_config`, `roboarena_config`) that only exist in recent commits. **You must update before uploading custom files.**

```bash
cd $SCRATCH/openpi

# Pull latest upstream (safe — you haven't committed custom changes on server)
git pull origin main

# Re-sync dependencies (frozen = skip resolution, just install from lockfile)
export UV_FROZEN=1
uv sync
```

If `git pull` fails due to local changes on the server:
```bash
git stash          # save local edits
git pull origin main
git stash pop      # re-apply (conflicts on config.py are expected — we overwrite it in Step 1)
```

If `git stash pop` has conflicts, don't worry — Step 1 overwrites `config.py` and `download.py` entirely.

## Important: Fix Your HF_LEROBOT_HOME Path

**Issue 1**: `~` expands to your HOME (`/home/changliu.chris`), not scratch. `~/scratch/...` is WRONG.

**Issue 2**: `HF_LEROBOT_HOME` must point to the **parent** of `ChangChrisLiu/`, because config repo_ids are `ChangChrisLiu/ur5e_planner_10hz`. OpenPI looks for `$HF_LEROBOT_HOME/ChangChrisLiu/ur5e_planner_10hz/`.

```bash
# WRONG:
export HF_LEROBOT_HOME=~/scratch/user/changliu.chris/ChangChrisLiu

# CORRECT — parent of ChangChrisLiu/, using absolute path:
export HF_LEROBOT_HOME=/scratch/user/changliu.chris

# Verify:
ls $HF_LEROBOT_HOME/ChangChrisLiu/ur5e_planner_10hz/
# Should show: data/, meta/, etc.
```

---

## Step 1: Upload Files from Local Machine

After `git pull` (Step 0), upload the **4 custom files + 4 asset dirs** from your local machine. Run `scp` from your **local terminal**:

```bash
# ---- Run on LOCAL machine ----
LOCAL_OPENPI=~/openpi
SERVER=changliu.chris@grace.hprc.tamu.edu
REMOTE_OPENPI=/scratch/user/changliu.chris/openpi

# 1. UR5e policy file (NEW — does not exist upstream)
scp $LOCAL_OPENPI/src/openpi/policies/ur5e_policy.py \
    $SERVER:$REMOTE_OPENPI/src/openpi/policies/ur5e_policy.py

# 2. Modified config.py (OVERWRITES upstream — adds 52 UR5e configs)
scp $LOCAL_OPENPI/src/openpi/training/config.py \
    $SERVER:$REMOTE_OPENPI/src/openpi/training/config.py

# 3. Modified download.py (OVERWRITES upstream — changes default cache dir)
scp $LOCAL_OPENPI/src/openpi/shared/download.py \
    $SERVER:$REMOTE_OPENPI/src/openpi/shared/download.py

# 4. Norm stats computation script (NEW)
scp $LOCAL_OPENPI/scripts/compute_all_ur5e_norm_stats.sh \
    $SERVER:$REMOTE_OPENPI/scripts/compute_all_ur5e_norm_stats.sh

# 5. Pre-computed norm stats (4 dirs, ~16KB each)
scp -r $LOCAL_OPENPI/assets/pi05_droid_ur5e_planner_lora_10hz \
    $SERVER:$REMOTE_OPENPI/assets/
scp -r $LOCAL_OPENPI/assets/pi05_droid_ur5e_e2e_lora_10hz \
    $SERVER:$REMOTE_OPENPI/assets/
scp -r $LOCAL_OPENPI/assets/pi05_droid_ur5e_correction_lora_10hz \
    $SERVER:$REMOTE_OPENPI/assets/
scp -r $LOCAL_OPENPI/assets/pi0_ur5e_lora \
    $SERVER:$REMOTE_OPENPI/assets/
```

**Note**: `misc/polaris_config.py` and `roboarena_config.py` are upstream files — `git pull` in Step 0 brings them in. No need to upload separately.

**What each file does:**

| File | Type | Purpose |
|------|------|---------|
| `src/openpi/policies/ur5e_policy.py` | NEW | UR5e input/output transforms (maps observations to model format) |
| `src/openpi/training/config.py` | MODIFIED | 52 UR5e TrainConfig entries + LeRobotUR5eDataConfig class |
| `src/openpi/shared/download.py` | MODIFIED | DEFAULT_CACHE_DIR changed to `~/openpi_data` |
| `scripts/compute_all_ur5e_norm_stats.sh` | NEW | Batch script to compute all norm stats |
| `assets/pi05_droid_ur5e_*_lora_10hz/` | NEW | 3 pre-computed norm stats (Pi0.5-DROID x 10hz) |
| `assets/pi0_ur5e_lora/` | NEW | 1 pre-computed norm stat (Pi0 x planner x 30hz) |

---

## Step 2: Set Environment Variables

Add to your `~/.bashrc` on GRACE:

```bash
# OpenPI environment (add to ~/.bashrc on GRACE)
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# CRITICAL: skip uv dependency re-resolution (avoids dlimp/tensorflow conflict)
export UV_FROZEN=1

# CRITICAL: Grace compute nodes have NO internet access without proxy
# Required for downloading model checkpoints from GCS and wandb logging
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

# Wandb logging (v1 key format — must use env var, `wandb login` rejects v1 keys)
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

Then: `source ~/.bashrc`

**Why each line matters:**

| Variable | What It Fixes |
|----------|--------------|
| `UV_CACHE_DIR` | Points uv to scratch storage (home quota is small) |
| `HF_LEROBOT_HOME` | Must be PARENT of `ChangChrisLiu/` — OpenPI looks for `$HF_LEROBOT_HOME/ChangChrisLiu/ur5e_*` |
| `HF_HOME` | HuggingFace model cache (FAST tokenizer etc.). Defaults to `~/.cache/huggingface` which may exceed home quota |
| `OPENPI_DATA_HOME` | OpenPI model checkpoint download cache (10+ GB per model, from GCS) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX pre-allocates GPU memory; 0.9 = use 90% |
| `UV_FROZEN=1` | Without this, `uv run` tries to re-resolve deps and hits `dlimp`/`tensorflow` conflict |
| `http_proxy` / `https_proxy` | Grace compute nodes have NO internet — proxy required for GCS checkpoint download + wandb |
| `WANDB_API_KEY` | Wandb v1 keys are 86 chars; `wandb login` CLI rejects anything != 40 chars. Env var bypasses validation |

---

## Step 3: Verify the Upload

SSH to Grace and run:

```bash
cd $SCRATCH/openpi

# 1. Check ur5e_policy.py exists
ls src/openpi/policies/ur5e_policy.py

# 2. Check config.py has UR5e configs (should show many matches)
grep -c "ur5e" src/openpi/training/config.py

# 3. Check norm stats uploaded
ls assets/ | grep ur5e
# Expected:
#   pi05_droid_ur5e_correction_lora_10hz
#   pi05_droid_ur5e_e2e_lora_10hz
#   pi05_droid_ur5e_planner_lora_10hz
#   pi0_ur5e_lora

# 4. Check datasets accessible
ls $HF_LEROBOT_HOME/ChangChrisLiu/
# Expected: ur5e_planner_10hz  ur5e_planner_30hz  ur5e_e2e_10hz
#           ur5e_e2e_30hz  ur5e_correction_10hz  ur5e_correction_30hz

# 5. Python import test (run on login node — no GPU needed)
uv run python -c "
from openpi.training.config import _CONFIGS
ur5e = [c for c in _CONFIGS if 'ur5e' in c.name]
print(f'Found {len(ur5e)} UR5e configs')
for c in ur5e:
    print(f'  {c.name}')
"
# Should print "Found 52 UR5e configs" and list all names
```

---

## Step 4: Compute Norm Stats

You have 4 pre-computed norm stats from the local machine. You need to compute the rest on the server.

### Fix the Script Path First

```bash
cd $SCRATCH/openpi

# Update HF_LEROBOT_HOME in the script to match the server
sed -i 's|export HF_LEROBOT_HOME=~/lerobot_datasets|export HF_LEROBOT_HOME=/scratch/user/changliu.chris|' \
    scripts/compute_all_ur5e_norm_stats.sh

chmod +x scripts/compute_all_ur5e_norm_stats.sh
```

### Submit as SLURM Job

Create `$SCRATCH/openpi/compute_stats.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ur5e_norm_stats
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=norm_stats_%j.log

## Environment (SLURM non-interactive bash skips ~/.bashrc — must export here)
module load WebProxy
export UV_FROZEN=1
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /scratch/user/changliu.chris/openpi
bash scripts/compute_all_ur5e_norm_stats.sh
```

```bash
sbatch compute_stats.slurm

# Monitor:
squeue -u changliu.chris
# View log:
tail -f norm_stats_*.log
```

This computes 12 unique norm stat configs (~18 min each, ~3.5 hours total).

### Or Compute Only What You Need

```bash
# Example: just one config
uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_e2e_lora_30hz
```

### Create Symlinks (After Stats Are Computed)

Configs sharing the same dataset + normalization type produce identical stats. The compute script only computes stats for **12 unique combinations** (6 Pi0 z-score + 6 Pi0-FAST quantile). All other configs get symlinks.

**How norm stats sharing works:**

| Norm Type | Models Using It | Compute From | Symlink To |
|-----------|----------------|--------------|------------|
| **z-score** | Pi0 | `pi0_ur5e_{target}_lora_{fps}hz` | Pi0 full finetune → Pi0 LoRA |
| **quantile** | Pi0-FAST, Pi0.5-base, Pi0.5-DROID | `pi0_fast_ur5e_{target}_lora_{fps}hz` | All others → Pi0-FAST LoRA |

**Total**: 12 computed + 37 symlinked + 3 pre-uploaded (Pi0.5-DROID 10hz) = 52 configs covered.

Run this **after** the norm stats job completes:

```bash
cd $SCRATCH/openpi/assets

# ================================================================
# GROUP 1: Pi0 full finetune → Pi0 LoRA (same z-score stats)
# 6 symlinks
# ================================================================
ln -sf pi0_ur5e_planner_lora_10hz pi0_ur5e_planner_10hz
ln -sf pi0_ur5e_planner_lora_30hz pi0_ur5e_planner_30hz
ln -sf pi0_ur5e_e2e_lora_10hz pi0_ur5e_e2e_10hz
ln -sf pi0_ur5e_e2e_lora_30hz pi0_ur5e_e2e_30hz
ln -sf pi0_ur5e_correction_lora_10hz pi0_ur5e_correction_10hz
ln -sf pi0_ur5e_correction_lora_30hz pi0_ur5e_correction_30hz

# ================================================================
# GROUP 2: Pi0-FAST full finetune → Pi0-FAST LoRA (same quantile stats)
# 6 symlinks
# ================================================================
ln -sf pi0_fast_ur5e_planner_lora_10hz pi0_fast_ur5e_planner_10hz
ln -sf pi0_fast_ur5e_planner_lora_30hz pi0_fast_ur5e_planner_30hz
ln -sf pi0_fast_ur5e_e2e_lora_10hz pi0_fast_ur5e_e2e_10hz
ln -sf pi0_fast_ur5e_e2e_lora_30hz pi0_fast_ur5e_e2e_30hz
ln -sf pi0_fast_ur5e_correction_lora_10hz pi0_fast_ur5e_correction_10hz
ln -sf pi0_fast_ur5e_correction_lora_30hz pi0_fast_ur5e_correction_30hz

# ================================================================
# GROUP 3: Pi0.5-base (ALL variants) → Pi0-FAST LoRA (same quantile)
# 12 symlinks (6 LoRA + 6 full finetune)
# ================================================================
ln -sf pi0_fast_ur5e_planner_lora_10hz pi05_ur5e_planner_lora_10hz
ln -sf pi0_fast_ur5e_planner_lora_30hz pi05_ur5e_planner_lora_30hz
ln -sf pi0_fast_ur5e_e2e_lora_10hz pi05_ur5e_e2e_lora_10hz
ln -sf pi0_fast_ur5e_e2e_lora_30hz pi05_ur5e_e2e_lora_30hz
ln -sf pi0_fast_ur5e_correction_lora_10hz pi05_ur5e_correction_lora_10hz
ln -sf pi0_fast_ur5e_correction_lora_30hz pi05_ur5e_correction_lora_30hz
ln -sf pi0_fast_ur5e_planner_lora_10hz pi05_ur5e_planner_10hz
ln -sf pi0_fast_ur5e_planner_lora_30hz pi05_ur5e_planner_30hz
ln -sf pi0_fast_ur5e_e2e_lora_10hz pi05_ur5e_e2e_10hz
ln -sf pi0_fast_ur5e_e2e_lora_30hz pi05_ur5e_e2e_30hz
ln -sf pi0_fast_ur5e_correction_lora_10hz pi05_ur5e_correction_10hz
ln -sf pi0_fast_ur5e_correction_lora_30hz pi05_ur5e_correction_30hz

# ================================================================
# GROUP 4: Pi0.5-DROID (ALL variants) → Pi0-FAST LoRA (same quantile)
# 12 symlinks (6 LoRA + 6 full finetune)
# Note: 10hz LoRA versions were uploaded in Step 1 — -sf overwrites with symlinks (same data)
# ================================================================
ln -sf pi0_fast_ur5e_planner_lora_10hz pi05_droid_ur5e_planner_lora_10hz
ln -sf pi0_fast_ur5e_planner_lora_30hz pi05_droid_ur5e_planner_lora_30hz
ln -sf pi0_fast_ur5e_e2e_lora_10hz pi05_droid_ur5e_e2e_lora_10hz
ln -sf pi0_fast_ur5e_e2e_lora_30hz pi05_droid_ur5e_e2e_lora_30hz
ln -sf pi0_fast_ur5e_correction_lora_10hz pi05_droid_ur5e_correction_lora_10hz
ln -sf pi0_fast_ur5e_correction_lora_30hz pi05_droid_ur5e_correction_lora_30hz
ln -sf pi0_fast_ur5e_planner_lora_10hz pi05_droid_ur5e_planner_10hz
ln -sf pi0_fast_ur5e_planner_lora_30hz pi05_droid_ur5e_planner_30hz
ln -sf pi0_fast_ur5e_e2e_lora_10hz pi05_droid_ur5e_e2e_10hz
ln -sf pi0_fast_ur5e_e2e_lora_30hz pi05_droid_ur5e_e2e_30hz
ln -sf pi0_fast_ur5e_correction_lora_10hz pi05_droid_ur5e_correction_10hz
ln -sf pi0_fast_ur5e_correction_lora_30hz pi05_droid_ur5e_correction_30hz

# ================================================================
# GROUP 5: Backward-compat (original 4 configs)
# 1 symlink (pi0_fast_ur5e and pi0_fast_ur5e_lora already point to planner_30hz in config)
# ================================================================
ln -sf pi0_ur5e_lora pi0_ur5e

echo "Created $(ls -l | grep '^l' | wc -l) symlinks"
```

---

## Step 5: Pre-Download Model Files on Login Node

**Critical**: OpenPI downloads model checkpoints from Google Cloud Storage (GCS) using `gcsfs`/`aiohttp`. **`aiohttp` ignores `http_proxy`/`https_proxy` env vars**, so GCS downloads fail on compute nodes even with `module load WebProxy`. The fix is to pre-download everything on a **login node** (which has direct internet access), then compute node jobs find the local cache and skip the network call.

### 5a. Check Disk Quota

Model checkpoints are ~10 GB each. Check you have space:

```bash
showquota
du -sh $SCRATCH/.cache/ 2>/dev/null
du -sh $SCRATCH/openpi_data/ 2>/dev/null
```

If quota is tight, clean old caches:
```bash
# Remove stale HF datasets Arrow caches
find $SCRATCH -name "*.arrow" -path "*/cache/*" -delete 2>/dev/null
# Remove old checkpoint experiments
rm -rf $SCRATCH/openpi/checkpoints/*/validate/
```

### 5b. Pre-Download PaliGemma Tokenizer

```bash
cd $SCRATCH/openpi
export UV_FROZEN=1
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface

uv run python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')
print('Tokenizer cached successfully')
"
```

Verify: `ls $HF_HOME/hub/models--google--paligemma-3b-pt-224/`

### 5c. Pre-Download Model Checkpoint

Download only the checkpoint for the model you plan to train. Each is ~10 GB, takes 10-30 minutes.

```bash
cd $SCRATCH/openpi
export UV_FROZEN=1
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data

# Pi0.5-DROID (recommended first — used in Training Run 1)
uv run python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi05_droid/params')
print('pi05_droid checkpoint cached')
"
```

Verify: `ls $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid/params/`

**Other model checkpoints** (download as needed):
```bash
# Pi0 base (~10 GB)
uv run python -c "from openpi.shared.download import maybe_download; maybe_download('gs://openpi-assets/checkpoints/pi0_base/params')"

# Pi0-FAST base (~10 GB)
uv run python -c "from openpi.shared.download import maybe_download; maybe_download('gs://openpi-assets/checkpoints/pi0_fast_base/params')"

# Pi0.5 base (~10 GB)
uv run python -c "from openpi.shared.download import maybe_download; maybe_download('gs://openpi-assets/checkpoints/pi05_base/params')"
```

**Why login node works**: Login nodes (grace4, grace5) have direct internet access. The `maybe_download` function caches to `$OPENPI_DATA_HOME/openpi-assets/checkpoints/`. Once cached, `maybe_download` returns the local path without any network call.

---

## Step 6: Quick Validation (Before Real Training)

Run a 5-step training test to verify everything works. **Model checkpoint and tokenizer must already be cached** (Step 5) — compute nodes cannot download them.

### Option A: Interactive GPU Session

```bash
srun --gres=gpu:1 --mem=64G --time=00:30:00 --partition=gpu --pty bash

# Once on the GPU node:
cd $SCRATCH/openpi
export UV_FROZEN=1

uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name validate \
    --num-train-steps 5 \
    --no-wandb-enabled \
    --overwrite

# Should complete in ~2-3 min, print loss values (~1.4)
```

### Option B: SLURM Batch Job

Create `$SCRATCH/openpi/validate.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ur5e_validate
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=validate_%j.log

## Environment (SLURM non-interactive bash skips ~/.bashrc — must export here)
module load WebProxy
export UV_FROZEN=1
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /scratch/user/changliu.chris/openpi

uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name validate \
    --num-train-steps 5 \
    --no-wandb-enabled \
    --overwrite
```

```bash
sbatch validate.slurm
```

---

## Step 7: Training

### LoRA Fine-Tuning (Single GPU)

Create `$SCRATCH/openpi/train_ur5e.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ur5e_lora
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=train_%j.log

## Environment (SLURM non-interactive bash skips ~/.bashrc — must export here)
module load WebProxy
export UV_FROZEN=1
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

cd /scratch/user/changliu.chris/openpi

# ===== CHANGE THESE TWO LINES =====
CONFIG="pi05_droid_ur5e_planner_lora_10hz"
EXP_NAME="planner_lora_10hz_grace_v1"
# ===================================

uv run scripts/train.py $CONFIG \
    --exp-name $EXP_NAME \
    --project-name ur5e-finetuning \
    --num-train-steps 50000 \
    --batch-size 32 \
    --fsdp-devices 1 \
    --overwrite
```

```bash
sbatch train_ur5e.slurm
```

### Full Finetune (Multi-GPU)

Grace has max 2 GPUs per node. For full finetune (batch_size=256):

```bash
#!/bin/bash
#SBATCH --job-name=ur5e_full
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=train_full_%j.log

## Environment (SLURM non-interactive bash skips ~/.bashrc — must export here)
module load WebProxy
export UV_FROZEN=1
export UV_CACHE_DIR=/scratch/user/changliu.chris/.cache/uv
export HF_HOME=/scratch/user/changliu.chris/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/user/changliu.chris
export OPENPI_DATA_HOME=/scratch/user/changliu.chris/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

cd /scratch/user/changliu.chris/openpi

CONFIG="pi05_droid_ur5e_e2e_30hz"
EXP_NAME="e2e_full_v1"

uv run scripts/train.py $CONFIG \
    --exp-name $EXP_NAME \
    --fsdp-devices 2 \
    --overwrite
```

**Note**: Grace max is 2 GPUs/node. If batch_size=256 causes OOM with 2 GPUs, override:
```bash
uv run scripts/train.py $CONFIG --exp-name $EXP_NAME --fsdp-devices 2 --batch-size 64 --overwrite
```

### SLURM Job Management

```bash
# Submit
sbatch train_ur5e.slurm

# Check your jobs
squeue -u changliu.chris

# Cancel a job
scancel <JOBID>

# View output log while running
tail -f train_*.log

# Check quota
showquota
```

---

## Step 8: Resume / Continue Training

Replace `--overwrite` with `--resume` and increase `--num-train-steps`:

```bash
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz --exp-name planner_v1 --num-train-steps 50000 --resume
```

**Three modes:**
- `--overwrite` = delete old checkpoints, start fresh
- `--resume` = resume from latest checkpoint
- **Neither** = error if checkpoint dir exists (`FileExistsError`)

Checkpoints are saved to `$SCRATCH/openpi/checkpoints/<config_name>/<exp_name>/`.

### CLI Overrides

```bash
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name custom_run \
    --num-train-steps 50000 \
    --batch-size 4 \
    --lr-schedule.peak-lr 1e-4 \
    --overwrite
```

---

## Config Quick Reference

### Recommended Starting Configs (LoRA, 1 GPU)

| Config | Model | Dataset | Use Case |
|--------|-------|---------|----------|
| `pi05_droid_ur5e_planner_lora_10hz` | Pi0.5-DROID | planner 10hz | Fastest iteration |
| `pi05_droid_ur5e_e2e_lora_10hz` | Pi0.5-DROID | e2e 10hz | Full task training |
| `pi05_droid_ur5e_correction_lora_10hz` | Pi0.5-DROID | correction 10hz | Grasp recovery |
| `pi0_ur5e_planner_lora_30hz` | Pi0 | planner 30hz | Pi0 baseline |
| `pi0_fast_ur5e_e2e_lora_30hz` | Pi0-FAST | e2e 30hz | Autoregressive baseline |

### All 52 Config Names

```
# Original backward-compat (all use ur5e_planner_30hz):
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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: ChangChrisLiu/ur5e_*` | `HF_LEROBOT_HOME` must point to PARENT of `ChangChrisLiu/` — should be `/scratch/user/changliu.chris` |
| `ModuleNotFoundError: openpi.policies.ur5e_policy` | `ur5e_policy.py` not uploaded to `src/openpi/policies/` |
| `ModuleNotFoundError: openpi.training.misc.polaris_config` | Server OpenPI is outdated — run `git pull origin main` (Step 0) |
| `uv` dependency resolution error (dlimp/tensorflow) | `export UV_FROZEN=1` — skips re-resolution |
| `Config 'xxx' not found` | `config.py` not updated — re-upload from local |
| `No norm stats found` | Run `compute_norm_stats.py` for that config, or create symlink |
| OOM during LoRA | Reduce batch size: `--batch-size 4` |
| OOM during full finetune | Use `--fsdp-devices 2 --batch-size 64` |
| `Unable to initialize backend 'cuda'` | Not on a GPU node — use `srun --gres=gpu:1` or submit via `sbatch` |
| numpy version conflict | `uv pip install numpy==1.26.4` (already fixed) |
| Slow first run | First run downloads ~10 GB base checkpoint from GCS to `$OPENPI_DATA_HOME`. This is a one-time cost per model. |
| GCS checkpoint download fails on compute node | `gcsfs`/`aiohttp` ignores `http_proxy` env vars. **Pre-download on login node** (Step 5) — `maybe_download` caches locally, compute nodes use cache |
| `OSError: paligemma_tokenizer` not found | Tokenizer not cached. Pre-download on login node: `uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"` |
| `Disk quota exceeded` during training | HF datasets Arrow cache filling scratch. Set `HF_HOME` to scratch, clean: `find $SCRATCH -name '*.arrow' -path '*/cache/*' -delete` |
| Wandb fails to connect | `module load WebProxy` in SLURM script — wandb uses `requests` which respects proxy (unlike gcsfs) |
| Job killed (time limit) | Increase `--time` in SLURM. Resume with same `--exp-name` + `--resume` flag. |
| `FileExistsError` on checkpoint dir | Must use `--overwrite` (start fresh) or `--resume` (continue). Omitting both = error. |
| `showquota` shows full | Clean old checkpoints: `rm -rf $SCRATCH/openpi/checkpoints/<old_exp>` |
| Login node: no GPU | Grace login nodes (grace4, grace5) have no GPUs. Use `srun` or `sbatch` for GPU work. |

---

## Execution Order Summary

```
0. git pull origin main + uv sync on server             (Step 0)
1. scp 4 custom files + 4 asset dirs from local         (Step 1)
2. Set environment variables in ~/.bashrc               (Step 2)
3. Verify upload: Python import test on login node      (Step 3)
4. Submit norm stats SLURM job                          (Step 4)
5. After stats complete: create symlinks                (Step 4)
6. Pre-download tokenizer + checkpoint on login node    (Step 5) ← CRITICAL
7. Quick 5-step validation (interactive or batch)       (Step 6)
8. Submit real training jobs                            (Step 7)
```
