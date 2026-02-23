# Training Run 1: Pi0.5-DROID LoRA x 3 Targets @ 10hz

Three sequential LoRA fine-tuning runs using Pi0.5-DROID on all three dataset targets at 10hz. Run locally first (immediate), then on GRACE server as backup.

## Training Matrix

| # | Config Name | Dataset | Frames | What It Learns |
|---|------------|---------|--------|----------------|
| 1 | `pi05_droid_ur5e_planner_lora_10hz` | `ur5e_planner_10hz` | 87,426 | Teleop approach — when to hand off to skill |
| 2 | `pi05_droid_ur5e_e2e_lora_10hz` | `ur5e_e2e_10hz` | 222,016 | Full task — all 4 phases autonomously |
| 3 | `pi05_droid_ur5e_correction_lora_10hz` | `ur5e_correction_10hz` | 29,109 | Grasp recovery after failed grasp |

Planner and E2E use 529 episodes; correction uses 527 episodes. All 10hz, batch_size=32, 30,000 steps each.

## Timing Estimates (30k steps each, 1 GPU, sequential)

| Machine | GPU | Per Run | 3 Runs Total |
|---------|-----|---------|--------------|
| **Local desktop** | RTX 5090 | ~12-15 hrs | ~36-45 hrs |
| **GRACE server** | Cluster GPU | ~12-15 hrs | ~36-45 hrs |

30k steps is conservative — leaves room to resume to 50k+ if loss is still decreasing.

---

# Part A: Local Desktop (RTX 5090)

## A.1 Prerequisites

Everything is already set up locally from previous sessions:

```bash
cd ~/openpi

# Verify
ls src/openpi/policies/ur5e_policy.py                    # policy file
ls assets/pi05_droid_ur5e_planner_lora_10hz/              # norm stats
ls assets/pi05_droid_ur5e_e2e_lora_10hz/
ls assets/pi05_droid_ur5e_correction_lora_10hz/
ls ~/lerobot_datasets/ChangChrisLiu/ur5e_planner_10hz/    # datasets
ls ~/lerobot_datasets/ChangChrisLiu/ur5e_e2e_10hz/
ls ~/lerobot_datasets/ChangChrisLiu/ur5e_correction_10hz/
```

## A.2 Environment Setup (One-Time)

Add these to `~/.bashrc` if not already present:

```bash
# CUDA 13.0 (must be on PATH for JAX/XLA to find nvcc and libs)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# OpenPI uv venv nvidia libs (cudnn, nccl, cublas, etc.)
# Fixes: ImportError: libcudnn.so.9: cannot open shared object file
export LD_LIBRARY_PATH="$(find ~/.cache/uv/archive-v0/ -maxdepth 4 -path '*/nvidia/*/lib' -type d 2>/dev/null | grep -v triton | grep -v tensorflow | tr '\n' ':')$LD_LIBRARY_PATH"

# LeRobot dataset location (OpenPI reads from here)
export HF_LEROBOT_HOME="$HOME/lerobot_datasets"

# JAX GPU memory — allow 90% of VRAM (prevents OOM during training)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Wandb logging (v1 key format — must use env var, `wandb login` rejects v1 keys)
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

Then: `source ~/.bashrc`

**Why each line matters:**

| Variable | What It Fixes |
|----------|--------------|
| `PATH` + `LD_LIBRARY_PATH` (cuda-13.0) | JAX/XLA needs CUDA toolkit on PATH |
| `LD_LIBRARY_PATH` (uv nvidia libs) | `ImportError: libcudnn.so.9` — uv installs nvidia wheels in its cache, not on system LD path |
| `HF_LEROBOT_HOME` | OpenPI looks for datasets at `$HF_LEROBOT_HOME/ChangChrisLiu/ur5e_*` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Without this, JAX pre-allocates only ~75% VRAM → OOM on large batches |
| `WANDB_API_KEY` | Wandb v1 keys are 86 chars; `wandb login` CLI rejects anything != 40 chars. Env var bypasses validation |

## A.3 Run Training (One at a Time)

Open a terminal and run each config sequentially. Ctrl+C anytime — checkpoints saved every 1,000 steps.

**Important**: Do NOT use inline env vars or multi-line `\` continuations when pasting into terminal — they break. The env vars are set in `~/.bashrc` above.

### Run 1: Planner

```bash
cd ~/openpi
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz --exp-name planner_v1 --project-name ur5e-finetuning --num-train-steps 30000 --overwrite
```

### Run 2: End-to-End

```bash
cd ~/openpi
uv run scripts/train.py pi05_droid_ur5e_e2e_lora_10hz --exp-name e2e_v1 --project-name ur5e-finetuning --num-train-steps 30000 --overwrite
```

### Run 3: Correction

```bash
cd ~/openpi
uv run scripts/train.py pi05_droid_ur5e_correction_lora_10hz --exp-name correction_v1 --project-name ur5e-finetuning --num-train-steps 30000 --overwrite
```

## A.4 Monitor

**Terminal**: Training prints `Step N: loss=X.XXXX, grad_norm=X.XX, param_norm=X.XX` every 100 steps.

**Wandb**: Open https://wandb.ai/losttemplor-texas-a-m-university/ur5e-finetuning to see live loss curves, gradient norms, and sample images.

## A.5 Resume / Extend Training

After 30k steps complete, if loss is still decreasing:

```bash
# Resume planner from 30k to 50k (--resume, same --exp-name)
cd ~/openpi
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz --exp-name planner_v1 --project-name ur5e-finetuning --num-train-steps 50000 --resume
```

You can also stop with Ctrl+C mid-training and resume later with the same command. It picks up from the latest saved checkpoint.

**Key rule — three modes:**
- `--overwrite` = delete old checkpoints, start fresh
- `--resume` = resume from latest checkpoint
- **Neither** = error if checkpoint dir already exists (`FileExistsError`)

**Important**: All env vars (`HF_LEROBOT_HOME`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, etc.) are set in `~/.bashrc` from Step A.2. Do NOT use inline env vars on the command line — they break when pasting.

## A.6 Checkpoints

```
~/openpi/checkpoints/
├── pi05_droid_ur5e_planner_lora_10hz/planner_v1/
│   ├── 1000/  ├── 2000/  ├── ...  └── 30000/
├── pi05_droid_ur5e_e2e_lora_10hz/e2e_v1/
│   ├── 1000/  ├── 2000/  ├── ...  └── 30000/
└── pi05_droid_ur5e_correction_lora_10hz/correction_v1/
    ├── 1000/  ├── 2000/  ├── ...  └── 30000/
```

Every 1,000-step checkpoint is kept. You can serve any of them to test on the robot.

## A.7 Serve Policy for Robot Testing

```bash
cd ~/openpi

# Serve planner (or swap config/dir for e2e/correction)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_droid_ur5e_planner_lora_10hz \
    --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000
```

**Client code** (from your robot control script):
```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer({
    "observation/image": base_rgb,           # (256,256,3) uint8
    "observation/wrist_image": wrist_rgb,    # (256,256,3) uint8
    "observation/state": state_7d,           # [q0-q5, gripper/255] float32
    "prompt": "Pick up the CPU",
})
actions = result["actions"]  # (10, 7) absolute joint positions
```

---

# Part B: HPRC GRACE Server

## B.1 Prerequisites

Complete ALL steps in [`SERVER_SETUP_HPRC.md`](SERVER_SETUP_HPRC.md) first:

0. `git pull origin main` + `uv sync` on server (brings in upstream deps like `misc/` configs)
1. Upload 4 custom files + 4 asset dirs from local `~/openpi/` to `$SCRATCH/openpi/`
2. Set environment variables in `~/.bashrc` (including `UV_FROZEN=1` and `WANDB_API_KEY`)
3. Verify datasets at `$HF_LEROBOT_HOME/ChangChrisLiu/`
4. Verify norm stats at `$SCRATCH/openpi/assets/`
5. **Pre-download tokenizer + checkpoint on login node** (Step 5 in SERVER_SETUP — gcsfs ignores proxy, must pre-cache)
6. Pass the Python config import test

## B.2 Quick Validation (5-step test)

```bash
# Request interactive GPU session
srun --gres=gpu:1 --mem=64G --time=00:30:00 --partition=gpu --pty bash

cd $SCRATCH/openpi
export UV_FROZEN=1

uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name validate \
    --num-train-steps 5 \
    --no-wandb-enabled \
    --overwrite

# Expected: loss ~1.4-1.5, completes in ~2-3 min
exit
```

## B.3 Create SLURM Job Script

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
echo "ALL 3 RUNS COMPLETE — $(date)"
SLURM
```

**Replace `<YOUR_WANDB_API_KEY>` before submitting.**

## B.4 Submit

```bash
cd $SCRATCH/openpi
sbatch train_all_3.slurm
```

## B.5 Monitor

```bash
# Job status
squeue -u changliu.chris

# Live log
tail -f $SCRATCH/openpi/train_all_3_*.log

# Wandb dashboard (same project as local runs):
# https://wandb.ai/losttemplor-texas-a-m-university/ur5e-finetuning
```

## B.6 Download Checkpoints to Local Machine

After GRACE training completes, download to local for inference:

```bash
# Run from LOCAL machine:
SERVER=changliu.chris@grace.hprc.tamu.edu
REMOTE=$SERVER:/scratch/user/changliu.chris/openpi/checkpoints

scp -r $REMOTE/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1_grace/30000

scp -r $REMOTE/pi05_droid_ur5e_e2e_lora_10hz/e2e_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_e2e_lora_10hz/e2e_v1_grace/30000

scp -r $REMOTE/pi05_droid_ur5e_correction_lora_10hz/correction_v1/30000 \
    ~/openpi/checkpoints/pi05_droid_ur5e_correction_lora_10hz/correction_v1_grace/30000
```

Note: saved to `*_grace/` subdirs to avoid overwriting local checkpoints.

## B.7 Resume on GRACE

Same as local — replace `--overwrite` with `--resume` and increase `--num-train-steps`:

```bash
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz \
    --exp-name planner_v1 \
    --project-name ur5e-finetuning \
    --num-train-steps 50000 \
    --resume
```

---

# Wandb Dashboard

Both local and GRACE runs log to the same wandb project:

**URL**: https://wandb.ai/losttemplor-texas-a-m-university/ur5e-finetuning

**Metrics logged every 100 steps:**
- `loss` — training loss (should decrease from ~1.4 to ~0.5-1.0)
- `grad_norm` — gradient magnitude (should be stable)
- `param_norm` — parameter magnitude
- Camera sample images at step 0 (sanity check)

Runs from local and GRACE appear side-by-side. Use wandb run names (`planner_v1`, `e2e_v1`, `correction_v1`) to identify them.

---

# Troubleshooting

| Issue | Fix |
|-------|-----|
| **Local**: `ImportError: libcudnn.so.9` | `LD_LIBRARY_PATH` not set — add the uv nvidia libs line to `~/.bashrc` (see A.2) |
| **Local**: OOM on RTX 5090 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` in `~/.bashrc`, or `--batch-size 16` |
| **Local**: `ModuleNotFoundError` | Must run from `cd ~/openpi` |
| **Local**: Slow first run | Downloads ~10 GB base checkpoint from GCS (one-time) |
| **Local**: Multi-line `\` commands fail | Terminal paste breaks multi-line. Use single-line commands (see A.3) |
| **Local**: Inline env vars fail | Don't use `VAR=val command`. Export in `~/.bashrc` instead (see A.2) |
| **GRACE**: `uv` resolution error | `export UV_FROZEN=1` |
| **GRACE**: `Unable to initialize backend 'cuda'` | Not on GPU node — use `srun --gres=gpu:1` or `sbatch` |
| **GRACE**: `FileNotFoundError` for dataset | `HF_LEROBOT_HOME` must be `/scratch/user/changliu.chris` (parent of `ChangChrisLiu/`) |
| **GRACE**: GCS checkpoint download fails/hangs | `gcsfs`/`aiohttp` ignores `http_proxy`. Pre-download on login node (see SERVER_SETUP Step 5) |
| **GRACE**: `Disk quota exceeded` | HF datasets cache filling scratch. `find $SCRATCH -name '*.arrow' -path '*/cache/*' -delete` |
| **Both**: Wandb not logging | Check `WANDB_API_KEY` is set |
| **Both**: Loss not decreasing | Check norm stats exist. Try `--lr-schedule.peak-lr 1e-5` |
| **Both**: Want to stop and continue later | Ctrl+C (local) or `scancel` (GRACE). Resume with same `--exp-name` + `--resume` flag |
| **Both**: Want more than 30k steps | Resume: `--num-train-steps 50000 --resume` |
| **Both**: `FileExistsError` on checkpoint dir | Must use either `--overwrite` (start fresh) or `--resume` (continue). Omitting both = error. |
