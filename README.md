# DataCollection: VLA Data Collection Pipeline for UR5e

A teleoperation data collection system for training Vision-Language-Action (VLA) models, built on the [GELLO](https://github.com/wuphilipp/gello_software) framework. Features a distributed multi-process architecture with dual-camera capture, HOSAS joystick control, automated skill execution with path blending and grasp verification, and a unified phase-labeled recording format that produces training data for three model architectures: **OpenVLA**, **OpenVLA-OFT**, and **OpenPI**.

## System Overview

Each data collection episode records a single, continuous frame stream annotated with phase labels:

| Phase | Description | When |
|-------|-------------|------|
| `armed` | Recording ready, waiting for input | After pressing record button (no frames captured) |
| `teleop` | Human joystick control (approach) | First joystick input after arming |
| `skill` | Autonomous skill execution | Triggered by skill button |
| `correction` | Human recovery after failed grasp | Activated on joystick input after grasp failure |
| `skill_resume` | Skill resumes absolute waypoints | Skill button pressed after correction |

From this single recording, **three training datasets** are derived during conversion. Each dataset (called a **"target"** in the CLI) selects different phases to train a different policy:

| Target (`--target`) | CLI Name | Phases Included | Stop Signal | What the Trained Model Learns |
|----------------------|----------|----------------|-------------|-------------------------------|
| **End-to-End** | `e2e` | all 4 phases | none | The full task from approach to placement — a single policy does everything |
| **Planner** | `planner` | teleop only | 3 frames appended | When to hand off to a pre-programmed skill — outputs a "call skill now" stop signal when the approach is complete |
| **Correction** | `correction` | correction only | 3 frames appended | How to recover after a failed grasp — repositions the gripper so the skill can retry |

Stop signals (3 copies of last frame with gripper=255) are **not stored** in raw recordings — they are synthesized during conversion.

## Hardware

| Component | Model | Connection |
|-----------|-------|------------|
| Robot | UR5e | RTDE at `10.125.144.209` |
| Gripper | Robotiq 2F-85 | Socket port 63352 (positions 0-255) |
| Wrist Camera | Intel RealSense D435i | USB, 1280x720 @ 30Hz (RGB) |
| Base Camera | Luxonis OAK-D Pro | USB, 1280x720 @ 30Hz (RGB) |
| Controller | Thrustmaster SOL-R2 HOSAS | USB, dual flight sticks |

## Architecture

Three independent processes communicate over ZMQ:

```
Terminal 1 ─ Robot Server
  ├── Port 6001: Control server (moveL, speedL, move_linear_path, speed_stop, stop_linear)
  └── Port 6002: Read-only observation server (get_observations, get_tcp_pose_raw)

Terminal 2 ─ Camera Publishers (PUB/SUB)
  ├── Port 5000: Wrist camera (RealSense D435i)
  └── Port 5001: Base camera (OAK-D Pro)

Terminal 3 ─ Data Collection Pipeline
  └── Connects to T1 + T2, runs joystick agent, skill executor, episode buffer, dataset writer
```

The dual-port robot architecture (6001 control + 6002 read-only) allows observation polling during blocking skill execution. Both servers wrap the same robot instance but access different RTDE interfaces (Control vs Receive), which is thread-safe.

## Installation

```bash
git clone https://github.com/ChangChrisLiu/DataCollection.git
cd DataCollection
```

### Conda Environment

```bash
conda create -n tele python=3.11 -y
conda activate tele

git submodule init && git submodule update
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
```

### Additional Dependencies

```bash
pip install ur_rtde       # UR robot control
pip install depthai       # OAK-D camera
pip install pygame        # Joystick input
pip install pyrealsense2  # RealSense camera
pip install tyro          # CLI argument parsing
pip install scipy         # Rotation transforms
```

For data conversion (see [Data Conversion](#data-conversion)):
```bash
# RLDS / OpenVLA
pip install tensorflow tensorflow-datasets tensorflow-hub
pip install sentence-transformers  # Universal Sentence Encoder language embeddings

# LeRobot / OpenPI
pip install lerobot
```

> **Note (PyTorch version):** `pip install lerobot` pulls in torch < 2.8. If your GPU requires a newer PyTorch (e.g. RTX 5090 needs CUDA 13.0 / torch 2.10+), install lerobot first, then force-reinstall your required PyTorch version. The pip resolver will report version-pin warnings, but the LeRobot dataset creation API (`LeRobotDataset.create()`, `add_frame()`, `save_episode()`) works correctly with both the bundled torch and newer versions (tested with torch 2.10.0+cu130).

---

## Data Collection Pipeline

### Step 0: Calibrate Camera Settings (One-Time)

Lock exposure, white balance, and gain to prevent out-of-distribution (OOD) issues across episodes. Run once before each session (or when lighting changes).

**Auto mode** (recommended):

```bash
python scripts/calibrate_cameras.py --auto
```

Opens both cameras with auto-exposure/WB, waits 20 seconds for stabilization, snapshots all parameters, saves to `configs/camera_settings.json`.

Options:
- `--warmup 30` — longer stabilization time (default: 20s)
- `--output path/to/settings.json` — custom output path

**Manual mode** (with live preview):

```bash
python scripts/calibrate_cameras.py
```

Shows a 2x2 live preview (RGB + depth for each camera). Press SPACE to snapshot, ESC to exit.

### Step 1: Terminal 1 - Robot Server

```bash
python experiments/launch_nodes.py --robot ur --robot-ip 10.125.144.209
```

Starts dual ZMQ servers:
- Port 6001: Full control (moveL, move_linear_path, speedL, servoJ, speed_stop, stop_linear, set_freedrive_mode)
- Port 6002: Read-only observations (get_observations, get_tcp_pose_raw, get_joint_state)

### Step 2: Terminal 2 - Camera Servers

```bash
python experiments/launch_camera_nodes.py --camera-settings configs/camera_settings.json
```

Starts async PUB/SUB camera publishers at 30Hz. The `--camera-settings` flag loads and applies the fixed camera parameters captured in Step 0.

> **Warning:** Running without `--camera-settings` uses auto-exposure/WB, which causes inconsistent lighting across episodes.

### Step 3: Terminal 3 - Data Collection

```bash
python experiments/run_collection.py
```

Optional arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/vla_dataset` | Output directory for episodes |
| `--control-hz` | `200` | Joystick control loop rate (Hz) |
| `--record-hz` | `30` | Recording frame rate (Hz) |
| `--cpu-skill-csv` | `CPU_Skills.csv` | Path to CPU extraction skill CSV |
| `--ram-skill-csv` | `RAM_Skills.csv` | Path to RAM extraction skill CSV |
| `--cpu-relative-count` | `20` | Number of relative waypoints in CPU skill |
| `--ram-relative-count` | `5` | Number of relative waypoints in RAM skill |
| `--skill-move-speed` | `0.1` | moveL speed during skill (m/s) |
| `--skill-move-accel` | `0.04` | moveL acceleration during skill (m/s^2) |
| `--image-size` | `256` | Resize captured images to NxN pixels |
| `--no-cameras` | `false` | Robot-only mode for testing |
| `--verbose` | `false` | Print joystick debug info |

---

## Controller Mapping

### Thrustmaster SOL-R2 HOSAS

```
LEFT STICK                              RIGHT STICK
  Stick X/Y   -> TCP X/Y translation      Stick Y      -> TCP Z (push/pull)
  Slider      -> Speed gain (0.1x-1.0x)   Twist (ax5)  -> TCP Rz rotation
  Mini Y      -> Gripper open/close        Mini X (ax3) -> TCP Ry rotation
                                           Mini Y (ax4) -> TCP Rx rotation

LEFT BUTTONS                            RIGHT BUTTONS
  Btn 25: Start recording                 Btn 25: Interrupt active skill
  Btn 34: Home + stop + save              Btn 34: Trigger CPU skill
  Btn 38: Vertical reorient               Btn 38: Trigger RAM skill
  Btn 16: Interrupt active skill          Btn 8:  90° CW rotation (fixed)
                                          Btn 9:  90° CCW rotation (fixed)
```

The left slider controls a speed gain multiplier: fully up = 100% speed, fully down = 10% speed. All axes have a 0.05 deadzone and are calibrated at startup.

**90° Auto-Rotation** (Btn 8/9): Triggers a fixed 90° TCP rotation around the Z axis at 0.393 rad/s (completes in ~4 seconds). The rotation speed is independent of the slider gain. Pressing the button again during an active rotation is ignored to prevent over-rotation. All other controls (translation, gripper, Rx/Ry, twist) remain active during the rotation.

---

## Episode Workflow

### Normal Data Collection (Per Episode)

1. **Position the robot** near the target object using joystick teleop
2. **Press Left Btn 25** — recording is **armed** (ready, no frames captured yet)
3. **Move joystick** — recording starts at 30Hz (camera-gated), phase: `teleop`
4. **Teleoperate** to approach and align the gripper with the component
5. **Press Right Btn 34** (CPU) or **Right Btn 38** (RAM) to trigger the skill:
   - Phase transitions to `skill`
   - Skill executes autonomously with path blending and 30Hz concurrent recording
   - **Grasp verification** runs automatically (see below)
6. **After skill completes** — episode auto-saves and robot moves home automatically
   - **Left Btn 34** (Home) available as fallback if needed
7. **Repeat** from step 1 for the next episode

### Grasp Verification (Automatic)

Each skill CSV contains a verification waypoint that lifts the gripper 5cm after the grasp close. The executor checks the actual gripper position against a per-skill threshold:

| Skill | Grasp Close | Threshold | Margin | Pass Condition |
|-------|-------------|-----------|--------|----------------|
| CPU | 168 | 158 | 10 | actual < 158 (object blocking fingers) |
| RAM | 229 | 226 | 3 | actual < 226 (object blocking fingers) |

- **Pass**: Object is held. Skill continues normally with **continuous drop monitoring** enabled (see below).
- **Fail**: Fingers closed on nothing (actual >= threshold). Recording pauses automatically and the pipeline waits for correction.

### Continuous Drop Detection

After grasp verification succeeds, the executor continuously monitors the actual gripper position during all remaining waypoint motions at 50Hz. If the gripper position reaches or exceeds the grasp threshold, a drop is detected:

- The Robotiq 2F-85 maintains commanded force — with an object gripped, actual position stays below threshold (object resists closure). If the object falls out, the fingers close further past the threshold.
- On drop detection: robot stops immediately, `on_grasp_failed()` fires, and the pipeline enters correction phase (same flow as grasp failure).
- Drop monitoring is also re-enabled during `skill_resume` after correction, using the same skill's threshold.
- The `grasp_info` dict includes `"drop_detected": True` to distinguish drops from manual interrupts.

### Correction Flow (After Grasp Failure)

1. Grasp verification fails — recording stops, message: `"GRASP FAILED — provide correction with joystick"`
2. Gripper speed stays at 128/255 (slow) for precise correction control
3. Move joystick to correct position — recording resumes with phase: `correction`
4. Press the **same skill button** to resume:
   - Phase transitions to `skill_resume`
   - Only absolute (base-frame) waypoints execute (relative already completed)
5. After skill completes — auto-saves and moves home

### Manual Interrupt Flow

1. During skill execution, press **Left Btn 16** or **Right Btn 25** to interrupt
2. Robot stops immediately, phase transitions to `correction`
3. Manually correct position with joystick (correction frames recorded)
4. Press the **same skill button** to resume with absolute waypoints only
5. If correction is impossible, press **Home** to save what you have

---

## Skill System

Skills replay pre-recorded manipulation trajectories from CSV files with **path-based blending** for smooth motion.

### CSV Format

| Column | Description |
|--------|-------------|
| timestamp | Recording timestamp |
| j0-j5 | 6 joint angles (radians) |
| tcp_x-tcp_rz | TCP pose [x, y, z, rx, ry, rz] in base frame |
| gripper_pos | Gripper position (0-255) |
| skill_id | Skill identifier |
| image_file | Associated image path (or `verification_wp` for verification waypoints) |

### Relative vs Absolute Waypoints

Each skill CSV is split into two sections:

- **Relative section** (first N waypoints): Applied relative to the trigger TCP pose using SE(3) transforms. This makes the manipulation work regardless of where the robot is when the skill is triggered.
  - Transform: `T_target = T_trigger @ inv(T_skill_origin) @ T_waypoint`

- **Absolute section** (remaining waypoints): Used as-is in base-frame coordinates. Typically the transfer-to-destination phase.

### Path Blending

The **first segment** (approach, before any gripper change) uses `moveL(path)` with blend radii for smooth trajectories:

- **Default blend radius**: 0.002m (2mm)
- **First 2 waypoints**: 0.0001m (0.1mm) — sensitive approach
- **First/last of segment**: 0.0m (exact stop)

**After any gripper change** (grasp/release), subsequent segments use individual `moveL()` per waypoint instead of path blending. This prevents protective stops on short post-grasp segments (e.g. the 5cm verification lift) where path blending can fail due to position/load changes after gripping.

### Current Skills

| Skill | Button | CSV File | Relative Count | Total Waypoints | Grasp Close | Grasp Threshold | Path Mode |
|-------|--------|----------|----------------|-----------------|-------------|-----------------|-----------|
| CPU Extraction | Right 34 | `CPU_Skills.csv` | 20 | 23 | 168 | 158 | Path blending (first segment) |
| RAM Extraction | Right 38 | `RAM_Skills.csv` | 5 | 8 | 229 | 226 | Individual moveL (all segments) |

---

## Output Data Format

### Unified Episode Structure

```
data/vla_dataset/
  CPU_Extraction/                      <- subdirectory for CPU episodes
    episode_cpu_0218_143000/
      frame_0000.pkl   <- phase: "teleop"
      ...
      frame_0044.pkl   <- phase: "teleop"
      frame_0045.pkl   <- phase: "skill"
      ...
      frame_0055.pkl   <- phase: "skill" (grasp failed, recording paused)
      frame_0056.pkl   <- phase: "correction" (resumed on joystick input)
      ...
      frame_0065.pkl   <- phase: "correction"
      frame_0066.pkl   <- phase: "skill_resume"
      ...
      frame_0070.pkl   <- phase: "skill_resume"
      episode_meta.json
  RAM_Extraction/                      <- subdirectory for RAM episodes
    episode_ram_0218_150000/
      frame_0000.pkl
      ...
      episode_meta.json
  episode_cpu_0217_100000/             <- episodes at root level also discovered
    frame_0000.pkl
    ...
```

Episode discovery (`discover_episodes()`) recurses into subdirectories automatically. Both root-level and nested episodes are found and merged into a single dataset.

### Frame Format (`.pkl`)

Each `.pkl` frame contains:

```python
{
    "timestamp": float,                    # Unix timestamp
    "phase": str,                          # "teleop" | "skill" | "correction" | "skill_resume"
    "joint_positions": [j0, ..., j5],      # 6 joint angles (radians)
    "tcp_pose": [x, y, z, rx, ry, rz],    # TCP pose, UR rotation vector
    "gripper_pos": int,                    # 0-255
    "wrist_rgb": np.ndarray,              # (256, 256, 3) uint8
    "wrist_timestamp": float,
    "base_rgb": np.ndarray,               # (256, 256, 3) uint8
    "base_timestamp": float,
}
```

### Episode Metadata (`episode_meta.json`)

```json
{
    "skill_name": "cpu",
    "skill_outcome": "completed_after_correction",
    "grasp_verified": true,
    "grasp_commanded": 132,
    "grasp_actual": 85,
    "phase_counts": {"teleop": 45, "skill": 11, "correction": 10, "skill_resume": 5},
    "phase_segments": [
        {"phase": "teleop", "start": 0, "end": 44},
        {"phase": "skill", "start": 45, "end": 55},
        {"phase": "correction", "start": 56, "end": 65},
        {"phase": "skill_resume", "start": 66, "end": 70}
    ],
    "num_frames": 71,
    "fps": 30,
    "saved_at": "2026-02-18T14:30:15"
}
```

Skill outcomes: `"completed"`, `"completed_after_correction"`, `"no_skill"`, `"incomplete"`

---

## Data Conversion

Raw `.pkl` episodes are converted to model-specific formats locally, then transferred to the training server. This avoids NumPy version compatibility issues since TFRecords and LeRobot datasets use version-agnostic serialization.

### Conversion Scripts

There are two entry-point scripts — one per output format. You only ever run these two files:

| Script | Output Format | Used By | Output Location |
|--------|--------------|---------|-----------------|
| `scripts/convert_to_rlds.py` | RLDS TFRecords | OpenVLA, OpenVLA-OFT | `~/tensorflow_datasets/ur5e_vla_<target>_<fps>hz/` |
| `scripts/convert_to_lerobot.py` | LeRobot v2.1 | OpenPI | `~/lerobot_datasets/<repo-id>/` |

Both scripts share the same processing logic from `scripts/conversion_utils.py` (episode discovery, phase filtering, downsampling, no-op removal, trigger/stop signal synthesis).

The RLDS script additionally depends on three small **TFDS builder modules** — `scripts/ur5e_vla_e2e/`, `scripts/ur5e_vla_planner/`, `scripts/ur5e_vla_correction/` — but you never run or touch these directly. They are ~20-line config files that each set flags (which phases to include, whether to add stop signals, etc.) and inherit all actual logic from `scripts/rlds_builder_base.py`. They exist as separate directories because the TFDS framework requires each dataset to be its own Python package. `convert_to_rlds.py` dynamically imports the correct builder based on `--target`.

```
scripts/
├── convert_to_rlds.py          ← run this for RLDS
│     └── imports builder from:
│         ├── ur5e_vla_e2e/         (PHASE_FILTER=all, no stop signal)
│         ├── ur5e_vla_planner/     (PHASE_FILTER=teleop, stop+trigger)
│         └── ur5e_vla_correction/  (PHASE_FILTER=correction, stop+trigger+near-grasp)
│               └── all inherit from rlds_builder_base.py (schema + processing)
│
├── convert_to_lerobot.py       ← run this for LeRobot (self-contained)
│
└── conversion_utils.py         ← shared utilities (used by both)
```

Both conversion scripts accept two key arguments:

- **`--target`** selects which dataset to convert (`e2e`, `planner`, or `correction`). Each target filters different phases from the raw recording and applies different post-processing (see table above). The RLDS script also accepts `all` to run all three targets in sequence, producing three separate datasets (`ur5e_vla_e2e`, `ur5e_vla_planner`, `ur5e_vla_correction`); LeRobot must be run separately per target. Note: `--target all` means "all targets", not "all phases" — the target that includes all *phases* is `e2e`.
- **`--fps`** sets the **output frame rate** of the training dataset (5, 10, 15, or 30 Hz). Raw data is always recorded at 30Hz. Lower FPS means fewer frames per episode, which reduces dataset size and speeds up training. For example, `--fps 10` keeps every 3rd frame. Default is 30 (no downsampling).

During conversion, the following processing is applied **in this order** (ordering matters):

1. **Phase filtering** — select frames matching the target's phase set
2. **Downsampling** — reduce from 30Hz to the output FPS set by `--fps`
3. **No-op frame removal + trigger signal amplification** — differs by target:
   - **planner/correction**: split frames into body + tail (~0.5s). Remove no-op frames from the body only, then repeat the preserved tail as a trigger signal
   - **e2e**: remove no-op frames from the entire episode (no tail preservation, no trigger signal)
4. **Stop signal synthesis** — 3 copies of last frame with gripper=255 (planner/correction targets only)
5. **Action computation** — delta EEF for RLDS, absolute next-step joints for LeRobot
6. **Image resizing** — 256x256 for RLDS (configurable)

No-op removal (step 3) drops consecutive frames where all joint positions changed by less than 1e-4 radians — i.e. the robot wasn't moving. This is applied to **all three targets**, but for planner/correction the last ~0.5s (the "tail") is protected from removal because those frames carry the trigger signal.

The trigger signal (step 3) marks the moment the human finished the approach. Its frame count scales with `--fps` to maintain ~0.5 seconds of real time:

| `--fps` | Tail frames | Repeats | Total trigger frames | Stop frames |
|---------|-------------|---------|---------------------|-------------|
| 30      | 15          | 3       | 45                  | 3           |
| 15      | 7           | 2       | 14                  | 3           |
| 10      | 5           | 2       | 10                  | 3           |
| 5       | 2           | 1       | 2                   | 3           |

**Language instructions** are auto-detected per episode from the directory path (CPU vs RAM). No `--task` flag is needed. Each skill has a fixed description:
- **CPU**: "Extract the CPU from the Bracket by unlocking it first, then extract the CPU and place it inside the yellow square area, then back home."
- **RAM**: "Extract the RAM from the slot and place it inside the blue square area, then back home."

**Episode discovery** recurses into subdirectories (`CPU_Extraction/`, `RAM_Extraction/`) and produces a single merged dataset containing both CPU and RAM episodes.

> **Important:** Phase filtering happens **before** downsampling so each phase has consistent sampling. Downsampling happens **before** no-op removal because at lower FPS, per-frame deltas are larger and the joint-threshold (1e-4 rad) naturally catches fewer frames.

### RLDS Conversion (OpenVLA / OpenVLA-OFT)

Per-target settings (set in each builder module, inherited from `rlds_builder_base.py`):

| Target | Builder Name | Phase Filter | No-op Removal | Trigger Amp | Stop Signal |
|--------|-------------|-------------|---------------|-------------|-------------|
| `e2e` | `ur5e_vla_e2e` | all phases | entire episode | no | no |
| `planner` | `ur5e_vla_planner` | teleop only | body only (tail preserved) | yes | yes (3 frames) |
| `correction` | `ur5e_vla_correction` | correction only | body only (tail preserved) | yes | yes (3 frames) |

All three inherit from `scripts/rlds_builder_base.py` which defines the shared RLDS schema. State and actions use **EEF (end-effector) coordinates with Euler RPY angles**, not joint angles.

**E2E target notes:**
- Includes all 4 phases as a single continuous trajectory (no phase boundary markers)
- Actions (delta EEF) are smooth across phase boundaries because the robot state is physically continuous
- No-op frames removed from the entire episode (no tail preservation, no trigger signal)
- Intended for training a policy that learns the entire task autonomously — the model must learn from visual context when to transition between approach, grasp, and transfer behaviors
- Episodes that included correction (grasp failed then recovered) are labeled `success=True` if eventually completed — the planner target also trains on the initial (potentially bad) approach from these episodes

**Correction target** additionally extracts **near-grasp segments** from successful episodes (no correction phase) as supplementary training data. These segments cover the critical approach/extract/lift window of each skill:
- **CPU**: 19.8s–22.2s after skill start
- **RAM**: 0.2s–3.0s after skill start

Near-grasp segments have no trigger amplification, no no-op removal, and no stop signals — they teach pure manipulation behavior.

**RLDS Schema** (image size matches `--image-size`, default 256):
```
observation.image:          (N, N, 3)     uint8    Base camera RGB (JPEG)
observation.wrist_image:    (N, N, 3)     uint8    Wrist camera RGB (JPEG)
observation.state:          (8,)          float32  [x, y, z, roll, pitch, yaw, 0.0, gripper_0to1]
action:                     (6,)          float32  Delta EEF [dx, dy, dz, droll, dpitch, dyaw]
action_gripper:             (1,)          float32  Next frame's gripper (0=open, 1=closed)
language_instruction:       string                 Task description (auto-detected per episode)
language_embedding:         (512,)        float32  Universal Sentence Encoder
```

- `observation.state[7]` (gripper): current frame's normalized gripper position (0.0=open, 1.0=closed, robot convention)
- `action_gripper`: next frame's normalized gripper position (same convention). For trigger/stop signal frames: always 1.0 (closed)
- `action`: delta from current frame to next frame. For trigger/stop/last frames: zeros

**Camera Images — Same Dataset, Two Models:**

Each RLDS dataset stores **both** camera images (`observation.image` = base, `observation.wrist_image` = wrist). This is intentional — the same TFRecord files are used for both OpenVLA and OpenVLA-OFT fine-tuning, but each model consumes them differently:

| Model | Images Used | Config Mechanism | Why |
|-------|-------------|-----------------|-----|
| **OpenVLA** | Base camera only (`image_primary`) | Hardcoded `load_camera_views=("primary",)` in dataset loader | OpenVLA was pretrained exclusively on **third-person camera** images ([paper](https://arxiv.org/abs/2406.09246): *"manipulation datasets with at least one 3rd person camera"*). It is a single-image architecture — the wrist image data exists in the TFRecord but is never loaded or processed. |
| **OpenVLA-OFT** | Base + wrist (`image_primary` + `image_wrist`) | Set `--num_images_in_input 2` at training time | OFT extends the architecture to process multiple images through the fused SigLIP+DINOv2 backbone independently, then concatenates patches (256 patches per image → 512 total). With `--num_images_in_input 1` (default), it behaves like base OpenVLA. |

The `image_obs_keys` in `configs.py` maps RLDS keys to the data loader:
```python
"image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"}
```
- `primary` → `image_primary` (base/third-person camera) — always loaded
- `wrist` → `image_wrist` (wrist camera) — loaded only when OFT's `--num_images_in_input > 1`
- `secondary` → `None` (padding, not used)

This means: **you do NOT need separate datasets for OpenVLA vs OFT**. Build once, train with both. The unused wrist image adds no overhead to OpenVLA training because it is discarded at load time.

**Dataset Size vs Raw Data:**

Raw `.pkl` files store images as uncompressed numpy arrays (192 KB per 256x256x3 image × 2 cameras = 384 KB per frame). RLDS uses JPEG encoding (`encoding_format="jpeg"`), compressing each image to ~10-30 KB — a 6-15x reduction. Combined with phase filtering (planner keeps only teleop frames, correction keeps only correction frames) and no-op removal, RLDS datasets are typically 5-20x smaller than the raw data while containing all the same information.

**Commands:**

```bash
# Convert planner dataset (teleop approach only) at full 30Hz
# CUDA_VISIBLE_DEVICES="" avoids TF GPU errors on unsupported GPUs (see Known Issues)
CUDA_VISIBLE_DEVICES="" python scripts/convert_to_rlds.py \
    --target planner \
    --data-path data/vla_dataset

# Convert all three targets at 10Hz output
CUDA_VISIBLE_DEVICES="" python scripts/convert_to_rlds.py \
    --target all \
    --data-path data/vla_dataset \
    --fps 10

```

The default `--image-size 256` works for both base OpenVLA and OFT — the training pipeline resizes images internally to match the model's vision backbone (224x224 for base OpenVLA's `dinosiglip-vit-so-224px`, 256x256 for OFT). You do not need separate datasets at different image sizes.

> **TFDS Caching:** TFDS does not overwrite existing output. If you re-run conversion with different data, you must delete the existing dataset directory first:
> ```bash
> rm -rf ~/tensorflow_datasets/ur5e_vla_planner_10hz/
> python scripts/convert_to_rlds.py --target planner --data-path data/vla_dataset --fps 10
> ```
> Different targets and FPS variants (`ur5e_vla_planner_10hz`, `ur5e_vla_planner_30hz`, etc.) have separate directories and can coexist.

**Transfer to server:**

```bash
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner_10hz/ server:~/tensorflow_datasets/ur5e_vla_planner_10hz/
```

**Verify:**

```bash
python -c "
import tensorflow_datasets as tfds
b = tfds.builder('ur5e_vla_planner_10hz', data_dir='$HOME/tensorflow_datasets')
ds = b.as_dataset(split='train')
for traj in ds.take(1):
    for step in traj['steps']:
        print('image:', step['observation']['image'].shape)        # (256,256,3)
        print('wrist:', step['observation']['wrist_image'].shape)  # (256,256,3)
        print('state:', step['observation']['state'].numpy())      # [x,y,z,r,p,y,0,grip]
        print('action:', step['action'].numpy())                   # [dx,dy,dz,dr,dp,dy]
        print('gripper:', step['action_gripper'].numpy())          # [grip_0to1]
        print('lang:', step['language_instruction'].numpy())       # per-episode
        break
"
```

### LeRobot Conversion (OpenPI)

Unlike RLDS, LeRobot uses **joint angles** (not EEF) for state and action. OpenPI's `DeltaActions` transform converts absolute positions to deltas during training.

> **Critical: Use OpenPI's Python environment for conversion.** The `tele` conda env has LeRobot 0.4.3 (dataset format v3.0), but OpenPI uses LeRobot 0.1.0 (format v2.1). These formats are **not interchangeable** — v3.0 datasets cannot be loaded by OpenPI. Always run the conversion script with OpenPI's Python to produce v2.1-compatible datasets:

```bash
# Build the planner dataset at 30Hz — using OpenPI's Python
cd /home/chris/DataCollection
/home/chris/openpi/.venv/bin/python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --fps 30

# Build all three targets at 10Hz
for target in e2e planner correction; do
    /home/chris/openpi/.venv/bin/python scripts/convert_to_lerobot.py \
        --target $target \
        --data-dir data/vla_dataset \
        --fps 10
done
```

Datasets are saved to `~/lerobot_datasets/` by default (e.g. `~/lerobot_datasets/ChangChrisLiu/ur5e_planner_30hz/`). The repo ID auto-includes the FPS suffix: `ChangChrisLiu/ur5e_<target>_<fps>hz`. OpenPI training must set `HF_LEROBOT_HOME=~/lerobot_datasets` so LeRobot resolves datasets from this location instead of the default HuggingFace cache.

Images are stored as **PNG embedded in parquet** (`dtype: "image"`), not as MP4 video. This is required for OpenPI compatibility — OpenPI's data loader reads `dtype: "image"` features as PIL images from parquet files. Using `dtype: "video"` would require MP4 files that OpenPI's LeRobot v2.1 cannot decode on all systems.

**LeRobot Schema** (joint-based, not EEF):
```
state:       (7,)          float32  [q0-q5, gripper/255]
base_rgb:    (256,256,3)   image    Base camera RGB (PNG in parquet)
wrist_rgb:   (256,256,3)   image    Wrist camera RGB (PNG in parquet)
action:      (7,)          float32  [q0_next..q5_next, gripper_next/255]
task:        string                 Language instruction (auto-detected from path)
```

- `state[6]` (gripper): current frame's normalized gripper (0.0=open, 1.0=closed, robot convention — no inversion)
- `action[0:6]`: next frame's absolute joint positions (not deltas — OpenPI's `DeltaActions` converts to deltas during training)
- `action[6]`: next frame's normalized gripper. For trigger/stop/last frames: 1.0 (closed)

Options:
| Flag | Description |
|------|-------------|
| `--target` | Which dataset: `e2e`, `planner`, `correction` (correction also extracts near-grasp segments from successful episodes) |
| `--fps` | Output frame rate: 5, 10, 15, or 30 Hz (default 30). Lower = fewer frames = faster training |
| `--task` | Override auto-detected language instruction with a custom string |
| `--root` | Output directory root (default: `~/lerobot_datasets/`). Dataset lands at `<root>/<repo-id>/` |
| `--keep-noops` | Disable no-op frame removal (keep frames where the robot didn't move) |
| `--push-to-hub` | Push finished dataset to HuggingFace Hub |

### Dataset Registration (OpenVLA / OpenVLA-OFT Fine-Tuning)

To fine-tune OpenVLA or OpenVLA-OFT on your RLDS datasets, you must register them in the training codebase. Two files need modification in the OpenVLA/OFT repository. The instructions below are **self-contained** — copy-paste ready for a fresh server installation.

All dataset names include an FPS suffix (e.g., `ur5e_vla_planner_30hz`) so that multiple FPS variants can coexist on the same server.

#### Step 1: Add the Transform Function

**File:** `prismatic/vla/datasets/rlds/oxe/transforms.py`

Add this function definition (before the `OXE_STANDARDIZATION_TRANSFORMS` registry dict):

```python
def ur5e_vla_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Transform for ur5e_vla_* datasets that have action_gripper as a separate RLDS field.

    RLDS schema:
        observation.state: (8,) [x, y, z, roll, pitch, yaw, 0.0, gripper_0to1]
        action:            (6,) [dx, dy, dz, droll, dpitch, dyaw]
        action_gripper:    (1,) next-step gripper position, or 1.0 for stop/trigger signals

    This transform:
        1. Reads gripper from action_gripper field (next-step position or stop signal)
        2. Inverts to OpenVLA convention (1=open, 0=closed)
        3. Concatenates with 6D delta EEF action -> 7D action

    Note: action_gripper is the correct source because it contains the next-step gripper
    position for normal frames and 1.0 (stop signal) for trigger/stop frames.
    Using observation.state[-1] would give the CURRENT frame's gripper instead,
    missing stop signals entirely.
    """
    gripper_action = trajectory["action_gripper"][:, :1]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :6], gripper_action), axis=-1,
    )
    return trajectory
```

Then add these entries to the `OXE_STANDARDIZATION_TRANSFORMS` dictionary:

```python
    "ur5e_vla_e2e_10hz": ur5e_vla_dataset_transform,
    "ur5e_vla_e2e_30hz": ur5e_vla_dataset_transform,
    "ur5e_vla_planner_10hz": ur5e_vla_dataset_transform,
    "ur5e_vla_planner_30hz": ur5e_vla_dataset_transform,
    "ur5e_vla_correction_10hz": ur5e_vla_dataset_transform,
    "ur5e_vla_correction_30hz": ur5e_vla_dataset_transform,
```

#### Step 2: Register Dataset Configuration

**File:** `prismatic/vla/datasets/rlds/oxe/configs.py`

Add these entries to the `OXE_DATASET_CONFIGS` dictionary. All UR5e VLA datasets use the same configuration — only the name differs:

```python
    "ur5e_vla_e2e_10hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ur5e_vla_e2e_30hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ur5e_vla_planner_10hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ur5e_vla_planner_30hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ur5e_vla_correction_10hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ur5e_vla_correction_30hz": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
```

| Field | Value | Meaning |
|-------|-------|---------|
| `image_obs_keys.primary` | `"image"` | Maps to `observation.image` in RLDS (base camera) |
| `image_obs_keys.wrist` | `"wrist_image"` | Maps to `observation.wrist_image` in RLDS (wrist camera) |
| `state_encoding` | `StateEncoding.POS_EULER` | State is 8D EEF: `[x, y, z, roll, pitch, yaw, 0.0, gripper]` |
| `action_encoding` | `ActionEncoding.EEF_POS` | Action is 6D delta EEF `[dx, dy, dz, droll, dpitch, dyaw]` (pre-transform) |

#### Step 3: Place TFRecord Files

The RLDS TFRecord files must be accessible at `~/tensorflow_datasets/<dataset_name>/1.0.0/` on the training machine:

```bash
# On the collection machine (local) — build all three targets at 10Hz
CUDA_VISIBLE_DEVICES="" python scripts/convert_to_rlds.py \
    --target all --data-path data/vla_dataset --fps 10

# Transfer to training server
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner_10hz/ server:~/tensorflow_datasets/ur5e_vla_planner_10hz/
rsync -avz ~/tensorflow_datasets/ur5e_vla_e2e_10hz/ server:~/tensorflow_datasets/ur5e_vla_e2e_10hz/
rsync -avz ~/tensorflow_datasets/ur5e_vla_correction_10hz/ server:~/tensorflow_datasets/ur5e_vla_correction_10hz/
```

#### Step 4: Launch Fine-Tuning

Reference the FPS-suffixed dataset name in the training command:

```bash
# OpenVLA fine-tuning (example, 10Hz planner)
torchrun --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --adapter_tmp_dir adapters/ \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "ur5e-planner" \
    --wandb_entity your-entity \
    --save_steps 2500
```

> **Note:** These two files (`configs.py` and `transforms.py`) are the **only** modifications needed in the OpenVLA/OFT codebase. No other files require changes. The dataset name you register must exactly match the TFDS builder name (e.g., `ur5e_vla_planner_10hz`).

#### Currently Registered Datasets

Both `Sibo/openvla/` and `Sibo/openvla-oft/` have these datasets already registered:

| Dataset Name | Target | FPS | State/Action Format | Registered In |
|-------------|--------|-----|---------------------|---------------|
| `ur5e_vla_e2e_10hz` | `--target e2e` | 10 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |
| `ur5e_vla_e2e_30hz` | `--target e2e` | 30 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |
| `ur5e_vla_planner_10hz` | `--target planner` | 10 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |
| `ur5e_vla_planner_30hz` | `--target planner` | 30 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |
| `ur5e_vla_correction_10hz` | `--target correction` | 10 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |
| `ur5e_vla_correction_30hz` | `--target correction` | 30 | EEF + Euler RPY (POS_EULER / EEF_POS) | Both repos |

### Dataset Registration (LeRobot / OpenPI Fine-Tuning)

LeRobot datasets don't need code registration. The conversion script creates a self-contained dataset referenced by its HuggingFace repo ID. Datasets are saved to `~/lerobot_datasets/` by default:

```
~/lerobot_datasets/ChangChrisLiu/
├── ur5e_correction_10hz/
├── ur5e_correction_30hz/
├── ur5e_e2e_10hz/
├── ur5e_e2e_30hz/
├── ur5e_planner_10hz/
└── ur5e_planner_30hz/
```

To make OpenPI find these datasets at training time, set the environment variable:

```bash
export HF_LEROBOT_HOME=~/lerobot_datasets
```

For OpenPI fine-tuning, reference the dataset by repo ID in the training config (e.g. `ChangChrisLiu/ur5e_planner_30hz`). OpenPI's `DeltaActions` transform automatically converts the absolute next-step joint actions to delta format during training.

---

## Standalone Testing

### joysticktst.py

A standalone test script that connects directly to the robot via RTDE (no ZMQ) for validating joystick mappings and skill execution before using the full pipeline.

```bash
python joysticktst.py
```

Features:
- Direct RTDE control at 200Hz (no server processes needed)
- Full HOSAS mapping with calibration
- Skill execution with interrupt/resume (Right Btn 34=CPU, Right Btn 38=RAM)
- Interrupt via Left Btn 16 or Right Btn 25
- CSV waypoint recording
- Camera preview (optional, RealSense only)

---

## Code Organization

```
├── experiments/                      # Launch scripts
│   ├── launch_nodes.py               # T1: Robot ZMQ server (dual-port)
│   ├── launch_camera_nodes.py        # T2: Camera PUB/SUB publishers
│   ├── run_collection.py             # T3: Unified collection pipeline (200Hz control / 30Hz record)
│   └── run_inference.py              # T4: VLA inference pipeline (server-client, all 3 backends)
├── gello/
│   ├── agents/                       # Control agents
│   │   ├── agent.py                  # Agent protocol (Action type)
│   │   ├── joystick_agent.py         # HOSAS dual-stick (velocity + interrupt)
│   │   ├── vla_agent.py              # VLA model agent + 3 backend adapters (OpenPI/OpenVLA/OFT)
│   │   ├── safety.py                 # Workspace/joint/velocity safety checks for VLA inference
│   │   ├── gello_agent.py            # GELLO Dynamixel (joint-space)
│   │   ├── spacemouse_agent.py       # 3Dconnexion SpaceMouse
│   │   └── zmq_agent.py              # ZMQ client wrapper
│   ├── cameras/                      # Camera drivers
│   │   ├── camera.py                 # CameraDriver protocol
│   │   ├── realsense_camera.py       # Intel RealSense D435i
│   │   └── oakd_camera.py            # Luxonis OAK-D Pro
│   ├── robots/                       # Robot interfaces
│   │   ├── robot.py                  # Robot protocol
│   │   └── ur.py                     # UR5e (RTDE, moveL, stopL, path blending)
│   ├── skills/                       # Skill execution
│   │   └── csv_skill_executor.py     # CSV waypoint replay (path blending, grasp verification)
│   ├── data_utils/                   # Data pipeline
│   │   ├── episode_buffer.py         # Phase-labeled recording buffer
│   │   ├── dataset_writer.py         # Unified + legacy episode persistence
│   │   └── dual_dataset_buffer.py    # Legacy dual-buffer (deprecated)
│   ├── zmq_core/                     # ZMQ networking
│   │   ├── robot_node.py             # Robot server/client (dual-port, path support)
│   │   └── camera_node.py            # Camera PUB/SUB client
│   ├── utils/                        # Utilities
│   │   └── transform_utils.py        # SE(3) transforms for skill replay
│   └── env.py                        # RobotEnv (rate-limited step loop)
├── scripts/                          # Conversion & calibration
│   ├── calibrate_cameras.py          # Camera settings auto-detect/manual
│   ├── convert_to_rlds.py            # ENTRY POINT: .pkl → RLDS TFRecords
│   ├── convert_to_lerobot.py         # ENTRY POINT: .pkl → LeRobot v2.1
│   ├── conversion_utils.py           # Shared utilities (both scripts use this)
│   ├── rlds_builder_base.py          # TFDS builder base class (schema + processing)
│   ├── ur5e_vla_e2e/                 # TFDS builder config: e2e target (all phases)
│   ├── ur5e_vla_planner/             # TFDS builder config: planner target (teleop only)
│   ├── ur5e_vla_correction/          # TFDS builder config: correction target
│   └── test_dual_camera.py           # Camera connectivity test
├── configs/                          # Generated config files
│   └── camera_settings.json          # Saved camera parameters
├── joysticktst.py                    # Standalone HOSAS test (direct RTDE, 200Hz)
├── CPU_Skills.csv                    # CPU extraction skill (20 rel + 3 abs)
├── RAM_Skills.csv                    # RAM extraction skill (5 rel + 3 abs)
└── ros2/                             # ROS 2 packages for Franka FR3
```

## Frequency Summary

The control loop and recording are decoupled for fluent joystick control, matching `joysticktst.py`:

| Component | Rate | Notes |
|-----------|------|-------|
| **Joystick control loop** | **200Hz** | `speedL` velocity commands + gripper (matches joysticktst.py) |
| Gripper step | 10 units/cycle | `GRIPPER_STEP=10` at 200Hz, identical to joysticktst.py |
| Camera publishers | 30Hz | Hardware-limited (D435i + OAK-D Pro) |
| **Recording (teleop)** | **30Hz** | Camera-frame gated (only records on new timestamp) |
| **Skill recording thread** | **30Hz** | Concurrent recording during skill execution |
| Robot observations | 200Hz | Polled at control rate; cameras return cached frames |

## Known Issues / Environment Notes

### TF CUDA Compatibility (RTX 5090)

The RTX 5090 (compute capability 12.0) is not supported by the CUDA kernels shipped with current TensorFlow releases. When building RLDS datasets, the Universal Sentence Encoder (USE) will fail with `CUDA_ERROR_INVALID_PTX` if TF tries to use the GPU.

**Workaround:** Force CPU-only mode for RLDS conversion:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/convert_to_rlds.py \
    --target all --data-path data/vla_dataset --fps 10
```

This only affects the conversion step (USE embedding is a one-time operation per build). Training on the server uses a different TF/CUDA version that supports the training GPU.

### LeRobot Version Mismatch (v3.0 vs v2.1)

The `tele` conda environment has LeRobot **0.4.3** (dataset format v3.0), while OpenPI's vendored LeRobot is **0.1.0** (format v2.1). The formats are **not interchangeable** — they differ in metadata layout, data path templates, and internal APIs (`finalize()` exists in v3.0 but not v2.1; import paths differ).

**Solution: Always run `convert_to_lerobot.py` using OpenPI's Python:**

```bash
cd /home/chris/DataCollection
/home/chris/openpi/.venv/bin/python scripts/convert_to_lerobot.py \
    --target planner --data-dir data/vla_dataset --fps 30
```

Datasets are saved to `~/lerobot_datasets/` by default. Set `HF_LEROBOT_HOME=~/lerobot_datasets` when running OpenPI training so LeRobot resolves datasets from this location.

The conversion script auto-detects which LeRobot version is installed:
- v2.1 import: `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset`
- v3.0 import: `from lerobot.datasets.lerobot_dataset import LeRobotDataset`
- `finalize()` is only called if available (v3.0)

This ensures the output dataset uses v2.1 format, which OpenPI can load directly.

### TorchCodec / FFmpeg (LeRobot read-back)

LeRobot v3 uses `torchcodec` for video decoding. If `libtorchcodec` fails to load (FFmpeg version mismatch), dataset **creation** still works (uses SVT-AV1 encoder directly), but **reading back** the dataset locally will fail. This does not affect training on a properly configured server.

**Fix:** Install a compatible FFmpeg version (4, 5, 6, or 7) and matching torchcodec:

```bash
conda install -c conda-forge ffmpeg=6
pip install torchcodec
```

---

## Pipeline Validation

The full conversion and training-ingestion pipeline has been validated end-to-end across all three frameworks using 4 test episodes (2 CPU + 2 RAM, including both successful and correction episodes).

**Validated conversions (3 targets x 2 FPS x 2 formats = 12 builds):**

| Target | 30Hz Frames | 10Hz Frames | Ratio | Episodes |
|--------|-------------|-------------|-------|----------|
| planner | 2,025 | 678 | 2.99x | 4 (all with stop/trigger signals) |
| e2e | 4,965 | 1,685 | 2.95x | 4 |
| correction | 594 | 191 | 3.11x | 4 (2 real correction + 2 near-grasp) |

**Production LeRobot conversions (529 episodes, verified loading in OpenPI v2.1):**

| Target | FPS | Episodes | Frames | Notes |
|--------|-----|----------|--------|-------|
| planner | 10 | 529 | 87,426 | All episodes (teleop phase + trigger/stop signals) |
| planner | 30 | 529 | 261,312 | All episodes (teleop phase + trigger/stop signals) |
| e2e | 10 | 529 | 222,016 | All episodes (all 4 phases) |
| e2e | 30 | 529 | 652,828 | All episodes (all 4 phases) |
| correction | 10 | 527 (134 + 393 near-grasp) | 29,109 | 134 real correction + 393 near-grasp from successful episodes |
| correction | 30 | 527 (134 + 393 near-grasp) | 86,028 | 134 real correction + 393 near-grasp from successful episodes |

All six LeRobot datasets stored at `~/lerobot_datasets/ChangChrisLiu/`. Verified: correct shapes (`state: [7]`, `action: [7]`, images: `[3, 256, 256]`), non-zero image pixels (99.6-99.9%), and per-episode language instructions present.

**Validated framework ingestion:**

| Framework | Environment | Result |
|-----------|------------|--------|
| OpenVLA | `conda run -n vla` in `Sibo/openvla/` | 7D actions, gripper inversion, images, language |
| OpenVLA-OFT | `conda run -n vla` in `Sibo/openvla-oft/` | Same pipeline, verified identical |
| OpenPI | OpenPI `.venv/bin/python` | 7D state/action, dual cameras, task strings, LeRobot v2.1 format |

**Gripper convention verified through full chain:**
```
Robot (0-255) → RLDS state/action_gripper (0.0-1.0, same convention)
             → OpenVLA transform inverts (1.0=open, 0.0=closed)
             → LeRobot state/action (0.0-1.0, same as robot, no inversion)
```

---

## Fine-Tuning Guide

This section covers the complete fine-tuning pipeline for all three model architectures. The workflow is:

```
Collect data → Convert to format → Transfer to training machine → Register dataset → Fine-tune → Deploy
```

### Overview of Architectures

| Architecture | Framework | Format | Action Space | Fine-Tuning | VRAM (LoRA) |
|-------------|-----------|--------|--------------|-------------|-------------|
| [OpenVLA](https://github.com/openvla/openvla) | PyTorch | RLDS TFRecord | 7D tokenized | LoRA (PEFT) | 48 GB+ |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | PyTorch | RLDS TFRecord | 7D continuous (L1/diffusion) | LoRA + action head | 48 GB+ |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | JAX (Flax) | LeRobot v2.1 | 7D flow matching | LoRA (JAX native) | 22.5 GB+ |

---

### A. OpenVLA Fine-Tuning (Server)

OpenVLA uses LoRA fine-tuning via HuggingFace PEFT on a 7B parameter VLA model. Actions are tokenized into discrete bins. OpenVLA is a **single-image model** — it was pretrained exclusively on third-person camera images and only uses the base camera (`image_primary`). The wrist camera image stored in the same RLDS dataset is ignored during training.

#### Prerequisites

```bash
# On the training server
cd /path/to/openvla
pip install -e .
pip install peft==0.11.1
```

#### A.1 Transfer RLDS Data

```bash
# On the collection machine (local) — build all three targets at 10Hz
CUDA_VISIBLE_DEVICES="" python scripts/convert_to_rlds.py \
    --target all \
    --data-path data/vla_dataset \
    --fps 10

# Transfer to training server (dataset names include FPS suffix)
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner_10hz/ server:~/tensorflow_datasets/ur5e_vla_planner_10hz/
rsync -avz ~/tensorflow_datasets/ur5e_vla_e2e_10hz/ server:~/tensorflow_datasets/ur5e_vla_e2e_10hz/
rsync -avz ~/tensorflow_datasets/ur5e_vla_correction_10hz/ server:~/tensorflow_datasets/ur5e_vla_correction_10hz/
```

#### A.2 Register Dataset (Already Done)

Both `Sibo/openvla/` and `Sibo/openvla-oft/` have the datasets pre-registered in `configs.py` and `transforms.py`. See [Dataset Registration (OpenVLA / OpenVLA-OFT Fine-Tuning)](#dataset-registration-openvla--openvla-oft-fine-tuning) for details. If you move to a new server installation, copy the config and transform entries.

#### A.3 Fine-Tune

```bash
# Single GPU (48 GB minimum, batch 12 max)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --adapter_tmp_dir adapters/ \
    --lora_rank 32 \
    --batch_size 12 \
    --max_steps 200000 \
    --save_steps 5000 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "ur5e-openvla" \
    --wandb_entity your-entity

# Multi-GPU (K GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --batch_size 16 \
    --learning_rate 5e-4
```

**Key Parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--vla_path` | `openvla/openvla-7b` | HuggingFace model ID or local path |
| `--dataset_name` | - | Must match TFDS builder name exactly (e.g., `ur5e_vla_planner_10hz`) |
| `--data_root_dir` | `datasets/open-x-embodiment` | Directory containing `ur5e_vla_*_<fps>hz/1.0.0/` |
| `--batch_size` | 16 | 12 max on 48 GB, 24 on 80 GB |
| `--lora_rank` | 32 | Higher = more capacity, more VRAM |
| `--use_lora` | True | Set False for full fine-tuning (80 GB+ required) |
| `--image_aug` | True | Recommended for small datasets |
| `--save_latest_checkpoint_only` | True | Set False to keep all checkpoints |

#### A.4 Inference with Fine-Tuned OpenVLA

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# Load fine-tuned model
processor = AutoProcessor.from_pretrained("runs/ur5e_vla_planner+b16+lr-5e-4/", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained("runs/ur5e_vla_planner+b16+lr-5e-4/", trust_remote_code=True)

# Predict action
image = Image.fromarray(base_rgb)
prompt = "In: What action should the robot take to pick up the CPU?\nOut:"
inputs = processor(prompt, image).to("cuda")
action_tokens = model.predict_action(**inputs)
# action_tokens: (7,) — 6 delta EEF [dx,dy,dz,dr,dp,dy] + 1 gripper (normalized)
```

---

### B. OpenVLA-OFT Fine-Tuning (Server)

OpenVLA-OFT adds a **continuous action head** (L1 regression or diffusion) on top of the OpenVLA backbone, producing much smoother actions than the tokenized baseline. OFT also supports **multi-image input** — it can use both the base (third-person) and wrist cameras simultaneously, unlike base OpenVLA which only uses one image.

**Same RLDS dataset, different training flags.** OFT uses the exact same TFRecord files as OpenVLA. The only differences are in `finetune.py` arguments — no dataset rebuild or separate conversion is needed.

#### B.1 Transfer RLDS Data

Same as OpenVLA — use `rsync` to transfer RLDS TFRecords to the server. The same dataset files work for both models.

#### B.2 Fine-Tune

**Single-camera mode** (base camera only, same as OpenVLA — good baseline):

```bash
# L1 regression, single image (default: --num_images_in_input 1)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --use_l1_regression True \
    --use_diffusion False \
    --num_images_in_input 1 \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 8 \
    --max_steps 200000 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --save_freq 10000 \
    --image_aug True \
    --wandb_project "ur5e-openvla-oft" \
    --wandb_entity your-entity
```

**Dual-camera mode** (base + wrist — leverages both cameras for richer visual context):

```bash
# L1 regression, two images (base + wrist)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --use_l1_regression True \
    --use_diffusion False \
    --num_images_in_input 2 \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 8 \
    --max_steps 200000 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --save_freq 10000 \
    --image_aug True \
    --wandb_project "ur5e-openvla-oft-dual" \
    --wandb_entity your-entity
```

When `--num_images_in_input 2`:
- The data loader automatically loads both `image_primary` (base camera) and `image_wrist` (wrist camera) from the same TFRecord
- Each image is processed independently through the fused SigLIP+DINOv2 backbone
- Feature patches are concatenated: 256 patches per image → **512 patches total**
- This doubles the visual token count, so expect ~1.5-2x slower training and higher VRAM usage

**Diffusion action head** (potentially better for multimodal actions):

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner_10hz \
    --run_root_dir runs/ \
    --use_l1_regression False \
    --use_diffusion True \
    --num_diffusion_steps_train 50 \
    --num_images_in_input 2 \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 8 \
    --max_steps 200000 \
    --learning_rate 5e-4
```

**Key OFT-Specific Parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--num_images_in_input` | 1 | **1** = base camera only (like OpenVLA). **2** = base + wrist cameras (OFT multi-image). Must be >=2 to use wrist camera. |
| `--use_l1_regression` | True | L1 loss continuous action head |
| `--use_diffusion` | False | DDIM diffusion action head (mutually exclusive with L1) |
| `--num_diffusion_steps_train` | 50 | Diffusion steps per training iteration |
| `--use_film` | False | FiLM conditioning for language → vision |
| `--use_proprio` | False | Include proprioceptive state in input |
| `--num_steps_before_decay` | 100,000 | Steps before 10x LR decay (MultiStepLR) |
| `--save_freq` | 10,000 | Checkpoint save interval |
| `--save_latest_checkpoint_only` | False | Keeps all checkpoints by default |
| `--resume` | False | Resume from checkpoint |

**OpenVLA vs OpenVLA-OFT Comparison:**

| Feature | OpenVLA | OpenVLA-OFT (1 image) | OpenVLA-OFT (2 images) |
|---------|---------|----------------------|----------------------|
| Camera input | Base only (hardcoded) | Base only | Base + wrist |
| Action representation | Tokenized (discrete bins) | Continuous (L1 or diffusion) | Continuous (L1 or diffusion) |
| Visual patches | 256 | 256 | 512 (concatenated) |
| Action smoothness | Quantized steps | Smooth continuous | Smooth continuous |
| LR schedule | Fixed | MultiStepLR with decay | MultiStepLR with decay |
| Default batch size | 16 | 8 | 8 (may need to lower) |
| Checkpoint strategy | Latest only | All checkpoints | All checkpoints |
| Additional heads | None | L1/Diffusion/FiLM/Proprio | L1/Diffusion/FiLM/Proprio |

---

### C. OpenPI Fine-Tuning (JAX LoRA / Full)

OpenPI provides Pi0, Pi0-FAST, and Pi0.5 base models pre-trained on 10k+ hours of robot data. Fine-tuning uses **JAX with Flax NNX** and supports LoRA for memory-efficient training on consumer GPUs (22.5 GB+). Full fine-tuning requires multi-GPU setups.

The OpenPI codebase is tracked as a git submodule at `third_party/openpi/` for version pinning. The actual working installation used for training lives at `~/openpi/` — custom UR5e config files are stored in `openpi_configs/` for reference and deployed into the working install.

#### C.1 Setup

```bash
cd /home/chris/openpi

# Virtual environment is already set up at .venv/
# Verify JAX sees your GPU
.venv/bin/python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

If starting from scratch on a new machine:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive

# Install uv (package manager) if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Fix numpy version conflict (openpi pins <2.0, but rerun-sdk needs >=2)
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' pyproject.toml
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' packages/openpi-client/pyproject.toml

# Install all dependencies (skip LFS for speed)
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Then deploy the UR5e custom files from `DataCollection/openpi_configs/` (see the README there for instructions).

**Requirements:**
- Python 3.11+
- JAX 0.5.3+ with CUDA 12 (`jax[cuda12]`)
- RTX 5090 (compute 12.0) is supported by JAX 0.4.38+ / jaxlib 0.5.3+
- 22.5 GB+ VRAM for LoRA fine-tuning

**Environment variables** (add to `~/.bashrc`):
```bash
# CUDA toolkit
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# OpenPI uv venv nvidia libs (cudnn, nccl, cublas — fixes: ImportError: libcudnn.so.9)
export LD_LIBRARY_PATH="$(find ~/.cache/uv/archive-v0/ -maxdepth 4 -path '*/nvidia/*/lib' -type d 2>/dev/null | grep -v triton | grep -v tensorflow | tr '\n' ':')$LD_LIBRARY_PATH"

# Dataset location, GPU memory, wandb
export HF_LEROBOT_HOME=~/lerobot_datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_API_KEY="<your-key>"                # v1 key — must use env var, `wandb login` rejects v1 format
```

Model weights and assets are cached at `~/openpi_data/` (configurable via `OPENPI_DATA_HOME`).

#### C.2 Convert Data to LeRobot Format

> **You MUST run the conversion script using OpenPI's Python** (not the `tele` conda env). OpenPI uses LeRobot v2.1 format; the `tele` env produces v3.0 format which OpenPI cannot read.

```bash
cd /home/chris/DataCollection

# Convert all three targets at both FPS rates
for target in e2e planner correction; do
    for fps in 10 30; do
        /home/chris/openpi/.venv/bin/python scripts/convert_to_lerobot.py \
            --target $target --data-dir data/vla_dataset --fps $fps
    done
done
```

Datasets are saved to `~/lerobot_datasets/ChangChrisLiu/ur5e_<target>_<fps>hz/`.

#### C.3 Config Matrix (52 Configs)

The OpenPI codebase contains **52 UR5e training configs**: 4 original backward-compat configs + 48 FPS-variant configs covering the full matrix.

**Naming convention**: `{model}_ur5e_{target}[_lora]_{fps}hz`

**Models (4):**

| Model Key | Architecture | Base Checkpoint | Notes |
|-----------|-------------|-----------------|-------|
| `pi0` | Flow matching | `pi0_base` | Recommended starting point for single-arm |
| `pi0_fast` | Autoregressive FAST | `pi0_fast_base` | Shorter action horizon (10 vs 50) |
| `pi05` | Flow matching (Pi0.5) | `pi05_base` | From base checkpoint |
| `pi05_droid` | Flow matching (Pi0.5) | `pi05_droid` | From DROID checkpoint (single-arm manipulation) |

**Targets (3) x FPS (2):**

| Target | Phases Included | 10hz Dataset | 30hz Dataset |
|--------|----------------|--------------|--------------|
| `planner` | teleop only | `ChangChrisLiu/ur5e_planner_10hz` | `ChangChrisLiu/ur5e_planner_30hz` |
| `e2e` | all 4 phases | `ChangChrisLiu/ur5e_e2e_10hz` | `ChangChrisLiu/ur5e_e2e_30hz` |
| `correction` | correction only | `ChangChrisLiu/ur5e_correction_10hz` | `ChangChrisLiu/ur5e_correction_30hz` |

**Example config names** (each model has 12 configs = 3 targets x 2 FPS x 2 variants):

```
pi0_ur5e_planner_lora_10hz          pi0_ur5e_planner_10hz
pi0_fast_ur5e_e2e_lora_30hz         pi0_fast_ur5e_e2e_30hz
pi05_ur5e_correction_lora_10hz      pi05_ur5e_correction_10hz
pi05_droid_ur5e_planner_lora_30hz   pi05_droid_ur5e_planner_30hz
```

Original backward-compat configs (all point to `ur5e_planner_30hz`): `pi0_ur5e`, `pi0_ur5e_lora`, `pi0_fast_ur5e`, `pi0_fast_ur5e_lora`.

See [`openpi_configs/README.md`](openpi_configs/README.md) for the full config reference and [`examples/ur5/README.md`](third_party/openpi/examples/ur5/README.md) for the complete training guide.

#### C.4 Compute Normalization Statistics

OpenPI normalizes states and actions using dataset statistics. Compute norm stats before training:

```bash
cd /home/chris/openpi
export HF_LEROBOT_HOME=~/lerobot_datasets

# Single config
uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_planner_lora_10hz

# All 6 datasets (batch script — ~18 min each, ~2 hrs total)
bash scripts/compute_all_ur5e_norm_stats.sh
```

**Stats are identical across all model types for the same dataset** — `compute_norm_stats.py` computes mean, std, q01, q99 from raw data without using model_type. Only **6 computations** needed (one per dataset). All 52 configs share these via symlinks (see `openpi_configs/SERVER_SETUP_HPRC.md` Step 4 for the symlink script).

#### C.5 Train

**LoRA fine-tuning (local desktop, 22.5 GB+ VRAM):**

All env vars (`HF_LEROBOT_HOME`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, `WANDB_API_KEY`, `LD_LIBRARY_PATH`) should be set in `~/.bashrc` (see C.1). Do NOT use inline env vars — they break when pasting.

```bash
cd /home/chris/openpi

# Pi0 LoRA — recommended starting point
uv run scripts/train.py pi0_ur5e_planner_lora_10hz --exp-name planner_v1 --project-name ur5e-finetuning --overwrite

# Pi0-FAST LoRA
uv run scripts/train.py pi0_fast_ur5e_e2e_lora_10hz --exp-name e2e_fast_v1 --project-name ur5e-finetuning --overwrite

# Pi0.5-DROID LoRA
uv run scripts/train.py pi05_droid_ur5e_planner_lora_10hz --exp-name planner_pi05d_v1 --project-name ur5e-finetuning --overwrite
```

**Full fine-tuning (server, multi-GPU):**

```bash
uv run scripts/train.py pi05_droid_ur5e_e2e_30hz --exp-name e2e_pi05d_full_v1 --project-name ur5e-finetuning --fsdp-devices 4 --overwrite
```

**Resume training** (replace `--overwrite` with `--resume`):

```bash
uv run scripts/train.py pi0_ur5e_planner_lora_10hz --exp-name planner_v1 --project-name ur5e-finetuning --num-train-steps 50000 --resume
```

**Key training parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--exp-name` | required | Unique experiment name |
| `--overwrite` | False | Delete existing checkpoints, start fresh |
| `--resume` | False | Resume from latest checkpoint in existing dir |
| `--num-train-steps` | 30,000 | Total training steps |
| `--batch-size` | 32 | Reduce if OOM (Pi0.5 LoRA needs ~22.5 GB) |
| `--fsdp-devices` | 1 | Set >1 for multi-GPU FSDP |

**FPS selection:** 10hz datasets are ~3x faster to train (fewer frames per episode). Use 10hz for LoRA iteration, 30hz for final training runs.

#### C.6 Model Selection: Pi0 vs Pi0-FAST vs Pi0.5

| Feature | Pi0 | Pi0-FAST | Pi0.5 (base / DROID) |
|---------|-----|----------|---------------------|
| Architecture | Flow matching | Autoregressive FAST | Flow matching (upgraded) |
| Action chunk | 50 steps | 10 steps | 10 steps |
| Internal action dim | 32 (auto-padded) | 7 (explicit) | 32 (auto-padded) |
| LoRA batch size | Default | Default | 32 (cosine decay lr=5e-5) |
| Full finetune batch | Default | Default | 256 (EMA 0.999) |
| Recommended for | Single-arm starting point | Shorter action horizons | Bimanual / complex tasks |

> **Note:** Community reports suggest Pi0 may outperform Pi0.5 on single-arm tasks. Start with Pi0 LoRA, try Pi0-FAST if you prefer shorter action horizons. Pi0.5-DROID may transfer better to UR5e than Pi0.5-base since DROID is single-arm manipulation data.

#### C.7 Training Run 1: Pi0.5-DROID LoRA x 3 Targets @ 10hz

First training campaign: Pi0.5-DROID LoRA, 30k steps each, all three targets at 10hz. Run on both local desktop and GRACE server in parallel.

| Config | Dataset | Steps | What It Learns |
|--------|---------|-------|----------------|
| `pi05_droid_ur5e_planner_lora_10hz` | planner 10hz (87k frames) | 30,000 | Teleop approach — when to hand off to skill |
| `pi05_droid_ur5e_e2e_lora_10hz` | e2e 10hz (222k frames) | 30,000 | Full task — all 4 phases autonomously |
| `pi05_droid_ur5e_correction_lora_10hz` | correction 10hz (29k frames) | 30,000 | Grasp recovery after failed grasp |

**Time**: ~36-45 hours (3 sequential runs, 1 GPU). **Wandb**: `ur5e-finetuning` project.

**Local** (RTX 5090): Run directly in terminal — commands ready to copy-paste.
**GRACE** (HPRC): Single SLURM job script runs all 3 sequentially.

See [`openpi_configs/TRAINING_RUN_1.md`](openpi_configs/TRAINING_RUN_1.md) for the complete guide (Part A: local, Part B: GRACE). See [`openpi_configs/SERVER_SETUP_HPRC.md`](openpi_configs/SERVER_SETUP_HPRC.md) for GRACE server setup.

#### C.8 Serve Fine-Tuned Policy

```bash
cd /home/chris/openpi

# Start policy server (WebSocket, port 8000)
# Swap config/dir for whichever model you trained
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_droid_ur5e_planner_lora_10hz \
    --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000
```

**Client code for UR5e inference:**

```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy(host="localhost", port=8000)

observation = {
    "observation/image": base_rgb,           # (256,256,3) uint8
    "observation/wrist_image": wrist_rgb,    # (256,256,3) uint8
    "observation/state": state_7d,           # [q0-q5, gripper/255] unnormalized
    "prompt": "Pick up the CPU and place it in the socket",
}

result = client.infer(observation)
action_chunk = result["actions"]  # (action_horizon, 7) absolute joint positions
```

State and images should be sent **unnormalized** — the server handles normalization internally.

#### C.9 Troubleshooting (OpenPI)

| Issue | Solution |
|-------|---------|
| `FileNotFoundError` for dataset | Set `HF_LEROBOT_HOME=~/lerobot_datasets` in `~/.bashrc` |
| `ImportError: libcudnn.so.9` | `LD_LIBRARY_PATH` missing nvidia libs from uv cache — see C.1 env vars |
| OOM during LoRA training | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` in `~/.bashrc`, or reduce `--batch-size` |
| Diverging loss | Check norm stats — dimensions with tiny std cause huge normalized values |
| CUDA error on RTX 5090 | Ensure jaxlib >= 0.5.3 (JAX 0.5.3+ works, verified) |
| Action dimension mismatch | Pi0: `action_dim=32` internal (auto-padded). Pi0-FAST: `action_dim=7` explicit |
| Strange robot movements | Verify `DeltaActions` mask: `make_bool_mask(6, -1)` = delta joints, absolute gripper |
| Cannot resume training | Use `--resume` flag (not just removing `--overwrite` — that errors if checkpoint dir exists) |
| Missing norm stats | Run `compute_norm_stats.py` for your config before training |
| `ModuleNotFoundError` | Run from openpi root: `cd /home/chris/openpi && uv run scripts/train.py ...` |
| Config not found | Check exact name with `uv run scripts/train.py --help` |
| Wandb not logging | Set `WANDB_API_KEY` in `~/.bashrc` (v1 keys must use env var, not `wandb login`) |
| Multi-line commands fail on paste | Use single-line commands. Env vars should be in `~/.bashrc`, not inline |
| GRACE: `uv` resolution error | `export UV_FROZEN=1` in SLURM script AND `~/.bashrc` |
| GRACE: GCS checkpoint download fails | `gcsfs`/`aiohttp` ignores `http_proxy`. Pre-download on login node — see `SERVER_SETUP_HPRC.md` Step 5 |
| GRACE: `Disk quota exceeded` | HF datasets Arrow cache on scratch. Set `HF_HOME` to scratch, clean: `find $SCRATCH -name '*.arrow' -path '*/cache/*' -delete` |

---

## Inference Pipeline

Deploy trained VLA models on the UR5e for performance validation. The inference script (`experiments/run_inference.py`) mirrors `run_collection.py` but replaces human joystick input with model predictions.

### Environment Architecture

All three backends use a **server-client architecture** so `run_inference.py` always runs from the `tele` conda env. Each model server runs in its own environment:

| Backend | Model Server Env | Server Script | Client Protocol | Default Port |
|---------|-----------------|---------------|-----------------|-------------|
| OpenPI | `uv` venv at `~/openpi/.venv/` | `scripts/serve_policy.py` | WebSocket (`openpi_client`) | 8000 |
| OpenVLA | `conda activate vla` | `vla-scripts/deploy.py` | REST (`requests` + `json_numpy`) | 8000 |
| OpenVLA-OFT | `conda activate oft` | `vla-scripts/deploy.py` | REST (`requests` + `json_numpy`) | 8777 |

This design means you **do not need model-specific conda envs in the inference script** — only lightweight client libraries.

### Prerequisites

Install the client dependencies in the `tele` conda env:

```bash
conda activate tele

# For OpenPI inference (WebSocket client)
pip install /home/chris/openpi/packages/openpi-client/

# For OpenVLA / OpenVLA-OFT inference (REST client)
pip install json-numpy requests
```

### Checkpoint Locations

Checkpoints are saved under each model's own directory:

| Model | Checkpoint Location | Example |
|-------|-------------------|---------|
| OpenPI | `~/openpi/checkpoints/<config>/<exp_name>/<step>/` | `~/openpi/checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000/` |
| OpenVLA | `~/Sibo/openvla/runs/<exp_name>/` | `~/Sibo/openvla/runs/ur5e_planner+b16+lr-5e-4/` |
| OpenVLA-OFT | `~/Sibo/openvla-oft/runs/<exp_name>/` | `~/Sibo/openvla-oft/runs/ur5e_planner_l1_2img/` |

### Terminal Architecture (Inference)

Four terminals, same as data collection but T3 runs a model server and T4 runs inference:

```
T1: conda activate tele
    python experiments/launch_nodes.py --robot ur --robot-ip 10.125.144.209

T2: conda activate tele
    python experiments/launch_camera_nodes.py --camera-settings configs/camera_settings.json

T3: Start model server (see per-backend commands below)

T4: conda activate tele
    python experiments/run_inference.py --model-type <backend> --server-port <port> \
        --mode planner --task cpu
```

### Starting Model Servers (T3)

#### OpenPI Server

```bash
cd /home/chris/openpi

# Planner model
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_droid_ur5e_planner_lora_10hz \
    --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/30000 \
    --port 8000

# Correction model (separate terminal, different port)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_droid_ur5e_correction_lora_10hz \
    --policy.dir checkpoints/pi05_droid_ur5e_correction_lora_10hz/correction_v1/30000 \
    --port 8001
```

#### OpenVLA Server

```bash
conda activate vla
cd /home/chris/Sibo/openvla

python vla-scripts/deploy.py \
    --pretrained_checkpoint runs/ur5e_planner+b16+lr-5e-4 \
    --port 8000
```

#### OpenVLA-OFT Server

```bash
conda activate oft
cd /home/chris/Sibo/openvla-oft

python vla-scripts/deploy.py \
    --pretrained_checkpoint runs/ur5e_planner_l1_2img \
    --unnorm_key ur5e_vla_planner_10hz \
    --use_l1_regression True \
    --use_proprio True \
    --num_images_in_input 2 \
    --port 8777
```

### Running Inference (T4)

#### Planner Mode (approach → skill → correction → skill_resume)

```bash
conda activate tele

# OpenPI planner (with correction model on port 8001)
python experiments/run_inference.py \
    --model-type openpi \
    --server-port 8000 \
    --mode planner \
    --task cpu \
    --fps 10 \
    --correction-server-port 8001

# OpenVLA planner (no correction model)
python experiments/run_inference.py \
    --model-type openvla \
    --server-port 8000 \
    --mode planner \
    --task cpu \
    --fps 10

# OpenVLA-OFT planner
python experiments/run_inference.py \
    --model-type openvla_oft \
    --server-port 8777 \
    --mode planner \
    --task cpu \
    --fps 10
```

#### E2E Mode (model handles entire trajectory)

```bash
# OpenPI e2e (prompt auto-detected from --task)
python experiments/run_inference.py \
    --model-type openpi \
    --server-port 8000 \
    --mode e2e \
    --task cpu \
    --fps 10 \
    --max-steps 500
```

### Language Instructions (Prompt Handling)

The `--task` flag is the **single source of truth** — it sets both the CSV skill and the language instruction. The prompt matches the exact strings used during training data conversion (`conversion_utils.py`):

| `--task` | Language Instruction Sent to Model |
|----------|-----------------------------------|
| `cpu` | "Extract the CPU from the Bracket by unlocking it first, then extract the CPU and place it inside the yellow square area, then back home." |
| `ram` | "Extract the RAM from the slot and place it inside the blue square area, then back home." |

Each model server wraps the prompt differently:

| Backend | What the client sends | What the server does internally |
|---------|----------------------|-------------------------------|
| **OpenPI** | `{"prompt": "<instruction>"}` | Passes to model as-is (no wrapping) |
| **OpenVLA** | `{"instruction": "<instruction>"}` | Wraps as `"In: What action should the robot take to <instruction>?\nOut:"` |
| **OpenVLA-OFT** | `{"instruction": "<instruction>"}` | Same wrapping as OpenVLA |

You only need to specify `--task cpu` or `--task ram` — the correct prompt and skill CSV are selected automatically. The server handles any model-specific prompt formatting.

### Action Application Summary

| Model | `infer()` returns | `apply_action()` | Robot command |
|-------|-------------------|-------------------|---------------|
| OpenPI | Chunk of 10 actions `(7,)`: absolute `[q0-q5, gripper]` | `target → servoJ` | `command_joint_state(target)` |
| OpenVLA | Single action `(7,)`: delta `[dx,dy,dz,dr,dp,dy, grip_inv]` | `current_tcp + delta → moveL` | `move_linear(pose)` + `set_gripper()` |
| OpenVLA-OFT | Chunk of 8 actions `(7,)`: same as OpenVLA | Same as OpenVLA | Same as OpenVLA |

### Stop Signal Detection

| Model | Training Convention | Stop Detection |
|-------|-------------------|----------------|
| OpenPI | gripper=1.0 (absolute, no inversion) | `action[6] > 0.95` |
| OpenVLA | gripper inverted (1=open, 0=close) | `action[6] < 0.05` |
| OpenVLA-OFT | Same as OpenVLA | `action[6] < 0.05` |

Physical gripper range: 3-230 (normalized 0.012-0.902). Stop thresholds give wide margin above/below the physical range.

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `openpi` | Backend: `openpi`, `openvla`, or `openvla_oft` |
| `--server-host` | `127.0.0.1` | Model server host |
| `--server-port` | `8000` | Model server port |
| `--task` | `cpu` | Task: `cpu` or `ram` (sets both skill CSV and language prompt) |
| `--mode` | `planner` | Pipeline mode: `planner` or `e2e` |
| `--fps` | `10` | Control rate (must match training FPS) |
| `--max-steps` | `300` | Timeout in inference steps |
| `--correction-server-port` | `0` | Correction model server port (0 = disabled) |
| `--open-loop-horizon` | `10` | OpenPI: chunk steps to use before re-querying |
| `--unnorm-key` | `ur5e_vla_planner_10hz` | OpenVLA/OFT normalization stats key |
| `--record` / `--no-record` | `True` | Record episodes for evaluation |
| `--data-dir` | `data/inference_episodes` | Output directory |
| `--disable-safety` | `False` | Disable workspace/velocity safety checks |

### Safety Monitor

The inference pipeline includes a `SafetyMonitor` (`gello/agents/safety.py`) that clamps actions before execution:

| Check | Limit | Applied To |
|-------|-------|------------|
| Max joint delta | 0.05 rad/step | OpenPI (servoJ) |
| Max EEF delta | 0.01 m/step | OpenVLA, OFT (moveL) |
| Max rotation delta | 0.05 rad/step | OpenVLA, OFT (moveL) |
| Workspace bounds | Configurable | All (target pose check) |

Disable with `--disable-safety` for unconstrained execution (not recommended for initial testing).

---

## License

This project is licensed under the MIT License.
