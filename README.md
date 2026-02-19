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

From this single recording, **three training datasets** are derived during conversion:

| Training Target | Phases Included | Stop Signal | Purpose |
|-----------------|----------------|-------------|---------|
| **End-to-End** | all 4 phases | none | Full trajectory for VLA executor models |
| **Planner** | teleop only | 3 frames appended | Teaches WHEN to call a skill |
| **Correction** | correction only | 3 frames appended | Teaches recovery after grasp failure |

Stop signals (3 copies of last frame with gripper=255) are **not stored** in raw recordings — they are synthesized during conversion.

## Hardware

| Component | Model | Connection |
|-----------|-------|------------|
| Robot | UR5e | RTDE at `10.125.144.209` |
| Gripper | Robotiq 2F-85 | Socket port 63352 (positions 0-255) |
| Wrist Camera | Intel RealSense D435i | USB, 1280x720 @ 30Hz (RGB + aligned depth) |
| Base Camera | Luxonis OAK-D Pro | USB, 1280x720 @ 30Hz (RGB + aligned depth) |
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
conda create -n datacollection python=3.11 -y
conda activate datacollection

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

| Skill | Grasp Close | Threshold | Pass Condition |
|-------|-------------|-----------|----------------|
| CPU | 165 | 155 | actual < 155 (object blocking fingers) |
| RAM | 225 | 225 | actual < 225 (object blocking fingers) |

- **Pass**: Object is held. Skill continues normally.
- **Fail**: Fingers closed on nothing (actual >= threshold). Recording pauses automatically and the pipeline waits for correction.

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
| CPU Extraction | Right 34 | `CPU_Skills.csv` | 20 | 23 | 165 | 155 | Path blending (first segment) |
| RAM Extraction | Right 38 | `RAM_Skills.csv` | 5 | 8 | 229 | 225 | Individual moveL (all segments) |

---

## Output Data Format

### Unified Episode Structure

```
data/vla_dataset/
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
```

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

### Conversion Pipeline

```
.pkl episodes (local)
      │
      ├──> RLDS TFRecords (for OpenVLA / OpenVLA-OFT)
      │       scripts/convert_to_rlds.py
      │       └── ~/tensorflow_datasets/ur5e_vla_<target>/
      │
      └──> LeRobot v3 datasets (for OpenPI)
              scripts/convert_to_lerobot.py
              └── HuggingFace Hub: ChangChrisLiu/ur5e_<target>
```

During conversion, the following processing is applied **in this order** (ordering matters):

1. **Phase filtering** — select frames by phase label(s)
2. **No-op frame removal** — drop frames where joints haven't changed (threshold: 1e-4 rad)
3. **Stop signal synthesis** — 3 copies of last frame with gripper=255 (planner/correction only)
4. **Delta joint computation** — for RLDS action format
5. **Image resizing** — 256x256 for RLDS (configurable)

> **Important:** No-op removal (step 2) must happen **before** stop signal synthesis (step 3). Stop signals duplicate the last frame's joint positions, so they would be incorrectly removed as no-ops if the order were reversed.

### RLDS Conversion (OpenVLA / OpenVLA-OFT)

Uses TFDS `GeneratorBasedBuilder` following the [kpertsch/rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) template. Three builder modules in `scripts/`:

| Builder | Directory | Phase Filter | Stop Signal |
|---------|-----------|-------------|-------------|
| `ur5e_vla_e2e` | `scripts/ur5e_vla_e2e/` | all phases | no |
| `ur5e_vla_planner` | `scripts/ur5e_vla_planner/` | teleop | yes (3 frames) |
| `ur5e_vla_correction` | `scripts/ur5e_vla_correction/` | correction | yes (3 frames) |

All three inherit from `scripts/rlds_builder_base.py` which defines the shared RLDS schema.

**RLDS Schema:**
```
observation.image:          (256, 256, 3) uint8   Base camera RGB (JPEG)
observation.wrist_image:    (256, 256, 3) uint8   Wrist camera RGB (JPEG)
observation.state:          (8,)          float32  [q0-q5, 0.0, gripper_0to1]
action:                     (6,)          float32  Delta joint positions [dq0-dq5]
action_gripper:             (1,)          float32  Gripper position 0-1
language_instruction:       string                 Task description
language_embedding:         (512,)        float32  Universal Sentence Encoder
```

**Build commands:**

```bash
# Build a single target
python scripts/convert_to_rlds.py \
    --target planner \
    --data-path data/vla_dataset \
    --task "Pick up the CPU and place it in the socket"

# Build all three targets
python scripts/convert_to_rlds.py \
    --target all \
    --data-path data/vla_dataset \
    --task "Pick up the CPU and place it in the socket"

# Custom image size (224 for base OpenVLA, 256 for OFT)
python scripts/convert_to_rlds.py \
    --target planner \
    --data-path data/vla_dataset \
    --task "Pick up the CPU" \
    --image-size 224
```

**Transfer to server:**

```bash
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner/ server:~/tensorflow_datasets/ur5e_vla_planner/
```

**Verify:**

```bash
python -c "
import tensorflow_datasets as tfds
b = tfds.builder('ur5e_vla_planner', data_dir='$HOME/tensorflow_datasets')
ds = b.as_dataset(split='train')
for traj in ds.take(1):
    for step in traj['steps']:
        print('image:', step['observation']['image'].shape)        # (256,256,3)
        print('wrist:', step['observation']['wrist_image'].shape)  # (256,256,3)
        print('state:', step['observation']['state'].shape)        # (8,)
        print('action:', step['action'].shape)                     # (6,)
        print('gripper:', step['action_gripper'].shape)            # (1,)
        break
"
```

### LeRobot Conversion (OpenPI)

```bash
python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --repo-id ChangChrisLiu/ur5e_planner \
    --task "Pick up the CPU and place it in the socket" \
    --fps 30
```

**LeRobot Schema:**
```
observation.state:              (7,)          float32  [q0-q5, gripper/255]
observation.images.base_rgb:    (256,256,3)   video    Base camera RGB
observation.images.wrist_rgb:   (256,256,3)   video    Wrist camera RGB
action:                         (7,)          float32  [q0_next..q5_next, gripper_next/255]
task:                           string                 Language instruction
```

Options:
- `--target` — `e2e`, `planner`, `correction`
- `--keep-noops` — disable no-op frame removal
- `--push` — push to HuggingFace Hub after conversion

### Dataset Registration (OpenVLA / OpenVLA-OFT Fine-Tuning)

To fine-tune OpenVLA or OpenVLA-OFT on your RLDS datasets, you must register them in the training codebase. Two files need modification in the OpenVLA/OFT repository:

#### Step 1: Register Dataset Configuration

**File:** `prismatic/vla/datasets/rlds/oxe/configs.py`

Add an entry to the `OXE_DATASET_CONFIGS` dictionary for each dataset. All UR5e datasets use the same configuration:

```python
"ur5e_vla_planner": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.JOINT,
    "action_encoding": ActionEncoding.JOINT_POS,
},
```

Repeat for `ur5e_vla_e2e` and `ur5e_vla_correction` with the same config block.

| Field | Value | Meaning |
|-------|-------|---------|
| `image_obs_keys.primary` | `"image"` | Maps to `observation.image` in RLDS (base camera) |
| `image_obs_keys.wrist` | `"wrist_image"` | Maps to `observation.wrist_image` in RLDS (wrist camera) |
| `state_encoding` | `StateEncoding.JOINT` | State is 8D joint-space: `[q0-q5, 0.0, gripper]` |
| `action_encoding` | `ActionEncoding.JOINT_POS` | Action is 6D delta joints (pre-transform) |

#### Step 2: Register Dataset Transform

**File:** `prismatic/vla/datasets/rlds/oxe/transforms.py`

All UR5e datasets reuse the same transform function. Add entries to the `OXE_STANDARDIZATION_TRANSFORMS` dictionary:

```python
"ur5e_vla_e2e": ur5e_disassembly_dataset_transform,
"ur5e_vla_planner": ur5e_disassembly_dataset_transform,
"ur5e_vla_correction": ur5e_disassembly_dataset_transform,
```

The transform function is already defined in the file:

```python
def ur5e_disassembly_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Extract gripper from observation state (last element)
    gripper_action = trajectory["observation"]["state"][:, -1:]
    # Invert: raw 0=open,1=close -> OpenVLA 1=open,0=close
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))
    # Combine: 6D delta joints + 1D inverted gripper = 7D action
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :6], gripper_action), axis=-1,
    )
    return trajectory
```

This takes the 6D delta joint action from RLDS, reads the gripper position from the observation state, inverts it (OpenVLA convention: 1=open, 0=closed), and produces the final 7D action vector.

#### Step 3: Place TFRecord Files

The RLDS TFRecord files must be accessible at `~/tensorflow_datasets/<dataset_name>/1.0.0/` on the training machine:

```bash
# On the collection machine (local)
python scripts/convert_to_rlds.py --target all --data-path data/vla_dataset --task "Pick up the CPU"

# Transfer to training server
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner/ server:~/tensorflow_datasets/ur5e_vla_planner/
rsync -avz ~/tensorflow_datasets/ur5e_vla_e2e/ server:~/tensorflow_datasets/ur5e_vla_e2e/
rsync -avz ~/tensorflow_datasets/ur5e_vla_correction/ server:~/tensorflow_datasets/ur5e_vla_correction/
```

#### Step 4: Launch Fine-Tuning

Reference the dataset by name in the training command:

```bash
# OpenVLA fine-tuning (example)
torchrun --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner \
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

> **Note:** These two files (`configs.py` and `transforms.py`) are the **only** modifications needed in the OpenVLA/OFT codebase. No other files require changes. The dataset name you register must exactly match the TFDS builder name (e.g., `ur5e_vla_planner`).

#### Currently Registered Datasets

Both `Sibo/openvla/` and `Sibo/openvla-oft/` have these datasets already registered:

| Dataset Name | Description | Registered In |
|-------------|-------------|---------------|
| `ur5e_vla_e2e` | Full trajectory (all phases) | Both repos |
| `ur5e_vla_planner` | Teleop approach + stop signal | Both repos |
| `ur5e_vla_correction` | Recovery after grasp failure + stop signal | Both repos |

### Dataset Registration (LeRobot / OpenPI Fine-Tuning)

LeRobot datasets don't need code registration. The conversion script creates a self-contained dataset that is referenced by its HuggingFace repo ID:

```bash
# Convert and push to Hub
python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --repo-id ChangChrisLiu/ur5e_planner \
    --task "Pick up the CPU" \
    --push-to-hub

# Or keep local only
python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --repo-id ChangChrisLiu/ur5e_planner \
    --task "Pick up the CPU" \
    --root /path/to/local/dataset
```

For OpenPI fine-tuning, reference the dataset by repo ID in the training config. OpenPI's `DeltaActions` transform automatically converts the absolute next-step actions to delta format during training.

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
│   └── run_collection.py             # T3: Unified collection pipeline (200Hz control / 30Hz record)
├── gello/
│   ├── agents/                       # Control agents
│   │   ├── agent.py                  # Agent protocol (Action type)
│   │   ├── joystick_agent.py         # HOSAS dual-stick (velocity + interrupt)
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
│   ├── conversion_utils.py           # Shared conversion utilities
│   ├── rlds_builder_base.py          # Shared TFDS GeneratorBasedBuilder base
│   ├── ur5e_vla_e2e/                 # RLDS builder: end-to-end
│   ├── ur5e_vla_planner/             # RLDS builder: planner
│   ├── ur5e_vla_correction/          # RLDS builder: correction
│   ├── convert_to_rlds.py            # RLDS conversion wrapper
│   ├── convert_to_lerobot.py         # LeRobot conversion
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
    --target all --data-path data/vla_dataset --task "Pick up the CPU"
```

This only affects the conversion step (USE embedding is a one-time operation per build). Training on the server uses a different TF/CUDA version that supports the training GPU.

### TorchCodec / FFmpeg (LeRobot read-back)

LeRobot v3 uses `torchcodec` for video decoding. If `libtorchcodec` fails to load (FFmpeg version mismatch), dataset **creation** still works (uses SVT-AV1 encoder directly), but **reading back** the dataset locally will fail. This does not affect training on a properly configured server.

**Fix:** Install a compatible FFmpeg version (4, 5, 6, or 7) and matching torchcodec:

```bash
conda install -c conda-forge ffmpeg=6
pip install torchcodec
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
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | JAX (Flax) | LeRobot v3 | 7D flow matching | LoRA (JAX native) | 22.5 GB+ |

---

### A. OpenVLA Fine-Tuning (Server)

OpenVLA uses LoRA fine-tuning via HuggingFace PEFT on a 7B parameter VLA model. Actions are tokenized into discrete bins.

#### Prerequisites

```bash
# On the training server
cd /path/to/openvla
pip install -e .
pip install peft==0.11.1
```

#### A.1 Transfer RLDS Data

```bash
# On the collection machine (local)
python scripts/convert_to_rlds.py \
    --target planner \
    --data-path data/vla_dataset \
    --task "Pick up the CPU and place it in the socket"

# Transfer to training server
rsync -avz ~/tensorflow_datasets/ur5e_vla_planner/ server:~/tensorflow_datasets/ur5e_vla_planner/
rsync -avz ~/tensorflow_datasets/ur5e_vla_e2e/ server:~/tensorflow_datasets/ur5e_vla_e2e/
rsync -avz ~/tensorflow_datasets/ur5e_vla_correction/ server:~/tensorflow_datasets/ur5e_vla_correction/
```

#### A.2 Register Dataset (Already Done)

Both `Sibo/openvla/` and `Sibo/openvla-oft/` have the datasets pre-registered in `configs.py` and `transforms.py`. See [Dataset Registration (OpenVLA / OpenVLA-OFT Fine-Tuning)](#dataset-registration-openvla--openvla-oft-fine-tuning) for details. If you move to a new server installation, copy the config and transform entries.

#### A.3 Fine-Tune

```bash
# Single GPU (48 GB minimum, batch 12 max)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner \
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
    --dataset_name ur5e_vla_planner \
    --run_root_dir runs/ \
    --batch_size 16 \
    --learning_rate 5e-4
```

**Key Parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--vla_path` | `openvla/openvla-7b` | HuggingFace model ID or local path |
| `--dataset_name` | - | Must match TFDS builder name exactly |
| `--data_root_dir` | `datasets/open-x-embodiment` | Directory containing `ur5e_vla_*/1.0.0/` |
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
# action_tokens: (7,) — 6 delta joints + 1 gripper (normalized)
```

---

### B. OpenVLA-OFT Fine-Tuning (Server)

OpenVLA-OFT adds a **continuous action head** (L1 regression or diffusion) on top of the OpenVLA backbone, producing much smoother actions than the tokenized baseline.

#### B.1 Transfer RLDS Data

Same as OpenVLA — use `rsync` to transfer RLDS TFRecords to the server.

#### B.2 Fine-Tune

```bash
# L1 regression action head (recommended starting point)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner \
    --run_root_dir runs/ \
    --use_l1_regression True \
    --use_diffusion False \
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

# Diffusion action head (potentially better for multimodal actions)
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name ur5e_vla_planner \
    --run_root_dir runs/ \
    --use_l1_regression False \
    --use_diffusion True \
    --num_diffusion_steps_train 50 \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 8 \
    --max_steps 200000 \
    --learning_rate 5e-4
```

**Key OFT-Specific Parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
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

| Feature | OpenVLA | OpenVLA-OFT |
|---------|---------|------------|
| Action representation | Tokenized (discrete bins) | Continuous (L1 or diffusion) |
| Action smoothness | Quantized steps | Smooth continuous |
| LR schedule | Fixed | MultiStepLR with decay |
| Default batch size | 16 | 8 |
| Checkpoint strategy | Latest only | All checkpoints |
| Additional heads | None | L1/Diffusion/FiLM/Proprio |

---

### C. OpenPI Fine-Tuning (Local Desktop — JAX LoRA)

OpenPI provides pi0/pi0-FAST/pi0.5 base models pre-trained on 10k+ hours of robot data. Fine-tuning uses **JAX with Flax NNX** and supports LoRA for memory-efficient training on consumer GPUs (22.5 GB+).

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

# Install all dependencies (skip LFS for speed)
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**Requirements:**
- Python 3.11+
- JAX 0.5.3+ with CUDA 12 (`jax[cuda12]`)
- RTX 5090 (compute 12.0) is supported by JAX 0.4.38+ / jaxlib 0.5.3+
- 22.5 GB+ VRAM for LoRA fine-tuning

#### C.2 Convert Data to LeRobot Format

```bash
cd /home/chris/DataCollection

# Convert planner dataset (or e2e, correction)
python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --repo-id ChangChrisLiu/ur5e_planner \
    --task "Pick up the CPU and place it in the socket" \
    --fps 30

# Push to HuggingFace Hub (recommended — OpenPI loads from Hub by default)
python scripts/convert_to_lerobot.py \
    --target planner \
    --data-dir data/vla_dataset \
    --repo-id ChangChrisLiu/ur5e_planner \
    --task "Pick up the CPU and place it in the socket" \
    --fps 30 \
    --push-to-hub
```

The dataset is stored at `~/.cache/huggingface/lerobot/ChangChrisLiu/ur5e_planner/`.

#### C.3 Register UR5e Config in OpenPI

Three files need to be added/modified in the OpenPI codebase.

**File 1: Create `src/openpi/policies/ur5e_policy.py`**

This defines how UR5e observations map to the model's input format and how model outputs map back to robot actions.

```python
"""UR5e policy transforms for OpenPI.

Maps UR5e observations (2 cameras, 6 joints + gripper) to the pi0 model
input format, and extracts 7D actions from model output.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 (H,W,C) — handles LeRobot float32 (C,H,W)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    """Convert UR5e observations to model input format.

    Input keys (after repack):
        observation/image:       base camera RGB
        observation/wrist_image: wrist camera RGB
        observation/state:       7D [q0-q5, gripper_norm]

    Output keys (model expects):
        state:       7D state vector
        image:       dict of named camera images
        image_mask:  dict of booleans (False = padding slot)
        actions:     action chunk (training only)
        prompt:      language instruction
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad unused right-wrist slot with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask padding for pi0 only; pi0-FAST treats all slots equally
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    """Extract 7D actions (6 joints + 1 gripper) from padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
```

**File 2: Add data config to `src/openpi/training/config.py`**

Add the import at the top of the file (with the other policy imports):

```python
import openpi.policies.ur5e_policy as ur5e_policy
```

Add the `LeRobotUR5eDataConfig` class (after `LeRobotLiberoDataConfig`):

```python
@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(DataConfigFactory):
    """UR5e single-arm robot with 2 cameras (base + wrist).

    Dataset schema (LeRobot v3):
        observation.images.base_rgb:  (256,256,3) video
        observation.images.wrist_rgb: (256,256,3) video
        observation.state:            (7,) float32 [q0-q5, gripper_norm]
        action:                       (7,) float32 [q0_next..q5_next, gripper_next_norm]

    Actions are absolute next-step joint positions. The DeltaActions transform
    converts joints to deltas during training; gripper stays absolute.
    """

    # Default prompt if dataset has no per-episode task field.
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack: map LeRobot dataset keys to the keys UR5eInputs expects.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.base_rgb",
                        "observation/wrist_image": "observation.images.wrist_rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms: UR5e-specific I/O mapping.
        data_transforms = _transforms.Group(
            inputs=[ur5e_policy.UR5eInputs(model_type=model_config.model_type)],
            outputs=[ur5e_policy.UR5eOutputs()],
        )

        # Delta action conversion: joints (first 6) → delta, gripper (7th) → absolute.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (tokenization, image resize, padding) — standard for all robots.
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

**File 3: Add training configs to `_CONFIGS` list in the same file**

Add these entries to the `_CONFIGS` list:

```python
#
# Fine-tuning UR5e configs.
#
# LoRA fine-tuning: pi0 (flow matching, 22.5 GB VRAM)
TrainConfig(
    name="pi0_ur5e_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotUR5eDataConfig(
        repo_id="ChangChrisLiu/ur5e_planner",
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,  # Disable EMA for LoRA
    num_train_steps=30_000,
),
# LoRA fine-tuning: pi0-FAST (autoregressive, 22.5 GB VRAM)
TrainConfig(
    name="pi0_fast_ur5e_lora",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,  # ~180 for single-arm
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotUR5eDataConfig(
        repo_id="ChangChrisLiu/ur5e_planner",
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_fast_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=7, action_horizon=10, max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,
),
```

To train on a different dataset target (e.g., `e2e` or `correction`), either:
- Create separate configs with different `repo_id` values, or
- Override at the command line: `--data.repo_id ChangChrisLiu/ur5e_e2e`

#### C.4 Compute Normalization Statistics

OpenPI normalizes states and actions using dataset statistics. You can either compute fresh stats or reuse the pre-trained UR5e stats from the base model.

**Option A: Reuse base model stats (recommended for transfer)**

Already configured via `AssetsConfig(assets_dir="gs://openpi-assets/...", asset_id="ur5e")` in the config above. The base model was pre-trained on UR5e data with the same joint angle conventions.

**Option B: Compute fresh stats**

```bash
cd /home/chris/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_lora
```

This saves stats to `assets/pi0_ur5e_lora/ChangChrisLiu/ur5e_planner/`. Remove the `AssetsConfig.assets_dir` override in the config to use local stats instead of pre-trained ones.

**Verify normalization stats are sane:**

Check that no dimension has a near-zero `std` or extremely tight `q01`/`q99` range. Dimensions with tiny variance cause huge normalized values and diverging loss.

#### C.5 Train

```bash
cd /home/chris/openpi

# LoRA fine-tuning with pi0 (recommended starting point)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_ur5e_lora \
    --exp-name ur5e_planner_v1 \
    --overwrite

# LoRA fine-tuning with pi0-FAST
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_fast_ur5e_lora \
    --exp-name ur5e_planner_fast_v1 \
    --overwrite
```

**Resume training** (remove `--overwrite`, auto-detects latest checkpoint):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_ur5e_lora \
    --exp-name ur5e_planner_v1
```

**Override hyperparameters from CLI:**

```bash
uv run scripts/train.py pi0_ur5e_lora \
    --exp-name ur5e_planner_v1 \
    --overwrite \
    --num-train-steps 50000 \
    --batch-size 4 \
    --lr-schedule.peak-lr 1e-4
```

**Key training parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--exp-name` | required | Unique experiment name |
| `--overwrite` | False | Overwrite existing checkpoint dir |
| `--num-train-steps` | 30,000 | Total training steps |
| `--batch-size` | 32 | Reduce to 2-4 for 24 GB GPU |
| `--save-interval` | 1000 | Checkpoint save interval |
| `--log-interval` | 100 | W&B logging interval |
| `--fsdp-devices` | 1 | Set >1 for multi-GPU FSDP |

**Checkpoints saved to:**
```
checkpoints/pi0_ur5e_lora/ur5e_planner_v1/<step>/
├── params/           # Model parameters
├── config.pkl        # Training config
├── data_config.pkl   # Dataset metadata
└── wandb_id.txt      # W&B run ID
```

**W&B integration:** Training automatically logs to W&B (loss, gradient norm, sample images). Set `WANDB_PROJECT` and `WANDB_ENTITY` environment variables, or configure in the TrainConfig.

#### C.6 Model Selection: pi0 vs pi0-FAST vs pi0.5

| Feature | pi0 | pi0-FAST | pi0.5 |
|---------|-----|----------|-------|
| Architecture | Flow matching | Autoregressive (FAST tokenizer) | Flow matching (upgraded) |
| Action chunk size | 50 steps | 10 steps | 50 steps |
| Internal action dim | 32 (auto-padded) | 7 (explicit) | 32 (auto-padded) |
| LoRA VRAM | ~22.5 GB | ~22.5 GB | ~22.5 GB |
| Single-arm performance | Good | Good | Mixed (see note) |
| Base checkpoint | `pi0_base` | `pi0_fast_base` | `pi05_base` |

> **Note:** Community reports suggest pi0 may outperform pi0.5 on single-arm tasks ([GitHub #692](https://github.com/Physical-Intelligence/openpi/issues/692)). Start with pi0 LoRA, try pi0-FAST if you prefer shorter action horizons.

#### C.7 Serve Fine-Tuned Policy

```bash
cd /home/chris/openpi

# Start policy server (WebSocket, port 8000)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_ur5e_lora \
    --policy.dir checkpoints/pi0_ur5e_lora/ur5e_planner_v1/30000
```

**Client code for UR5e inference:**

```python
# Install client: pip install -e openpi/packages/openpi-client
from openpi_client import websocket_client_policy as wcp
from openpi_client import image_tools

# Connect to server
client = wcp.WebsocketClientPolicy(host="localhost", port=8000)

# Send observation, get action chunk
observation = {
    "observation/image": image_tools.resize_with_pad(base_rgb, 224, 224),
    "observation/wrist_image": image_tools.resize_with_pad(wrist_rgb, 224, 224),
    "observation/state": state_7d,  # [q0-q5, gripper_norm] - unnormalized
    "prompt": "Pick up the CPU and place it in the socket",
}

result = client.infer(observation)
action_chunk = result["actions"]  # shape: (action_horizon, 7)
# action_chunk[0] = immediate next action [q0..q5, gripper]
# action_chunk[1:] = predicted future actions
```

**Action chunking strategy:** The model returns a full action chunk (50 steps for pi0, 10 for pi0-FAST). Execute N steps open-loop, then re-query with a fresh observation:

```python
chunk = client.infer(obs)["actions"]
for i in range(execute_steps):
    send_to_robot(chunk[i])  # absolute joint positions + gripper
# Re-query with fresh observation
```

State and images should be sent **unnormalized** — the server handles normalization internally.

#### C.8 Troubleshooting (OpenPI)

| Issue | Solution |
|-------|---------|
| OOM during LoRA training | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, reduce batch size to 1-2 |
| Diverging loss | Check norm stats — dimensions with tiny std cause huge normalized values |
| CUDA error on RTX 5090 | Ensure jaxlib >= 0.4.38 (JAX 0.5.3+ works, verified) |
| Action dimension mismatch | pi0: uses `action_dim=32` internally (auto-padded). pi0-FAST: set `action_dim=7` explicitly |
| Strange robot movements | Verify `DeltaActions` mask: `make_bool_mask(6, -1)` = delta for 6 joints, absolute for gripper |
| Cannot resume training | Remove `--overwrite` flag — auto-detects latest checkpoint |
| Missing norm stats | Run `compute_norm_stats.py` or use pre-trained stats via `asset_id="ur5e"` |
| `ModuleNotFoundError` | Run from the openpi root: `cd /home/chris/openpi && uv run scripts/train.py ...` |

---

## License

This project is licensed under the MIT License.
