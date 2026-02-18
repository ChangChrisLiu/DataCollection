# DataCollection: VLA Data Collection Pipeline for UR5e

A teleoperation data collection system for training Vision-Language-Action (VLA) models, built on the [GELLO](https://github.com/wuphilipp/gello_software) framework. Features a distributed multi-process architecture with dual-camera capture, HOSAS joystick control, automated skill execution with path blending and grasp verification, and a unified phase-labeled recording format that produces training data for three model architectures: **OpenVLA**, **OpenVLA-OFT**, and **OpenPI**.

## System Overview

Each data collection episode records a single, continuous frame stream annotated with **4 phase labels**:

| Phase | Description | When |
|-------|-------------|------|
| `teleop` | Human joystick control (approach) | Operator maneuvers to target |
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
  ├── Port 6001: Control server (moveL, speedL, move_linear_path, speed_stop)
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
- Port 6001: Full control (moveL, move_linear_path, speedL, servoJ, speed_stop, set_freedrive_mode)
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
| `--cpu-skill-csv` | `CPU_Skills.csv` | Path to CPU extraction skill CSV |
| `--ram-skill-csv` | `RAM_Skills.csv` | Path to RAM extraction skill CSV |
| `--cpu-relative-count` | `20` | Number of relative waypoints in CPU skill |
| `--ram-relative-count` | `15` | Number of relative waypoints in RAM skill |
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
  Btn 25: Start recording                 Btn 15: Trigger CPU skill
  Btn 34: Home + stop + save              Btn 16: Trigger RAM skill
  Btn 38: Vertical reorient               Btn 17: Connector skill (reserved)
  Btn 16: Interrupt active skill          Btn 18: Skill 4 (reserved)
```

The left slider controls a speed gain multiplier: fully up = 100% speed, fully down = 10% speed. All axes have a 0.05 deadzone and are calibrated at startup.

---

## Episode Workflow

### Normal Data Collection (Per Episode)

1. **Position the robot** near the target object using joystick teleop
2. **Press Left Btn 25** — recording starts at 30Hz (camera-gated), phase: `teleop`
3. **Teleoperate** to approach and align the gripper with the component
4. **Press Right Btn 15** (CPU) or **Right Btn 16** (RAM) to trigger the skill:
   - Phase transitions to `skill`
   - Skill executes autonomously with path blending and 30Hz concurrent recording
   - **Grasp verification** runs automatically (see below)
5. **After skill completes**, press **Left Btn 34** (Home) to:
   - Save unified episode (all frames + phase metadata)
   - Rate episode quality (g = Good / n = Not Good)
   - Move robot to home position
6. **Repeat** from step 1 for the next episode

### Grasp Verification (Automatic)

Each skill CSV contains a verification waypoint that lifts the gripper 5cm with fingers open after the grasp close. The executor checks the actual gripper position:

- **Pass** (actual < 200/255): Object is held. Skill continues normally.
- **Fail** (actual >= 200/255): Fingers closed on nothing. Recording pauses automatically and the pipeline waits for correction.

### Correction Flow (After Grasp Failure)

1. Grasp verification fails — recording stops, message: `"GRASP FAILED — provide correction with joystick"`
2. Move joystick to correct position — recording resumes with phase: `correction`
3. Press the **same skill button** to resume:
   - Phase transitions to `skill_resume`
   - Only absolute (base-frame) waypoints execute (relative already completed)
4. After skill completes, press **Home** to save

### Manual Interrupt Flow

1. During skill execution, press **Left Btn 16** to interrupt
2. Robot stops immediately via `speed_stop()`, phase transitions to `correction`
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

Consecutive waypoints with the same gripper value are grouped into path segments and executed as a single `moveL(path)` call with blend radii for smooth trajectories:

- **Default blend radius**: 0.002m (2mm)
- **First 2 waypoints**: 0.0001m (0.1mm) — sensitive approach
- **First/last of segment**: 0.0m (exact stop)
- **Near gripper changes**: 0.0m (exact positioning)

### Current Skills

| Skill | Button | CSV File | Relative Count | Total Waypoints |
|-------|--------|----------|----------------|-----------------|
| CPU Extraction | Right 15 | `CPU_Skills.csv` | 20 | 23 |
| RAM Extraction | Right 16 | `RAM_Skills.csv` | 15 | 19 |

---

## Output Data Format

### Unified Episode Structure

```
data/vla_dataset/
  episode_0218_143000/
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

During conversion, the following processing is applied:
1. **Phase filtering** — select frames by phase label(s)
2. **Stop signal synthesis** — 3 copies of last frame with gripper=255
3. **No-op frame removal** — drop frames where joints haven't changed (threshold: 1e-4 rad)
4. **Delta joint computation** — for RLDS action format
5. **Image resizing** — 256x256 for RLDS (configurable)

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

### Dataset Registration (OpenVLA / OpenVLA-OFT)

The three RLDS datasets are registered in both `/home/chris/Sibo/openvla/` and `/home/chris/Sibo/openvla-oft/`:

- **`prismatic/vla/datasets/rlds/oxe/configs.py`** — dataset configurations (JOINT state encoding, JOINT_POS action encoding, image keys)
- **`prismatic/vla/datasets/rlds/oxe/transforms.py`** — reuses `ur5e_disassembly_dataset_transform` (takes 6D delta joints + appends inverted gripper from state → 7D action)

---

## Standalone Testing

### joysticktst.py

A standalone test script that connects directly to the robot via RTDE (no ZMQ) for validating joystick mappings and skill execution before using the full pipeline.

```bash
python joysticktst.py
```

Features:
- Direct RTDE control (no server processes needed)
- Full HOSAS mapping with calibration
- Skill execution with interrupt/resume
- CSV waypoint recording
- Camera preview (optional, RealSense only)

---

## Code Organization

```
├── experiments/                      # Launch scripts
│   ├── launch_nodes.py               # T1: Robot ZMQ server (dual-port)
│   ├── launch_camera_nodes.py        # T2: Camera PUB/SUB publishers
│   └── run_collection.py             # T3: Unified collection pipeline
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
│   │   └── ur.py                     # UR5e (RTDE, moveL path blending)
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
├── joysticktst.py                    # Standalone HOSAS test (direct RTDE)
├── CPU_Skills.csv                    # CPU extraction skill (20 rel + 3 abs)
├── RAM_Skills.csv                    # RAM extraction skill (15 rel + 4 abs)
└── ros2/                             # ROS 2 packages for Franka FR3
```

## Frequency Summary

All components run at **30Hz**, synchronized by camera frame rate:

| Component | Rate | Notes |
|-----------|------|-------|
| Camera publishers | 30Hz | Hardware-limited (D435i + OAK-D Pro) |
| Pipeline main loop | 30Hz | `env.py` Rate class, camera-frame gated |
| Skill recording thread | 30Hz | Concurrent recording during skill execution |
| Robot observations | 30Hz | Polled at loop rate |
| Joystick input polling | 200Hz | Internal only (responsiveness), consumed at 30Hz |

## Citation

Built on the GELLO framework:

```bibtex
@misc{wu2023gello,
    title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
    author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
    year={2023},
}
```

## License

This project is licensed under the MIT License.
