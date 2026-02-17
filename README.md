# DataCollection: VLA Dual-Dataset Collection Pipeline

A teleoperation data collection system for training Vision-Language-Action (VLA) models, built on the [GELLO](https://github.com/wuphilipp/gello_software) framework. Features a distributed multi-process architecture with dual-camera capture, HOSAS joystick control, automated skill execution with interrupt/resume, and dual-dataset output for training both a high-level planner and a low-level executor.

## System Overview

Each data collection episode produces **two datasets** from a single round of teleoperation:

| Dataset | Contents | Purpose |
|---------|----------|---------|
| **VLA Planner** (`vla_planner/`) | Teleop frames + 3 stop-signal frames (gripper=255) | Teaches the model WHEN to call a skill |
| **VLA Executor** (`vla_executor/`) | Teleop frames + full skill execution frames | Teaches the model the COMPLETE motion trajectory |

## Hardware

| Component | Model | Connection |
|-----------|-------|------------|
| Robot | UR5e | RTDE at `10.125.144.209` |
| Gripper | Robotiq 2F-85 | Socket port 63352 (positions 0-255) |
| Wrist Camera | Intel RealSense D435i | USB, 1280x720 @ 30Hz (RGB + aligned depth) |
| Base Camera | Luxonis OAK-D Pro | USB, 1280x720 @ 30Hz (RGB + aligned depth) |
| Controller | Thrustmaster SOL-R2 HOSAS | USB, dual flight sticks |

## Architecture

Four independent processes communicate over ZMQ, all running at **30Hz**:

```
Terminal 1 ─ Robot Server
  ├── Port 6001: Control server (moveL, speedL, servoJ, speed_stop)
  └── Port 6002: Read-only observation server (get_observations, get_tcp_pose_raw)

Terminal 2 ─ Camera Publishers (PUB/SUB)
  ├── Port 5000: Wrist camera (RealSense D435i)
  └── Port 5001: Base camera (OAK-D Pro)

Terminal 3 ─ Data Collection Pipeline
  └── Connects to T1 + T2, runs joystick agent, skill executor, dataset writer
```

The dual-port robot architecture (6001 control + 6002 read-only) allows observation polling to remain responsive even during blocking skill execution. Both servers wrap the same robot instance but access different RTDE interfaces (Control vs Receive), which is thread-safe.

## Installation

```bash
git clone https://github.com/ChangChrisLiu/DataCollection.git
cd DataCollection
```

### Virtual Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11
source .venv/bin/activate
git submodule init && git submodule update
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python
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

---

## Data Collection Pipeline

### Step 0: Calibrate Camera Settings (One-Time)

Lock exposure, white balance, and gain to prevent out-of-distribution (OOD) issues across episodes. Run this once before each session (or when lighting changes).

**Auto mode** (recommended, no GUI needed):

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
- Port 6001: Full control (moveL, speedL, servoJ, speed_stop, set_freedrive_mode)
- Port 6002: Read-only observations (get_observations, get_tcp_pose_raw, get_joint_state)

### Step 2: Terminal 2 - Camera Servers

```bash
python experiments/launch_camera_nodes.py --camera-settings configs/camera_settings.json
```

Starts async PUB/SUB camera publishers at 30Hz. The `--camera-settings` flag loads and applies the fixed camera parameters captured in Step 0.

> **Warning:** Running without `--camera-settings` uses auto-exposure/WB, which causes inconsistent lighting across episodes.

### Step 3: Terminal 3 - Data Collection

```bash
python experiments/run_dual_dataset.py
```

Optional arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/vla_dataset` | Output directory for episodes |
| `--cpu-skill-csv` | `CPU_Skills.csv` | Path to CPU extraction skill CSV |
| `--ram-skill-csv` | `RAM_Skills.csv` | Path to RAM extraction skill CSV |
| `--cpu-relative-count` | `19` | Number of relative waypoints in CPU skill |
| `--ram-relative-count` | `14` | Number of relative waypoints in RAM skill |
| `--skill-move-speed` | `0.1` | moveL speed during skill (m/s) |
| `--skill-move-accel` | `0.04` | moveL acceleration during skill (m/s^2) |
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
2. **Press Left Btn 25** — recording starts at 30Hz (camera-gated)
3. **Teleoperate** to approach and align the gripper with the component
4. **Press Right Btn 15** (CPU) or **Right Btn 16** (RAM) to trigger the skill:
   - 3 stop-signal frames are automatically inserted (same pose, gripper=255)
   - Skill executes autonomously using async moveL with concurrent 30Hz recording
5. **After skill completes**, press **Left Btn 34** (Home) to:
   - Save both datasets (vla_planner + vla_executor)
   - Rate episode quality (g = Good / n = Not Good)
   - Move robot to home position
6. **Repeat** from step 1 for the next episode

### Interrupt/Resume Flow (If Skill Fails Mid-Execution)

1. During skill execution, press **Left Btn 16** to interrupt
2. Robot stops immediately via `speed_stop()` (async moveL enables mid-waypoint stopping)
3. Manually correct position with joystick (frames continue recording as skill data)
4. Press the **same skill button** to resume:
   - Skips relative waypoints (already completed)
   - Executes only absolute (base-frame) waypoints
5. If correction is impossible, press **Left Btn 34** (Home) to abandon and save what you have

---

## Skill System

Skills replay pre-recorded manipulation trajectories from CSV files. Each CSV contains waypoints with joint angles, TCP poses, and gripper positions.

### CSV Format

| Column | Description |
|--------|-------------|
| timestamp | Recording timestamp |
| j0-j5 | 6 joint angles (radians) |
| tcp_x-tcp_rz | TCP pose [x, y, z, rx, ry, rz] in base frame |
| gripper_pos | Gripper position (0-255) |
| skill_id | Skill identifier |
| image_file | Associated image path |

### Relative vs Absolute Waypoints

Each skill CSV is split into two sections:

- **Relative section** (first N waypoints): Applied relative to the trigger TCP pose using SE(3) transforms. This makes the manipulation work regardless of where the robot is when the skill is triggered.
  - Transform: `T_target = T_trigger @ inv(T_skill_origin) @ T_waypoint`

- **Absolute section** (remaining waypoints): Used as-is in base-frame coordinates. Typically the transfer-to-destination phase.

### Current Skills

| Skill | Button | CSV File | Relative Count | Total Waypoints |
|-------|--------|----------|----------------|-----------------|
| CPU Extraction | Right 15 | `CPU_Skills.csv` | 19 | 23 |
| RAM Extraction | Right 16 | `RAM_Skills.csv` | 14 | 20 |

---

## Output Data Format

```
data/vla_dataset/
  episode_0217_143000/
    vla_planner/
      frame_0000_20260217_143000_000000.pkl
      frame_0001_20260217_143000_033333.pkl
      ...
      episode_meta.json
    vla_executor/
      frame_0000_20260217_143000_000000.pkl
      frame_0001_20260217_143000_033333.pkl
      ...
      episode_meta.json
```

Each `.pkl` frame contains:

```python
{
    "timestamp": float,                    # Unix timestamp
    "joint_positions": [j0, ..., j5],      # 6 joint angles (radians)
    "tcp_pose": [x, y, z, rx, ry, rz],    # TCP pose, UR rotation vector
    "gripper_pos": int,                    # 0-255 (255 = stop signal)
    "wrist_rgb": np.ndarray,              # (720, 1280, 3) uint8
    "wrist_depth": np.ndarray,            # (720, 1280, 1) uint16
    "base_rgb": np.ndarray,               # (720, 1280, 3) uint8
    "base_depth": np.ndarray,             # (720, 1280, 1) uint16
}
```

### Metadata (`episode_meta.json`)

```json
{
    "phase": "post_skill",
    "num_teleop_frames": 150,
    "num_stop_frames": 3,
    "num_skill_frames": 45,
    "stop_signal_value": 255,
    "type": "vla_planner",
    "num_frames_saved": 153,
    "saved_at": "2026-02-17T14:30:15"
}
```

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
│   └── run_dual_dataset.py           # T3: Main collection pipeline
├── gello/
│   ├── agents/                       # Control agents
│   │   ├── agent.py                  # Agent protocol (Action type)
│   │   ├── joystick_agent.py         # HOSAS dual-stick (velocity control)
│   │   ├── gello_agent.py            # GELLO Dynamixel (joint-space)
│   │   ├── spacemouse_agent.py       # 3Dconnexion SpaceMouse
│   │   └── zmq_agent.py              # ZMQ client wrapper
│   ├── cameras/                      # Camera drivers
│   │   ├── camera.py                 # CameraDriver protocol
│   │   ├── realsense_camera.py       # Intel RealSense D435i
│   │   └── oakd_camera.py            # Luxonis OAK-D Pro
│   ├── robots/                       # Robot interfaces
│   │   ├── robot.py                  # Robot protocol
│   │   └── ur.py                     # UR5e (RTDE, async moveL support)
│   ├── skills/                       # Skill execution
│   │   └── csv_skill_executor.py     # CSV waypoint replay with interrupt
│   ├── data_utils/                   # Data pipeline
│   │   ├── dual_dataset_buffer.py    # Teleop + skill frame management
│   │   └── dataset_writer.py         # Pickle + metadata persistence
│   ├── zmq_core/                     # ZMQ networking
│   │   ├── robot_node.py             # Robot server/client (dual-port)
│   │   └── camera_node.py            # Camera PUB/SUB client
│   ├── utils/                        # Utilities
│   │   └── transform_utils.py        # SE(3) transforms for skill replay
│   └── env.py                        # RobotEnv (rate-limited step loop)
├── scripts/                          # Calibration & testing
│   ├── calibrate_cameras.py          # Camera settings auto-detect/manual
│   └── test_dual_camera.py           # Camera connectivity test
├── configs/                          # Generated config files
│   └── camera_settings.json          # Saved camera parameters (auto-generated)
├── joysticktst.py                    # Standalone HOSAS test (direct RTDE)
├── CPU_Skills.csv                    # CPU extraction skill waypoints
├── RAM_Skills.csv                    # RAM extraction skill waypoints
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
