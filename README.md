# DataCollection: VLA Data Collection & Inference Pipeline for UR5e

A teleoperation data collection system for training Vision-Language-Action (VLA) models, built on the [GELLO](https://github.com/wuphilipp/gello_software) framework. Features a distributed multi-process architecture with dual-camera capture, HOSAS joystick control, automated skill execution with path blending and grasp verification, and a unified phase-labeled recording format that produces training data for three model architectures: **OpenVLA**, **OpenVLA-OFT**, and **OpenPI**.

## System Overview

Each data collection episode records a continuous frame stream annotated with **phase labels**:

| Phase | Description |
|-------|-------------|
| `teleop` | Human joystick control — approach and align with the target |
| `skill` | Autonomous skill execution — grasp, lift, and place |
| `correction` | Human recovery after a failed grasp attempt |
| `skill_resume` | Skill resumes from corrected position |

From this single recording, **three training datasets** are derived during conversion:

| Target | Phases Included | Stop Signal | What the Trained Model Learns |
|--------|----------------|-------------|-------------------------------|
| **End-to-End** (`e2e`) | All 4 phases | None | Full task from approach to placement |
| **Planner** (`planner`) | Teleop only | Appended | When to hand off to a pre-programmed skill |
| **Correction** (`correction`) | Correction only | Appended | How to recover after a failed grasp |

## Hardware

| Component | Model |
|-----------|-------|
| Robot | Universal Robots UR5e |
| Gripper | Robotiq 2F-85 |
| Wrist Camera | Intel RealSense D435i (1280x720 @ 30Hz) |
| Base Camera | Luxonis OAK-D Pro (1280x720 @ 30Hz) |
| Controller | Thrustmaster SOL-R2 HOSAS (dual flight sticks) |

## Architecture

Three independent processes communicate over ZMQ:

```
Terminal 1 — Robot Server
  ├── Port 6001: Control (moveL, speedL, servoJ, skill execution)
  └── Port 6002: Read-only observations (TCP pose, joint state, gripper)

Terminal 2 — Camera Publishers (PUB/SUB)
  ├── Wrist camera stream
  └── Base camera stream

Terminal 3 — Pipeline (Data Collection or Inference)
  └── Connects to T1 + T2, runs agent, skill executor, recording buffer
```

The dual-port robot architecture allows observation polling during blocking skill execution.

## Installation

```bash
git clone https://github.com/ChangChrisLiu/DataCollection.git
cd DataCollection

conda create -n tele python=3.11 -y
conda activate tele

git submodule init && git submodule update
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
```

Additional dependencies:

```bash
pip install ur_rtde pyrealsense2 depthai pygame tyro scipy
```

## Data Collection

### 1. Start the robot server

```bash
python experiments/launch_nodes.py --robot ur
```

### 2. Start camera publishers

```bash
python experiments/launch_camera_nodes.py --camera-settings configs/camera_settings.json
```

Camera settings should be calibrated first with `scripts/calibrate_cameras.py` to lock exposure and white balance.

### 3. Run data collection

```bash
python experiments/run_collection.py
```

**Workflow per episode:** Arm recording (button) &rarr; teleop approach (joystick) &rarr; trigger skill (button) &rarr; auto-save & home.

If the grasp fails, the system enters correction mode automatically. The operator corrects with the joystick, then resumes the skill. Key CLI options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--record-hz` | `30` | Recording frame rate |
| `--cpu-skill-csv` | `CPU_Skills.csv` | Skill trajectory for CPU extraction |
| `--ram-skill-csv` | `RAM_Skills.csv` | Skill trajectory for RAM extraction |
| `--image-size` | `256` | Resize images to NxN |

## Data Conversion

Raw `.pkl` episodes are converted to model-specific formats:

| Script | Output Format | Used By |
|--------|--------------|---------|
| `scripts/convert_to_rlds.py` | RLDS TFRecords | OpenVLA, OpenVLA-OFT |
| `scripts/convert_to_lerobot.py` | LeRobot v2.1 | OpenPI |

Both scripts support `--target` (e2e, planner, correction), `--fps` (10, 30), and `--task` (cpu, ram) flags. Stop signals are synthesized during conversion, not stored in raw recordings.

```bash
# Example: convert to RLDS for the planner target at 10Hz
python scripts/convert_to_rlds.py --target planner --fps 10

# Example: convert to LeRobot for end-to-end at 30Hz
python scripts/convert_to_lerobot.py --target e2e --fps 30
```

## Inference Pipeline

Deploy fine-tuned VLA models on the real robot. The inference script replaces human joystick input with model predictions while keeping the same multi-terminal architecture.

### Supported Backends

| Backend | Action Space | Client Protocol |
|---------|-------------|-----------------|
| **OpenPI** | Absolute joint angles (servoJ) | WebSocket |
| **OpenVLA** | EEF deltas (moveL) | REST |
| **OpenVLA-OFT** | EEF deltas (moveL) | REST |

### Three Inference Modes

| Mode | Description |
|------|-------------|
| `planner` | Model approaches &rarr; stop signal fires &rarr; pre-programmed skill executes |
| `e2e` | Model controls the entire task from approach to placement |
| `correction` | Model corrects a failed grasp &rarr; stop signal &rarr; skill resumes |

### Running Inference

**Terminal 1 & 2** are the same as data collection (robot server + cameras).

**Terminal 3:** Start the model server (backend-specific).

**Terminal 4:** Run the inference client:

```bash
python experiments/run_inference.py \
    --model-type <openpi|openvla|openvla_oft> \
    --mode <planner|e2e|correction> \
    --task <cpu|ram> \
    --fps <10|30>
```

The `--task` flag controls three things: language instruction, skill CSV, and stop detection thresholds.

#### CLI Reference

| Argument | Description |
|----------|-------------|
| `--model-type` | Backend: `openpi`, `openvla`, `openvla_oft` |
| `--mode` | Inference mode: `planner`, `e2e`, `correction` |
| `--task` | Task: `cpu` or `ram` |
| `--fps` | Must match training FPS (10 or 30) |
| `--server-port` | Model server port (default: 8000) |
| `--correction-server-port` | Port for correction model in planner mode |
| `--openpi-base` | OpenPI base model: `droid` or `base` |

The inference script runs a **continuous episode loop** — after each episode, the robot homes and starts the next episode automatically. Press **Ctrl+C** to stop gracefully (in-progress episode is saved).

## Code Organization

```
experiments/
  run_collection.py          # Data collection entry point
  run_inference.py           # Inference pipeline entry point
  launch_nodes.py            # Robot ZMQ server
  launch_camera_nodes.py     # Camera ZMQ publishers

gello/
  robots/ur.py               # UR5e control, gripper, moveL/servoJ
  agents/
    joystick_agent.py        # HOSAS joystick teleoperation
    vla_agent.py             # VLA inference adapters (OpenPI/OpenVLA/OFT)
  skills/csv_skill_executor.py  # Skill replay with path blending
  data_utils/episode_buffer.py  # Phase-labeled recording buffer

scripts/
  convert_to_rlds.py         # RLDS/TFRecord conversion
  convert_to_lerobot.py      # LeRobot conversion
  calibrate_cameras.py       # Camera exposure/WB calibration
```

## License

This project is licensed under the MIT License.
