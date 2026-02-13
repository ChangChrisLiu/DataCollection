# DataCollection: General-Purpose Teleoperation Data Collection

Originally based on the [GELLO](https://github.com/wuphilipp/gello_software) teleoperation framework, this repository has been extensively modified into a **general-purpose teleoperation data collection system**. It features a distributed multi-process architecture, dual-camera capture with timestamp synchronization, and support for multiple input agents including GELLO, joystick, and SpaceMouse.

## Architecture

The system runs as four independent processes communicating over ZMQ:

```
Terminal 1 (T1) ─ Robot Server        ──► ZMQ port 6001
Terminal 2 (T2) ─ Camera Publishers   ──► PUB ports 5000 (wrist) / 5001 (base)
Terminal 3 (T3) ─ Agent Server        ──► ZMQ port 6000
Terminal 4 (T4) ─ Control Loop        ──► Connects to T1, T2, T3
```

This decoupled design allows each component to run at its own rate and be restarted independently.

## Supported Hardware

### Robots
- **Universal Robots (UR5e, etc.)** via [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html)
- **Franka Panda / FR3** via Polymetis or ROS 2 (see [ROS 2 README](ros2/README.md))

### Cameras (Dual-Camera Setup)
- **Intel RealSense D435** — Wrist camera (424x240 RGB + aligned depth)
- **Luxonis OAK-D Pro** — Base camera (416x240 RGB + aligned depth)

Both cameras stream via async PUB/SUB with timestamps for synchronization.

### Control Agents
- **GELLO** — Dynamixel-based teleoperation device (joint-space mirroring)
- **Joystick (Thrustmaster SOL-R2 HOSAS)** — Dual flight stick with Cartesian end-effector control via IK
- **SpaceMouse** — 3Dconnexion 6-DOF input device (Cartesian end-effector control)

## Installation

```bash
git clone https://github.com/ChangChrisLiu/DataCollection.git
cd DataCollection
```

### Virtual Environment (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv --python 3.11
source .venv/bin/activate
git submodule init && git submodule update
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python
```

### Robot-Specific Dependencies

```bash
# UR robots
pip install ur_rtde

# Franka (if using Polymetis)
# See: https://facebookresearch.github.io/fairo/polymetis/installation.html

# OAK-D camera
pip install depthai

# Joystick support
pip install pygame
```

## Quick Start

### 1. Launch Robot Server (T1)

```bash
# UR robot (real hardware)
python experiments/launch_nodes.py --robot ur

# UR robot (simulation)
python experiments/launch_nodes.py --robot sim_ur
```

### 2. Launch Camera Publishers (T2)

```bash
python experiments/launch_camera_nodes.py
```

This auto-detects connected cameras (RealSense + OAK-D) and starts PUB streaming on ports 5000/5001.

### 3. Launch Agent Server (T3) — GELLO only

Only needed when using the GELLO teleoperation device. Joystick and SpaceMouse run directly inside T4.

```bash
python experiments/launch_gello_node.py
```

### 4. Launch Control Loop (T4)

```bash
# GELLO teleoperation (requires T3)
python experiments/run_env.py --agent=gello

# Joystick control (no T3 needed)
python experiments/run_env.py --agent=joystick

# SpaceMouse control (no T3 needed)
python experiments/run_env.py --agent=spacemouse

# With data collection (works with any agent)
python experiments/run_env.py --agent=gello --use-save-interface

# Without cameras (for testing)
python experiments/run_env.py --agent=joystick --no-cameras
```

## Data Collection

### Recording

Add `--use-save-interface` to the control loop to enable recording:

```bash
python experiments/run_env.py --agent=gello --use-save-interface
```

**Controls during recording:**
- `S` — Start recording a trajectory
- `Q` — Stop recording
- After stopping, you are prompted to mark the trajectory as **Good** or **Not Good**

Data is saved at camera frame rate (~30Hz) as timestamped pickle files:
```
data/<agent>/<date_time>/frame_NNNN_<timestamp>.pkl
```

Each frame contains:
```python
{
    "obs": {
        "wrist_rgb": np.ndarray,        # (240, 424, 3)
        "wrist_depth": np.ndarray,      # (240, 424, 1)
        "wrist_timestamp": float,
        "base_rgb": np.ndarray,         # (240, 416, 3)
        "base_depth": np.ndarray,       # (240, 416, 1)
        "base_timestamp": float,
        "joint_positions": np.ndarray,
        "joint_velocities": np.ndarray,
        "ee_pos_quat": np.ndarray,
        "gripper_position": np.ndarray,
    },
    "action": np.ndarray,               # commanded joint angles
}
```

### Post-Processing

Convert collected pickle data to HDF5 trajectories with visualization:

```bash
python gello/data_utils/demo_to_gdict.py --source-dir=data
```

Output:
```
data/_conv/multiview/
├── train/none/traj_*.h5          # Training trajectories
├── val/none/traj_*.h5            # Validation trajectories (10%)
└── vis/
    ├── rgb/traj_*_rgb_*.mp4      # Multi-view video playback
    ├── depth/traj_*_depth_*.mp4  # Depth visualization
    ├── state/traj_*_states.png   # Joint state plots
    └── action/traj_*_actions.png # Action plots
```

## Standalone Camera Capture

For offline data collection without robot control (e.g., scene scanning):

```bash
# Interactive mode — capture on demand
python nostream.py

# Batch mode — timed captures
python nostream.py --batch 10 5.0
```

This uses persistent OAK-D Pro connections with 12MP still capture, aligned depth, and point cloud generation.

## GELLO Hardware Setup

### Offset Calibration

Set your GELLO to a known configuration matching the robot, then run:

**UR Robot:**
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/<your-gello-port>
```

**Franka FR3:**
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 -1.57 0 1.57 0 \
    --joint-signs 1 1 1 1 1 -1 1 \
    --port /dev/serial/by-id/<your-gello-port>
```

Add the generated offsets to `gello/agents/gello_agent.py` in the `PORT_CONFIG_MAP`.

### Joint Signs Reference

| Robot | Signs |
|-------|-------|
| UR | `1 1 -1 1 1 1` |
| Franka FR3 | `1 1 1 1 1 -1 1` |

## Thrustmaster SOL-R2 HOSAS Setup

The joystick agent is designed for the **Thrustmaster SOL-R2 HOSAS** dual flight stick controller. It uses Cartesian velocity control with MuJoCo IK, similar to the SpaceMouse agent.

### Controller Mapping

```
LEFT STICK                          RIGHT STICK
┌────────────────────┐              ┌────────────────────┐
│  Axis 0,1: X/Y     │              │  Axis 0,1: Rx/Ry   │
│  (TCP translation)  │              │  (TCP rotation)     │
│                     │              │  Axis 2 (twist): Rz │
│  Axis 3 (slider):  │              │  Axis 3 (mini): Z   │
│   speed gain        │              │   (TCP up/down)     │
│                     │              │                     │
│  Mini-stick Y:      │              │  Button 2:          │
│   gripper open/close│              │   vertical reorient │
│                     │              │  Button 3:          │
│  Button 0 (trigger):│              │   go to home        │
│   toggle recording  │              │                     │
└────────────────────┘              └────────────────────┘
```

### Usage

```bash
# Basic joystick teleoperation
python experiments/run_env.py --agent=joystick

# With data collection
python experiments/run_env.py --agent=joystick --use-save-interface

# With verbose output (shows IK status and skill triggers)
python experiments/run_env.py --agent=joystick --verbose
```

The base slider on each stick controls speed gain (min 10% at zero, 100% at max). The `HOSASConfig` dataclass in `gello/agents/joystick_agent.py` contains all tunable parameters including speed limits, deadzone, and axis mappings.

### Skills

- **Home** (Right Button 3): Returns the robot to a safe home joint configuration
- **Vertical Reorient** (Right Button 2): Forces the end-effector to point straight down

### Swap Left/Right Sticks

If your OS enumerates the sticks in reverse order, swap the indices in `run_env.py`:

```python
agent_cfg = {
    "_target_": "gello.agents.joystick_agent.JoystickAgent",
    "left_index": 1,   # swap
    "right_index": 0,  # swap
}
```

## Code Organization

```
├── experiments/              # Launch scripts
│   ├── run_env.py            # Main control loop (T4)
│   ├── launch_nodes.py       # Robot server (T1)
│   ├── launch_camera_nodes.py# Camera publishers (T2)
│   └── launch_gello_node.py  # Agent server (T3)
├── gello/
│   ├── agents/               # Control agents
│   │   ├── agent.py          # Agent protocol
│   │   ├── gello_agent.py    # GELLO Dynamixel agent
│   │   ├── joystick_agent.py # Thrustmaster HOSAS dual-stick (Cartesian IK)
│   │   ├── zmq_agent.py      # ZMQ client agent (connects to T3)
│   │   └── spacemouse_agent.py
│   ├── cameras/              # Camera drivers
│   │   ├── realsense_camera.py  # Intel RealSense D435
│   │   └── oakd_camera.py       # Luxonis OAK-D Pro
│   ├── robots/               # Robot interfaces
│   │   ├── ur.py             # Universal Robots
│   │   └── panda.py          # Franka Panda/FR3
│   ├── data_utils/           # Data pipeline (pickle → HDF5)
│   ├── dynamixel/            # Dynamixel servo driver
│   ├── zmq_core/             # ZMQ camera/robot networking
│   └── utils/                # Control loop & save interface
├── ros2/                     # ROS 2 packages for Franka FR3
├── configs/                  # YAML robot/agent configurations
├── nostream.py               # Standalone dual-camera capture
└── scripts/                  # Calibration utilities
```

## Citation

This project is built on top of the GELLO framework:

```bibtex
@misc{wu2023gello,
    title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
    author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
    year={2023},
}
```

## License

This project is licensed under the MIT License.
