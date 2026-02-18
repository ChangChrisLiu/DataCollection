from __future__ import annotations

import csv
import datetime
import os
import socket
import threading
import time
from typing import Any

import cv2
import numpy as np
import pygame
import rtde_control
import rtde_receive

try:
    import pyrealsense2 as rs
    VISION_AVAILABLE = True
except ImportError:
    rs: Any = None
    VISION_AVAILABLE = False
    print("[WARNING] pyrealsense2 not found. Camera features disabled.")

# ============================================================================
# CONFIGURATION & MAPPING
# ============================================================================
ROBOT_IP = "10.125.144.209"
GRIPPER_PORT = 63352

# Kinematic constraints
MAX_SPEED_L = 0.05   # Max translation speed (m/s)
MAX_SPEED_R = 0.10   # Max rotation speed (rad/s) for Rx, Ry
MAX_SPEED_RZ = 0.25  # Max Rz twist speed (rad/s)
ACCELERATION = 0.5   # Acceleration (rad/s^2)
DT = 0.01            # Control loop period (10ms / 100Hz)
WATCHDOG_TIME = 0.1  # Safety watchdog for speedL

DEADZONE_VAL = 0.05
MIN_GAIN = 0.1       
GRIPPER_STEP = 10

HOME_JOINTS = [-1.613282, -1.802157, -1.189979, -1.72829, 1.552156, 3.058179]
HOME_GRIPPER_POS = 3

# --- Input Mappings ---
# Adjust these indices if Pygame enumerates your axes/buttons differently
L_AXIS_X = 0        # Left Stick X (TCP X)
L_AXIS_Y = 1        # Left Stick Y (TCP Y)
L_AXIS_SLIDER = 2   # Left Slider (Speed) — range [0=up, 1=down]

R_AXIS_Z = 1        # Right Stick Push/Pull (TCP Z)
R_AXIS_RZ = 5       # Right Stick Twist (TCP Rz)
R_AXIS_MINI_X = 3   # Right Mini-stick left/right (TCP Rx)
R_AXIS_MINI_Y = 4   # Right Mini-stick up/down (TCP Ry)
L_AXIS_MINI_Y = 4   # Left Mini-stick up/down (Gripper)

# Discrete Buttons (Left)
L_BTN_REC_START = 25
L_BTN_WAYPOINT = 23
L_BTN_HOME = 34
L_BTN_VERT = 38
L_BTN_INTERRUPT = 16

# Discrete Buttons (Right)
R_BTN_REC_STOP = 25
R_BTN_UNDO = 4
R_BTN_PAUSE = 38
R_BTN_SKILLS = [15, 16, 17, 18]


# ============================================================================
# SE(3) TRANSFORM HELPERS
# ============================================================================

def pose6d_to_homogeneous(pose):
    """Convert [x,y,z,rx,ry,rz] to 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    rotvec = np.array(pose[3:6], dtype=np.float64)
    R, _ = cv2.Rodrigues(rotvec)
    T[:3, :3] = R
    return T


def homogeneous_to_pose6d(T):
    """Convert 4x4 homogeneous matrix to [x,y,z,rx,ry,rz]."""
    pos = T[:3, 3]
    R = T[:3, :3]
    rotvec, _ = cv2.Rodrigues(R)
    return list(pos) + list(rotvec.flatten())


def load_skill_csv(csv_path):
    """Load waypoints from a skill CSV.

    Returns list of (joints, tcp_pose, gripper_pos) tuples.
    """
    waypoints = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 14:
                continue
            joints = [float(row[i]) for i in range(1, 7)]
            tcp = [float(row[i]) for i in range(7, 13)]
            gripper = int(float(row[13]))
            waypoints.append((joints, tcp, gripper))
    return waypoints


# ============================================================================
# SKILL CONFIGURATION
# ============================================================================
CPU_SKILL_CSV = "CPU_Skills.csv"
CPU_RELATIVE_COUNT = 20  # First 20 waypoints are relative to trigger pose (incl. verification WP)

RAM_SKILL_CSV = "RAM_Skills.csv"
RAM_RELATIVE_COUNT = 15  # First 15 waypoints are relative to trigger pose (incl. verification WP)

SKILL_MOVE_SPEED = 0.1   # m/s for moveL during skill
SKILL_MOVE_ACCEL = 0.04   # m/s^2 for moveL during skill
SKILL_GRIPPER_SPEED = 30 # Gripper speed during skill (0-255, lower = slower)


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class CameraHandler:
    """Handles RealSense D435i connection, streaming, and PNG saving."""
    def __init__(self, width=1920, height=1080, fps=30):
        self.active = False
        self.pipeline = None
        self.width = width
        self.height = height
        self.fps = fps
        self.current_frame = None

    def connect(self):
        if not VISION_AVAILABLE: return False
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            print(f"[CAMERA] Starting RealSense ({self.width}x{self.height})...")
            self.pipeline.start(config)
            
            for _ in range(15):
                self.pipeline.wait_for_frames()
            
            self.active = True
            print("[CAMERA] Connected successfully.")
            return True
        except Exception as e:
            print(f"[CAMERA] Failed to connect: {e}. Running without camera.")
            self.active = False
            return False

    def update_frame(self):
        if not self.active or self.pipeline is None: return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if color_frame:
                self.current_frame = np.asanyarray(color_frame.get_data())
            return self.current_frame
        except Exception:
            return None

    def save_png(self, folder_path, filename):
        if not self.active or self.current_frame is None: return False
        full_path = os.path.join(folder_path, filename)
        cv2.imwrite(full_path, self.current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return True

    def stop(self):
        if self.active and self.pipeline is not None:
            self.pipeline.stop()


class GripperHandler:
    """Handles Robotiq 2F-85 connection via socket over port 63352."""
    def __init__(self):
        self.socket = None
        self.lock = threading.Lock()
        self.connected = False
        self.current_pos = 0

    def connect(self, ip, port=63352):
        print(f"[GRIPPER] Connecting to {ip}:{port}...")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((ip, port))
            self.connected = True
            try: self.socket.recv(1024) 
            except: pass
            print("[GRIPPER] Connected successfully.")
            return True
        except Exception as e:
            print(f"[GRIPPER] Failed to connect: {e}.")
            self.connected = False
            return False

    def _drain_response(self):
        """Read and discard any pending data from the gripper socket."""
        if self.socket is None: return
        try:
            self.socket.setblocking(False)
            while True:
                try:
                    self.socket.recv(1024)
                except BlockingIOError:
                    break
        except Exception:
            pass
        finally:
            self.socket.setblocking(True)

    def _send_cmd(self, cmd):
        if not self.connected or self.socket is None: return None
        try:
            with self.lock:
                self._drain_response()
                data = (cmd + '\n').encode('ascii')
                self.socket.sendall(data)
                time.sleep(0.01)
                self.socket.settimeout(0.1)
                try:
                    resp = self.socket.recv(1024).decode('ascii').strip()
                except (socket.timeout, OSError):
                    resp = None
                self.socket.settimeout(2.0)
                return resp
        except Exception as e:
            print(f"[GRIPPER] Comm error: {e}")
            return None

    def get_actual_pos(self):
        """Read the actual gripper position from the robot (0-255)."""
        if not self.connected: return self.current_pos
        resp = self._send_cmd("GET POS")
        if resp and resp.startswith("POS"):
            try:
                return int(resp.split()[1])
            except (IndexError, ValueError):
                pass
        return self.current_pos

    def activate(self):
        if not self.connected: return
        print("[GRIPPER] Activating...")
        self._send_cmd("SET ACT 0")
        time.sleep(0.5)
        self._send_cmd("SET ACT 1")
        time.sleep(2.0)
        self._send_cmd("SET GTO 1")
        time.sleep(0.3)
        self._send_cmd("SET SPE 255")
        self._send_cmd("SET FOR 255")
        # Read initial position
        self._send_cmd("GET POS")
        time.sleep(0.1)
        self._drain_response()
        print("[GRIPPER] Activation complete.")

    def set_speed(self, speed):
        """Set gripper finger speed (0-255). Lower = slower."""
        if not self.connected: return
        speed = max(0, min(255, int(speed)))
        self._send_cmd(f"SET SPE {speed}")

    def move(self, pos):
        if not self.connected: return
        self.current_pos = max(0, min(255, int(pos)))
        self._send_cmd(f"SET POS {self.current_pos}")

    def move_and_wait(self, pos, tolerance=5):
        """Move gripper and block until it reaches target or stalls on object."""
        if not self.connected:
            return
        target = max(0, min(255, int(pos)))
        self.move(target)
        last_pos = -1
        stall_count = 0
        while True:
            actual = self.get_actual_pos()
            if abs(actual - target) <= tolerance:
                break
            if actual == last_pos:
                stall_count += 1
                if stall_count >= 3:
                    break
            else:
                stall_count = 0
            last_pos = actual
            time.sleep(0.1)

    def stop(self):
        if self.socket: self.socket.close()


class DataLogger:
    """Handles folder creation, Waypoint caching, and CSV writing."""
    def __init__(self, has_camera=False):
        self.is_recording = False
        self.is_paused = False
        self.has_camera = has_camera
        self.waypoints = []
        self.session_dir = ""
        self.img_dir = ""

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.is_paused = False
            self.waypoints = []
            
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_dir = os.path.join(os.getcwd(), f"episode_{now_str}")
            os.makedirs(self.session_dir, exist_ok=True)
            
            self.img_dir = os.path.join(self.session_dir, "images")
            if self.has_camera:
                os.makedirs(self.img_dir, exist_ok=True)
            
            print(f"\n[LOGGER] RECORDING STARTED -> {self.session_dir}\n")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.is_paused = False
            
            csv_path = os.path.join(self.session_dir, "trajectory_data.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ["timestamp", "j0", "j1", "j2", "j3", "j4", "j5",
                           "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", 
                           "gripper_pos", "skill_id", "image_file"]
                writer.writerow(headers)
                writer.writerows(self.waypoints)

            print(f"\n[LOGGER] RECORDING STOPPED. Saved {len(self.waypoints)} waypoints.\n")

    def toggle_pause(self):
        if self.is_recording:
            self.is_paused = not self.is_paused
            state = "PAUSED" if self.is_paused else "RESUMED"
            print(f"[LOGGER] Recording {state}.")

    def record_waypoint(self, q, tcp, gripper_pos, skill_id, camera_handler=None):
        if self.is_paused:
            print("[LOGGER] Warning: Recording is paused.")
            return
        if not self.is_recording:
            self.start_recording()

        ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        img_filename = "None"
        
        if self.has_camera and camera_handler:
            img_filename = f"frame_{len(self.waypoints):04d}_{ts_str}.png"
            if not camera_handler.save_png(self.img_dir, img_filename):
                img_filename = "Error"

        img_rel_path = f"images/{img_filename}" if img_filename not in ["None", "Error"] else img_filename
        row = [ts_str] + q + tcp + [gripper_pos, skill_id, img_rel_path]
        
        self.waypoints.append(row)
        print(f"[LOGGER] Waypoint #{len(self.waypoints)} saved.")

    def undo_last_waypoint(self):
        if self.is_recording and len(self.waypoints) > 0:
            removed_row = self.waypoints.pop()
            img_path = removed_row[-1] # The relative image path
            
            # Cleanup the saved image file if it exists
            if img_path not in ["None", "Error"]:
                full_img_path = os.path.join(self.session_dir, img_path)
                if os.path.exists(full_img_path):
                    os.remove(full_img_path)
                    
            print(f"[LOGGER] UNDO: Removed last waypoint. {len(self.waypoints)} points remain.")
        else:
            print("[LOGGER] UNDO FAILED: No waypoints to remove.")


class DualTeleopController:
    """Main application orchestrating joysticks, robot, and hardware."""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        num_joys = pygame.joystick.get_count()
        if num_joys < 2:
            raise RuntimeError(f"Need 2 joysticks for HOSAS, but found {num_joys}.")
        
        # Identify left/right by GUID (stable across USB re-plug)
        LEFT_GUID  = "03006df94f0400002a04000011010000"
        RIGHT_GUID = "03006df94f0400002204000011010000"

        self.joy_left = None
        self.joy_right = None
        for i in range(num_joys):
            j = pygame.joystick.Joystick(i)
            j.init()
            guid = j.get_guid()
            if guid == LEFT_GUID:
                self.joy_left = j
            elif guid == RIGHT_GUID:
                self.joy_right = j

        if self.joy_left is None or self.joy_right is None:
            # Fallback: print GUIDs and use index order
            print("[INPUT] WARNING: Could not match GUIDs, falling back to index order.")
            print("[INPUT] Expected LEFT_GUID: ", LEFT_GUID)
            print("[INPUT] Expected RIGHT_GUID:", RIGHT_GUID)
            for i in range(num_joys):
                j = pygame.joystick.Joystick(i)
                j.init()
                print(f"[INPUT]   Joy {i}: guid={j.get_guid()}")
            self.joy_left = pygame.joystick.Joystick(0)
            self.joy_right = pygame.joystick.Joystick(1)
            self.joy_left.init()
            self.joy_right.init()

        print(f"[INPUT] LEFT:  {self.joy_left.get_name()} (guid={self.joy_left.get_guid()})")
        print(f"[INPUT] RIGHT: {self.joy_right.get_name()} (guid={self.joy_right.get_guid()})")

        # Capture instance IDs for event handling
        self.left_id = self.joy_left.get_instance_id()
        self.right_id = self.joy_right.get_instance_id()

        print("[INPUT] Calibrating axes — keep hands off sticks...")
        time.sleep(1.0)
        num_samples = 50
        left_accum = np.zeros(max(8, self.joy_left.get_numaxes()))
        right_accum = np.zeros(max(8, self.joy_right.get_numaxes()))
        for _ in range(num_samples):
            pygame.event.pump()
            for ax in range(self.joy_left.get_numaxes()): left_accum[ax] += self.joy_left.get_axis(ax)
            for ax in range(self.joy_right.get_numaxes()): right_accum[ax] += self.joy_right.get_axis(ax)
            time.sleep(0.01)
            
        self._left_center = left_accum / num_samples
        self._right_center = right_accum / num_samples

        print(f"[ROBOT] Connecting to UR5e at {ROBOT_IP}...")
        self.rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("[ROBOT] Connected.")

        self.gripper = GripperHandler()
        if self.gripper.connect(ROBOT_IP, GRIPPER_PORT):
            self.gripper.activate()
            
        self.camera = CameraHandler()
        cam_active = self.camera.connect()
        self.logger = DataLogger(has_camera=cam_active)

        self.keep_running = True
        self.current_skill_id = 0
        self._skill_interrupted = False
        self._interrupted_skill = None  # (name, csv_path, rel_count)

    def _apply_deadzone(self, val):
        return 0.0 if abs(val) < DEADZONE_VAL else val

    def _safe_get_axis(self, joy, axis_idx, calibrate=True):
        """Read a joystick axis with bounds checking and optional calibration."""
        if axis_idx >= joy.get_numaxes(): return 0.0
        raw = joy.get_axis(axis_idx)
        if calibrate:
            center = self._left_center if joy is self.joy_left else self._right_center
            if axis_idx < len(center): raw -= center[axis_idx]
        return raw

    def _get_gain(self):
        # Slider range: 0 (up/fast) to 1 (down/slow) — read raw, no calibration
        val = self._safe_get_axis(self.joy_left, L_AXIS_SLIDER, calibrate=False)
        return MIN_GAIN + ((1.0 - val) * (1.0 - MIN_GAIN))

    def _check_interrupt(self):
        """Check if skill interrupt button (left stick btn 16) was pressed."""
        pygame.event.pump()
        for ev in pygame.event.get():
            if (ev.type == pygame.JOYBUTTONDOWN
                    and ev.instance_id == self.left_id
                    and ev.button == L_BTN_INTERRUPT):
                return True
        return False

    def execute_skill_reorient(self):
        print("[SKILL] Orienting gripper 90 degrees vertically downwards...")
        self.rtde_c.speedStop()
        current_pose = self.rtde_r.getActualTCPPose()
        # Force rotation to point straight down (Check your specific tool coordinate system)
        current_pose[3:] = [3.14159, 0.0, 0.0] 
        self.rtde_c.moveL(current_pose, speed=0.1, acceleration=0.2)

    def execute_skill_home(self):
        print("[SKILL] Returning to Home position...")
        self.rtde_c.speedStop()
        self.rtde_c.moveJ(HOME_JOINTS, speed=0.5, acceleration=0.3)
        if self.gripper.connected:
            self.gripper.move(HOME_GRIPPER_POS)

    def _execute_skill_csv(self, name, csv_path, rel_count, resume_absolute_only=False):
        """Execute a skill from CSV with relative/absolute waypoints.

        If resume_absolute_only=True, skip relative waypoints and execute only
        the absolute (base-frame) portion of the skill.
        """
        if not os.path.exists(csv_path):
            print(f"[SKILL] CSV not found: {csv_path}")
            return

        print(f"[SKILL] Loading {name} skill from {csv_path}...")
        waypoints = load_skill_csv(csv_path)
        if not waypoints:
            print("[SKILL] No waypoints found in CSV.")
            return

        total = len(waypoints)

        # Stop current motion
        self.rtde_c.speedStop()
        time.sleep(0.1)

        T_offset = None
        if resume_absolute_only:
            start_idx = rel_count
            print(
                f"[SKILL] RESUMING {name} skill (absolute only): "
                f"waypoints {rel_count + 1}-{total}"
            )
        else:
            start_idx = 0
            # Capture trigger TCP pose and compute transform
            trigger_tcp = self.rtde_r.getActualTCPPose()
            T_trigger = pose6d_to_homogeneous(trigger_tcp)
            T_skill_origin = pose6d_to_homogeneous(waypoints[0][1])
            T_offset = T_trigger @ np.linalg.inv(T_skill_origin)
            print(
                f"[SKILL] Executing {name} skill: {total} waypoints "
                f"({rel_count} relative + {total - rel_count} absolute)"
            )

        # Slow down gripper for smooth motion during skill
        if self.gripper.connected:
            self.gripper.set_speed(SKILL_GRIPPER_SPEED)
            print(f"[SKILL] Gripper speed set to {SKILL_GRIPPER_SPEED}/255")

        interrupted = False
        for i in range(start_idx, total):
            # Check for interrupt between waypoints
            if self._check_interrupt():
                interrupted = True
                print(f"\n[SKILL] *** INTERRUPTED before waypoint {i + 1}/{total} ***")
                break

            joints, tcp, gripper = waypoints[i]
            if i < rel_count and T_offset is not None:
                # Relative: transform waypoint to trigger frame
                T_wp = pose6d_to_homogeneous(tcp)
                T_target = T_offset @ T_wp
                target_pose = homogeneous_to_pose6d(T_target)
            else:
                # Absolute: use base-frame position directly
                target_pose = list(tcp)

            # Execute moveL (blocking)
            try:
                self.rtde_c.moveL(target_pose, speed=SKILL_MOVE_SPEED, acceleration=SKILL_MOVE_ACCEL)
            except Exception as e:
                print(f"[SKILL] moveL failed at waypoint {i}: {e}")
                break

            # Set gripper position (block until gripper finishes before next moveL)
            if self.gripper.connected:
                self.gripper.move_and_wait(gripper)

            if (i + 1) % 5 == 0 or i == total - 1:
                print(
                    f"[SKILL] Waypoint {i + 1}/{total} complete "
                    f"(gripper={gripper})"
                )

        # Restore full gripper speed for manual control
        if self.gripper.connected:
            self.gripper.set_speed(255)
            print("[SKILL] Gripper speed restored to 255")

        if interrupted:
            self._skill_interrupted = True
            self._interrupted_skill = (name, csv_path, rel_count)
            self.rtde_c.speedStop()
            print(
                "[SKILL] Manual teleop active. Press skill trigger again "
                "to resume (absolute waypoints only)."
            )
        else:
            self._skill_interrupted = False
            self._interrupted_skill = None
            print(f"[SKILL] {name} skill execution complete.")

    def trigger_blank_skill(self, skill_id):
        """Dispatch skill execution based on button ID."""
        if self._skill_interrupted and self._interrupted_skill is not None:
            # Resume interrupted skill — skip relative, execute absolute only
            name, csv_path, rel_count = self._interrupted_skill
            self._skill_interrupted = False
            self._interrupted_skill = None
            self._execute_skill_csv(name, csv_path, rel_count, resume_absolute_only=True)
        elif skill_id == 15:
            self._execute_skill_csv("CPU", CPU_SKILL_CSV, CPU_RELATIVE_COUNT)
        elif skill_id == 16:
            self._execute_skill_csv("RAM", RAM_SKILL_CSV, RAM_RELATIVE_COUNT)
        else:
            print(f"[SKILL] Skill {skill_id} not implemented yet. Staying at current position.")
        self.current_skill_id = skill_id

    def run(self):
        print("\n--- SYSTEM READY ---")
        print("LEFT HAND                      RIGHT HAND")
        print("  Stick X/Y  -> TCP X/Y          Stick Y    -> TCP Z")
        print("  Slider     -> Speed gain        Twist(ax5) -> TCP Rz")
        print("  Mini Y(ax4)-> Gripper           Mini X(ax3)-> TCP Rx")
        print("                                  Mini Y(ax4)-> TCP Ry")
        print("  Btn 26/27  -> Start record      Btn 26/27  -> Stop record")
        print("  Btn 24/25  -> Save waypoint     Btn 16     -> Undo waypoint")
        print("  Btn 35     -> Home              Btn 35     -> Pause")
        print("  Btn 17     -> Interrupt skill    Btn 5-8    -> Skills")
        print("  Btn 39     -> Vertical orient")
        print("--------------------\n")
        try:
            while self.keep_running:
                start_time = time.time()

                if not self.rtde_c.isConnected():
                    print("[ERROR] Robot disconnected.")
                    break

                self.camera.update_frame()
                if self.camera.active and self.camera.current_frame is not None:
                    cv2.imshow('VLA Collection Feed', cv2.resize(self.camera.current_frame, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.keep_running = False

                # ==========================================================
                # DISCRETE EVENT HANDLING (Buttons & Menus)
                # ==========================================================
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.keep_running = False
                    
                    elif event.type == pygame.JOYBUTTONDOWN:
                        joy_id = event.instance_id
                        btn = event.button

                        # --- LEFT HAND COMMANDS ---
                        if joy_id == self.left_id:
                            if btn == L_BTN_REC_START:
                                self.logger.start_recording()
                            elif btn == L_BTN_WAYPOINT:
                                q = self.rtde_r.getActualQ()
                                tcp = self.rtde_r.getActualTCPPose()
                                self.logger.record_waypoint(q, tcp, self.gripper.get_actual_pos(), self.current_skill_id, self.camera)
                            elif btn == L_BTN_HOME:
                                self.execute_skill_home()
                            elif btn == L_BTN_VERT:
                                self.execute_skill_reorient()

                        # --- RIGHT HAND COMMANDS ---
                        elif joy_id == self.right_id:
                            if btn == R_BTN_REC_STOP:
                                self.logger.stop_recording()
                            elif btn == R_BTN_UNDO:
                                self.logger.undo_last_waypoint()
                            elif btn == R_BTN_PAUSE:
                                self.logger.toggle_pause()
                            elif btn in R_BTN_SKILLS:
                                self.trigger_blank_skill(btn)

                # ==========================================================
                # CONTINUOUS TELEOPERATION (SpeedL)
                # ==========================================================
                gain = self._get_gain()

                # Translation (Left Stick X/Y, Right Stick Push/Pull Z)
                val_x = self._apply_deadzone(self._safe_get_axis(self.joy_left, L_AXIS_X))
                val_y = self._apply_deadzone(self._safe_get_axis(self.joy_left, L_AXIS_Y))
                val_z = self._apply_deadzone(self._safe_get_axis(self.joy_right, R_AXIS_Z))

                # Rotation (Right Mini-stick Rx/Ry, Right Twist Rz)
                val_rx = self._apply_deadzone(self._safe_get_axis(self.joy_right, R_AXIS_MINI_X))
                val_ry = self._apply_deadzone(self._safe_get_axis(self.joy_right, R_AXIS_MINI_Y))
                val_rz = self._apply_deadzone(self._safe_get_axis(self.joy_right, R_AXIS_RZ))

                target_vel = [0.0] * 6

                # Apply velocities if not paused
                if not self.logger.is_paused:
                    target_vel[0] = val_x * MAX_SPEED_L * gain
                    target_vel[1] = -val_y * MAX_SPEED_L * gain
                    target_vel[2] = -val_z * MAX_SPEED_L * gain

                    target_vel[3] = val_ry * MAX_SPEED_R * gain
                    target_vel[4] = -val_rx * MAX_SPEED_R * gain
                    target_vel[5] = -val_rz * MAX_SPEED_RZ * gain

                self.rtde_c.speedL(target_vel, ACCELERATION, time=WATCHDOG_TIME)

                # ==========================================================
                # GRIPPER CONTROL (Left Mini-stick up/down)
                # ==========================================================
                if self.gripper.connected:
                    grip_val = self._apply_deadzone(self._safe_get_axis(self.joy_left, L_AXIS_MINI_Y))
                    if grip_val != 0:
                        new_pos = self.gripper.current_pos + (-grip_val * GRIPPER_STEP)
                        self.gripper.move(new_pos)

                # ==========================================================
                # LOOP TIMING
                # ==========================================================
                elapsed = time.time() - start_time
                if elapsed < DT:
                    time.sleep(DT - elapsed)

        except KeyboardInterrupt:
            print("\n[SYSTEM] Manual termination requested.")
        finally:
            self.shutdown()

    def shutdown(self):
        print("[SYSTEM] Cleaning up resources...")
        try:
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
            if self.logger.is_recording:
                self.logger.stop_recording()
            self.gripper.stop()
            self.camera.stop()
            if VISION_AVAILABLE: cv2.destroyAllWindows()
        except Exception as e:
            print(f"[SYSTEM] Cleanup warning: {e}")
        pygame.quit()
        print("[SYSTEM] Exited cleanly.")

if __name__ == "__main__":
    controller = DualTeleopController()
    controller.run()