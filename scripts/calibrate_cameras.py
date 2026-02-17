#!/usr/bin/env python3
"""Calibrate camera settings for reproducible VLA data collection.

Two modes:

  Auto mode (--auto):
    Opens both cameras with auto-exposure/WB, waits for stabilization
    (default 20s), snapshots settings, saves to JSON, and exits.
    No GUI or keyboard interaction needed.

  Manual mode (default):
    Shows a live 2x2 preview (RGB + depth for each camera).
    Press SPACE to snapshot settings and save to JSON.
    Press ESC to exit without saving.

Usage:
    python scripts/calibrate_cameras.py --auto
    python scripts/calibrate_cameras.py --auto --warmup 30
    python scripts/calibrate_cameras.py                      # manual mode
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def _snapshot_and_save(rs_cam, oak_cam, out_path: str) -> None:
    """Snapshot current settings from both cameras and save to JSON."""
    rs_settings = rs_cam.get_settings()
    oak_settings = oak_cam.get_settings()

    config = {
        "timestamp": datetime.now().isoformat(),
        "realsense": rs_settings,
        "oakd": oak_settings,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nSaved camera settings to {out_path}")
    print(f"\nRealSense settings:")
    for k, v in rs_settings.items():
        print(f"  {k}: {v}")
    print(f"\nOAK-D settings:")
    for k, v in oak_settings.items():
        print(f"  {k}: {v}")
    print()


def _init_cameras():
    """Initialize both cameras and return (rs_cam, oak_cam)."""
    from gello.cameras.oakd_camera import OAKDCamera
    from gello.cameras.oakd_camera import get_device_ids as oak_ids
    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.cameras.realsense_camera import get_device_ids as rs_ids

    rs_devices = rs_ids()
    oak_devices = oak_ids()
    print(f"RealSense devices: {rs_devices}")
    print(f"OAK-D devices:     {oak_devices}")

    if not rs_devices:
        print("ERROR: No RealSense device found.")
        sys.exit(1)
    if not oak_devices:
        print("ERROR: No OAK-D device found.")
        sys.exit(1)

    print("\nInitializing RealSense...")
    rs_cam = RealSenseCamera(device_id=rs_devices[0])

    print("Initializing OAK-D...")
    oak_cam = OAKDCamera(device_id=oak_devices[0], fps=30)

    return rs_cam, oak_cam


def auto_calibrate(warmup_seconds: float, out_path: str) -> None:
    """Auto-calibrate: open cameras, wait for auto-settings to stabilize, save."""
    rs_cam, oak_cam = _init_cameras()

    print(f"\nAuto-calibration: waiting {warmup_seconds:.0f}s for auto-exposure/WB to stabilize...")
    print("(Both cameras running with AUTO exposure, white balance, gain)\n")

    t0 = time.time()
    frame_count = 0
    while time.time() - t0 < warmup_seconds:
        rs_cam.read()
        oak_cam.read()
        frame_count += 1
        elapsed = time.time() - t0
        remaining = warmup_seconds - elapsed
        if frame_count % 30 == 0:
            print(f"  Stabilizing... {remaining:.0f}s remaining", flush=True)

    print(f"\nWarmup complete ({warmup_seconds:.0f}s, {frame_count} frames read).")
    _snapshot_and_save(rs_cam, oak_cam, out_path)
    print("Auto-calibration done. Use this file with:")
    print(f"  python experiments/launch_camera_nodes.py --camera-settings {out_path}")


def manual_calibrate(out_path: str) -> None:
    """Manual calibrate: live preview, spacebar to snapshot."""
    import cv2
    import numpy as np

    rs_cam, oak_cam = _init_cameras()

    # Short warmup
    print("\nWarming up cameras (auto-exposure/WB stabilizing)...")
    warmup_start = time.time()
    warmup_seconds = 3.0
    while time.time() - warmup_start < warmup_seconds:
        rs_cam.read()
        oak_cam.read()
    print(f"Warmup complete ({warmup_seconds:.0f}s).\n")

    print("--- Live Preview ---")
    print("Press SPACE to snapshot settings and save to JSON.")
    print("Press ESC to exit without saving.\n")

    frame_count = 0
    t_fps = time.time()

    while True:
        rs_rgb, rs_depth = rs_cam.read()
        oak_rgb, oak_depth = oak_cam.read()

        # Convert for display
        rs_bgr = rs_rgb[:, :, ::-1]
        oak_bgr = oak_rgb[:, :, ::-1]

        # Depth colormaps
        rs_depth_viz = cv2.applyColorMap(
            cv2.normalize(rs_depth.squeeze(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET,
        )
        oak_depth_viz = cv2.applyColorMap(
            cv2.normalize(
                oak_depth.squeeze(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            ),
            cv2.COLORMAP_JET,
        )

        # Resize for display
        disp_w, disp_h = 640, 360
        rs_bgr_s = cv2.resize(rs_bgr, (disp_w, disp_h))
        oak_bgr_s = cv2.resize(oak_bgr, (disp_w, disp_h))
        rs_dep_s = cv2.resize(rs_depth_viz, (disp_w, disp_h))
        oak_dep_s = cv2.resize(oak_depth_viz, (disp_w, disp_h))

        # Labels
        cv2.putText(
            rs_bgr_s,
            "RealSense RGB",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            oak_bgr_s,
            "OAK-D RGB",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            rs_dep_s,
            "RealSense Depth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            oak_dep_s,
            "OAK-D Depth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        top_row = np.hstack([rs_bgr_s, oak_bgr_s])
        bot_row = np.hstack([rs_dep_s, oak_dep_s])
        canvas = np.vstack([top_row, bot_row])

        # FPS counter
        frame_count += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps = frame_count / (now - t_fps)
            cv2.setWindowTitle(
                "Camera Calibration", f"Camera Calibration  |  {fps:.1f} Hz"
            )
            frame_count = 0
            t_fps = now

        cv2.imshow("Camera Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exiting without saving.")
            break
        elif key == ord(" "):  # SPACE
            print("Snapshotting camera settings...")
            _snapshot_and_save(rs_cam, oak_cam, out_path)
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera settings for VLA data collection."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto mode: wait for stabilization and save without GUI.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=20.0,
        help="Warmup time in seconds for auto mode (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/camera_settings.json",
        help="Output path for the settings JSON file.",
    )
    args = parser.parse_args()

    if args.auto:
        auto_calibrate(args.warmup, args.output)
    else:
        manual_calibrate(args.output)


if __name__ == "__main__":
    main()
