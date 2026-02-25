#!/usr/bin/env python3
"""Quick dual-camera stream test: RealSense + OAK-D side-by-side at 1280x720 @ 30Hz.

Usage:
    python scripts/test_dual_camera.py

Controls:
    S     - Save current frames to disk (scripts/camera_test_frames/)
    ESC   - Exit
"""

import os
import time

import cv2
import numpy as np


def main():
    # ------------------------------------------------------------------
    # 1. Detect devices
    # ------------------------------------------------------------------
    from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids as rs_ids
    from gello.cameras.oakd_camera import OAKDCamera, get_device_ids as oak_ids

    rs_devices = rs_ids()
    oak_devices = oak_ids()
    print(f"RealSense devices: {rs_devices}")
    print(f"OAK-D devices:     {oak_devices}")

    if not rs_devices:
        print("ERROR: No RealSense device found.")
        return
    if not oak_devices:
        print("ERROR: No OAK-D device found.")
        return

    # ------------------------------------------------------------------
    # 2. Initialize cameras (both 1280x720 @ 30Hz)
    # ------------------------------------------------------------------
    print("\nInitializing RealSense...")
    rs_cam = RealSenseCamera(device_id=rs_devices[0])

    print("Initializing OAK-D...")
    oak_cam = OAKDCamera(device_id=oak_devices[0], fps=30)

    # First reads (verify resolution)
    rs_rgb, rs_depth = rs_cam.read()
    oak_rgb, oak_depth = oak_cam.read()
    print(f"\nRealSense -> RGB: {rs_rgb.shape} {rs_rgb.dtype}, "
          f"Depth: {rs_depth.shape} {rs_depth.dtype}")
    print(f"OAK-D     -> RGB: {oak_rgb.shape} {oak_rgb.dtype}, "
          f"Depth: {oak_depth.shape} {oak_depth.dtype}")

    # ------------------------------------------------------------------
    # 3. Stream loop
    # ------------------------------------------------------------------
    save_dir = "scripts/camera_test_frames"
    save_count = 0
    frame_count = 0
    t_start = time.time()
    t_fps = time.time()

    print("\n--- Streaming (ESC to exit, S to save) ---\n")

    while True:
        rs_rgb, rs_depth = rs_cam.read()
        oak_rgb, oak_depth = oak_cam.read()

        # Convert RGB -> BGR for cv2 display
        rs_bgr = rs_rgb[:, :, ::-1]
        oak_bgr = oak_rgb[:, :, ::-1]

        # Depth colormap (squeeze HxWx1 -> HxW)
        rs_depth_2d = rs_depth.squeeze()
        oak_depth_2d = oak_depth.squeeze()
        rs_depth_viz = cv2.applyColorMap(
            cv2.normalize(rs_depth_2d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET,
        )
        oak_depth_viz = cv2.applyColorMap(
            cv2.normalize(oak_depth_2d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET,
        )

        # Resize for display (half width so side-by-side fits on screen)
        disp_w, disp_h = 640, 360
        rs_bgr_s = cv2.resize(rs_bgr, (disp_w, disp_h))
        oak_bgr_s = cv2.resize(oak_bgr, (disp_w, disp_h))
        rs_dep_s = cv2.resize(rs_depth_viz, (disp_w, disp_h))
        oak_dep_s = cv2.resize(oak_depth_viz, (disp_w, disp_h))

        # Add labels
        cv2.putText(rs_bgr_s, "RealSense RGB", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(oak_bgr_s, "OAK-D RGB", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rs_dep_s, "RealSense Depth", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(oak_dep_s, "OAK-D Depth", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Stack: top row = RGB, bottom row = depth
        top_row = np.hstack([rs_bgr_s, oak_bgr_s])
        bot_row = np.hstack([rs_dep_s, oak_dep_s])
        canvas = np.vstack([top_row, bot_row])

        # FPS counter
        frame_count += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps = frame_count / (now - t_fps)
            cv2.setWindowTitle("Dual Camera Test",
                               f"Dual Camera Test  |  {fps:.1f} Hz")
            frame_count = 0
            t_fps = now

        cv2.imshow("Dual Camera Test", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("s"):
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%H%M%S")
            cv2.imwrite(f"{save_dir}/rs_rgb_{save_count}_{ts}.png", rs_bgr)
            cv2.imwrite(f"{save_dir}/oak_rgb_{save_count}_{ts}.png", oak_bgr)
            np.save(f"{save_dir}/rs_depth_{save_count}_{ts}.npy", rs_depth)
            np.save(f"{save_dir}/oak_depth_{save_count}_{ts}.npy", oak_depth)
            print(f"Saved frame pair #{save_count} to {save_dir}/")
            save_count += 1

    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"\nDone. Ran for {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
