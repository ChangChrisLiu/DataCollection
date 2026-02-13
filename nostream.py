#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time
import os
from datetime import datetime
import open3d as o3d
import threading
from typing import Dict, List, Tuple, Optional
import json

class OakDProCamera:
    """Single OAK-D Pro camera handler - fixed version"""
    def __init__(self, device_info: dai.DeviceInfo, camera_id: str, save_dir_base: str):
        self.device_info = device_info
        self.camera_id = camera_id
        self.save_dir = os.path.join(save_dir_base, camera_id)
        self.mxid = device_info.getMxId()
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Resolution settings
        self.preview_width = 1280
        self.preview_height = 800
        self.mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
        self.has_autofocus = True
        
        # Device and pipeline
        self.device = None
        self.pipeline = None
        self.queues = {}
        self.intrinsics_800p = None
        self.camRgb_node = None
        
        # 3A stabilization tracking
        self.connection_time = None
        self.last_capture_time = None
        self.warmup_time = 3.0
        self.maintenance_interval = 30.0
        
        # Check capabilities
        self.check_device_capabilities()
        
    def check_device_capabilities(self):
        """Quick check of device capabilities"""
        try:
            temp_pipeline = dai.Pipeline()
            # Use new API with UsbSpeed
            with dai.Device(temp_pipeline, self.device_info, dai.UsbSpeed.SUPER) as device:
                cam_features = device.getConnectedCameraFeatures()
                
                mono_supports_800p = False
                self.has_autofocus = False
                
                for cam in cam_features:
                    # Use new socket names
                    if cam.socket == dai.CameraBoardSocket.CAM_A:  # RGB camera
                        self.has_autofocus = cam.hasAutofocus
                        
                    if cam.socket in [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]:  # Mono cameras
                        if "OV9282" in cam.sensorName or "OV9782" in cam.sensorName:
                            mono_supports_800p = True
                
                if not mono_supports_800p:
                    print(f"⚠️ Mono cameras don't support 800P, using 720P")
                    self.mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
                    self.preview_height = 720
                    
        except Exception as e:
            print(f"Warning: Could not check device capabilities: {e}")
            self.mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
            self.preview_height = 720
    
    def create_pipeline(self) -> dai.Pipeline:
        """Create minimal pipeline for capture only"""
        pipeline = dai.Pipeline()
        
        # RGB camera
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(self.preview_width, self.preview_height)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A instead of RGB
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(30)
        
        # Set up 3A
        camRgb.initialControl.setAutoExposureEnable()
        camRgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        if self.has_autofocus:
            camRgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        
        # Don't use setIsp3aFps as it's unstable
        # camRgb.setIsp3aFps(15)  # Removed due to deprecation warning
        
        self.camRgb_node = camRgb
        
        # Still encoder for 12MP
        stillEncoder = pipeline.create(dai.node.VideoEncoder)
        stillEncoder.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
        stillEncoder.setQuality(95)
        stillEncoder.setNumFramesPool(1)
        camRgb.still.link(stillEncoder.input)
        
        # Mono cameras
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(self.mono_resolution)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Use CAM_B instead of LEFT
        monoRight.setResolution(self.mono_resolution)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Use CAM_C instead of RIGHT
        monoLeft.setNumFramesPool(2)
        monoRight.setNumFramesPool(2)
        
        # Depth
        depth = pipeline.create(dai.node.StereoDepth)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # Use DEFAULT instead of HIGH_DENSITY
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(True)
        depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to CAM_A
        
        # IMPORTANT: Set output size to ensure width is multiple of 16
        # For 1280x800, both dimensions are already good
        # But we need to explicitly set it to avoid the 4056 width issue
        depth.setOutputSize(self.preview_width, self.preview_height)
        
        depth.setNumFramesPool(2)
        
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        
        # Outputs with minimal queues
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.preview.link(xoutRgb.input)
        
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        depth.depth.link(xoutDepth.input)
        
        xoutStill = pipeline.create(dai.node.XLinkOut)
        xoutStill.setStreamName("still")
        stillEncoder.bitstream.link(xoutStill.input)
        
        xinControl = pipeline.create(dai.node.XLinkIn)
        xinControl.setStreamName("control")
        xinControl.out.link(camRgb.inputControl)
        
        return pipeline
    
    def connect(self) -> bool:
        """Connect to camera and maintain connection"""
        try:
            if self.device is not None:
                print(f"Camera {self.camera_id} already connected")
                return True
                
            self.pipeline = self.create_pipeline()
            # Use new API with UsbSpeed
            self.device = dai.Device(self.pipeline, self.device_info, dai.UsbSpeed.SUPER)
            
            # Get queues with minimal size
            self.queues['rgb'] = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
            self.queues['depth'] = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            # Still queue should allow more frames and not block
            self.queues['still'] = self.device.getOutputQueue("still", maxSize=3, blocking=False)
            self.queues['control'] = self.device.getInputQueue("control")
            
            # Get intrinsics
            calibData = self.device.readCalibration()
            try:
                self.intrinsics_800p = np.array(calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A,  # Use CAM_A
                    self.preview_width, 
                    self.preview_height
                ))
            except RuntimeError:
                fx = fy = self.preview_width * 0.8
                cx = self.preview_width / 2
                cy = self.preview_height / 2
                self.intrinsics_800p = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            self.connection_time = time.time()
            print(f"✓ Connected to camera {self.camera_id}")
            
            # Initial warmup
            print(f"  ⏳ Initial 3A warmup ({self.warmup_time}s)...")
            self._maintain_3a_active(self.warmup_time)
            print(f"  ✓ Camera {self.camera_id} ready")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to camera {self.camera_id}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.device:
            try:
                self.device.close()
            except:
                pass
            self.device = None
            self.pipeline = None
            self.queues = {}
            print(f"✓ Disconnected camera {self.camera_id}")
    
    def _maintain_3a_active(self, duration: float = 1.0):
        """Keep 3A algorithms active by reading frames"""
        start_time = time.time()
        errors = 0
        max_errors = 5
        
        while time.time() - start_time < duration and errors < max_errors:
            try:
                # Read and discard frames to keep 3A active
                if 'rgb' in self.queues and self.queues['rgb']:
                    self.queues['rgb'].tryGet()
                if 'depth' in self.queues and self.queues['depth']:
                    self.queues['depth'].tryGet()
                time.sleep(0.05)  # 20 FPS reading
                errors = 0  # Reset error count on success
            except RuntimeError as e:
                errors += 1
                print(f"  ⚠️ Error in 3A maintenance: {e}")
                if errors >= max_errors:
                    print(f"  ✗ Too many errors in 3A maintenance")
                    break
                time.sleep(0.1)
    
    def _check_3a_maintenance(self):
        """Check if we need to run maintenance to keep 3A active"""
        if self.last_capture_time is None:
            return
            
        time_since_capture = time.time() - self.last_capture_time
        if time_since_capture > self.maintenance_interval:
            print(f"  🔄 Running 3A maintenance for {self.camera_id}...")
            self._maintain_3a_active(1.0)
            self.last_capture_time = time.time()
    
    def capture_data(self, timestamp: str) -> bool:
        """Capture all data"""
        if not self.device:
            print(f"✗ Camera {self.camera_id} not connected")
            return False
        
        print(f"\n📷 Capturing from {self.camera_id}...")
        
        try:
            # Check if 3A needs maintenance
            self._check_3a_maintenance()
            
            # Brief 3A stabilization
            self._maintain_3a_active(0.5)
            
            # Clear any stale frames
            for _ in range(3):
                try:
                    self.queues['rgb'].tryGet()
                    self.queues['depth'].tryGet()
                except:
                    pass
            
            # Get fresh frames
            rgb_frame = None
            depth_frame = None
            
            # Try to get synchronized frames
            for _ in range(10):
                try:
                    in_rgb = self.queues['rgb'].tryGet()
                    in_depth = self.queues['depth'].tryGet()
                    
                    if in_rgb is not None:
                        # Get the frame and keep it in RGB format for saving
                        rgb_frame_raw = in_rgb.getCvFrame()
                        # Since we set ColorOrder.RGB, the frame is already in RGB
                        # For saving, we'll keep it as RGB (PNG supports RGB)
                        rgb_frame = rgb_frame_raw  # Keep as RGB
                        
                    if in_depth is not None:
                        # Get raw depth values in millimeters (uint16)
                        depth_frame = in_depth.getFrame()
                        
                    if rgb_frame is not None and depth_frame is not None:
                        break
                except RuntimeError as e:
                    print(f"  ⚠️ Error getting frames: {e}")
                time.sleep(0.05)
            
            # Lock exposure and white balance
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(True)
            ctrl.setAutoWhiteBalanceLock(True)
            self.queues['control'].send(ctrl)
            time.sleep(0.1)
            
            # Capture 12MP still
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            self.queues['control'].send(ctrl)
            
            rgb_12mp = None
            max_wait_time = 5.0
            start_time = time.time()
            
            try:
                # Use tryGet in a loop with timeout
                while time.time() - start_time < max_wait_time:
                    stillPacket = self.queues['still'].tryGet()
                    if stillPacket is not None:
                        stillData = stillPacket.getData()
                        rgb_12mp = cv2.imdecode(stillData, cv2.IMREAD_COLOR)
                        if rgb_12mp is not None:
                            break
                    time.sleep(0.1)  # Small delay between checks
                
                if rgb_12mp is None:
                    print(f"  ⚠️ Timeout waiting for 12MP capture")
                    
            except Exception as e:
                print(f"✗ Error capturing still: {e}")
            
            # Unlock exposure
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(False)
            ctrl.setAutoWhiteBalanceLock(False)
            self.queues['control'].send(ctrl)
            
            # Update last capture time
            self.last_capture_time = time.time()
            
            # Save all data
            success = True
            
            if rgb_12mp is not None:
                filename_12mp = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_rgb_12mp.jpg")
                cv2.imwrite(filename_12mp, rgb_12mp, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  ✓ 12MP RGB saved")
            else:
                print(f"  ✗ Failed to capture 12MP image")
                success = False
            
            if rgb_frame is not None:
                # Save RGB frame (keeping original RGB format)
                filename_rgb = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_rgb_aligned.png")
                # Convert to BGR only for OpenCV saving (OpenCV expects BGR)
                rgb_bgr_for_save = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename_rgb, rgb_bgr_for_save)
                print(f"  ✓ Aligned RGB saved")
            
            if depth_frame is not None:
                # Save raw depth data as 16-bit PNG (lossless, preserves exact depth values)
                filename_depth_png = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_depth_raw.png")
                cv2.imwrite(filename_depth_png, depth_frame)  # Saves as 16-bit grayscale PNG
                
                # Also save as numpy array for easy loading
                filename_depth_npy = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_depth_raw.npy")
                np.save(filename_depth_npy, depth_frame)
                
                # Print depth statistics
                valid_depths = depth_frame[depth_frame > 0]
                if len(valid_depths) > 0:
                    print(f"  ✓ Depth data saved (range: {np.min(valid_depths)}-{np.max(valid_depths)}mm)")
                else:
                    print(f"  ✓ Depth data saved (no valid depths)")
                
                # Create visualization for checking (but don't save unless requested)
                if False:  # Set to True if you want to save depth visualization
                    depth_vis = self.visualize_depth(depth_frame)
                    filename_depth_vis = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_depth_visualization.jpg")
                    cv2.imwrite(filename_depth_vis, depth_vis)
                
                if rgb_frame is not None and self.intrinsics_800p is not None:
                    # For point cloud, we need BGR format
                    rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    pcd = self.create_point_cloud(rgb_bgr, depth_frame)
                    if pcd:
                        filename_pcd = os.path.join(self.save_dir, f"{timestamp}_{self.camera_id}_pointcloud.ply")
                        o3d.io.write_point_cloud(filename_pcd, pcd, write_ascii=False)
                        print(f"  ✓ Point cloud saved ({len(pcd.points)} points)")
            
            return success
            
        except Exception as e:
            print(f"✗ Unexpected error during capture: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_point_cloud(self, rgb_bgr: np.ndarray, depth: np.ndarray) -> Optional[o3d.geometry.PointCloud]:
        """Create point cloud from RGB and depth"""
        if self.intrinsics_800p is None:
            return None
        
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        
        o3d_rgb = o3d.geometry.Image(rgb_rgb)
        o3d_depth = o3d.geometry.Image(depth)
        
        fx, fy = self.intrinsics_800p[0, 0], self.intrinsics_800p[1, 1]
        cx, cy = self.intrinsics_800p[0, 2], self.intrinsics_800p[1, 2]
        
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb_bgr.shape[1], height=rgb_bgr.shape[0],
            fx=fx, fy=fy, cx=cx, cy=cy
        )
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        return pcd
    
    def visualize_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create a visualization of depth data (for debugging only)"""
        # Normalize depth for visualization
        depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return depth_colored


class PersistentDualOakDPro:
    """Dual camera system with persistent connections"""
    def __init__(self):
        self.save_dir_base = "persistent_oak_captures"
        if not os.path.exists(self.save_dir_base):
            os.makedirs(self.save_dir_base)
        
        self.cameras = {}
        self.maintenance_thread = None
        self.maintenance_running = False
        
    def find_and_connect_cameras(self) -> bool:
        """Find and connect to all cameras"""
        device_infos = dai.Device.getAllAvailableDevices()
        if not device_infos:
            print("✗ No OAK devices found!")
            return False
        
        print(f"\nFound {len(device_infos)} OAK device(s):")
        for i, info in enumerate(device_infos):
            print(f"  {i+1}. MxId: {info.getMxId()} | State: {info.state}")
        
        # Create and connect cameras
        num_cameras = min(2, len(device_infos))
        for i in range(num_cameras):
            camera_id = f"camera_{i+1}"
            camera = OakDProCamera(device_infos[i], camera_id, self.save_dir_base)
            
            if camera.connect():
                self.cameras[camera_id] = camera
            else:
                print(f"✗ Failed to initialize {camera_id}")
                # Disconnect any already connected cameras
                self.disconnect_all_cameras()
                return False
        
        print(f"\n✓ Successfully connected {len(self.cameras)} camera(s)")
        print("✓ Cameras will maintain stable 3A state")
        
        # Start maintenance thread
        self.start_maintenance_thread()
        
        return True
    
    def start_maintenance_thread(self):
        """Start thread to maintain 3A state"""
        self.maintenance_running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()
        print("✓ 3A maintenance thread started")
    
    def stop_maintenance_thread(self):
        """Stop maintenance thread"""
        self.maintenance_running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=2)
        print("✓ 3A maintenance thread stopped")
    
    def _maintenance_loop(self):
        """Background thread to keep 3A active"""
        while self.maintenance_running:
            try:
                for camera_id, camera in self.cameras.items():
                    if camera.device:
                        camera._check_3a_maintenance()
            except Exception as e:
                print(f"Error in maintenance loop: {e}")
            time.sleep(5)  # Check every 5 seconds
    
    def disconnect_all_cameras(self):
        """Disconnect all cameras"""
        self.stop_maintenance_thread()
        for camera in self.cameras.values():
            camera.disconnect()
        self.cameras.clear()
    
    def capture_synchronized(self):
        """Capture from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f"\n{'='*50}")
        print(f"Synchronized capture: {timestamp}")
        print(f"{'='*50}")
        
        # Capture from each camera in separate threads
        threads = []
        results = {}
        
        def capture_camera(cam_id: str, cam: OakDProCamera):
            results[cam_id] = cam.capture_data(timestamp)
        
        for camera_id, camera in self.cameras.items():
            thread = threading.Thread(target=capture_camera, args=(camera_id, camera))
            threads.append(thread)
            thread.start()
        
        # Wait for all captures
        for thread in threads:
            thread.join()
        
        # Report results
        print(f"\nCapture Summary:")
        for cam_id, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {cam_id}: {status}")
        print(f"Files saved to: {self.save_dir_base}")
        print(f"{'='*50}\n")
    
    def save_calibration_data(self):
        """Save calibration data"""
        calib_data = {}
        for cam_id, cam in self.cameras.items():
            if cam.intrinsics_800p is not None:
                calib_data[cam_id] = {
                    'mxid': cam.mxid,
                    'intrinsics': cam.intrinsics_800p.tolist(),
                    'resolution': f'{cam.preview_width}x{cam.preview_height}',
                    'has_autofocus': cam.has_autofocus
                }
        
        if calib_data:
            calib_file = os.path.join(self.save_dir_base, "calibration.json")
            with open(calib_file, 'w') as f:
                json.dump(calib_data, f, indent=2)
            print(f"✓ Saved calibration to {calib_file}")
    
    def run_interactive(self):
        """Run in interactive mode"""
        if not self.find_and_connect_cameras():
            return
        
        self.save_calibration_data()
        
        print("\n🚀 OAK-D Pro Persistent Connection System Ready!")
        print("=" * 50)
        print("Cameras remain connected with stable 3A state")
        print("No live streaming - minimal resource usage")
        print("=" * 50)
        print("\nCommands:")
        print("  [c] - Capture from all cameras")
        print("  [1] - Capture from camera 1 only")
        print("  [2] - Capture from camera 2 only")
        print("  [s] - Show camera status")
        print("  [q] - Quit")
        print("=" * 50)
        
        try:
            while True:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'q':
                    print("Shutting down...")
                    break
                elif command == 'c':
                    self.capture_synchronized()
                elif command == '1' and 'camera_1' in self.cameras:
                    print("\nCapturing from camera_1...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    self.cameras['camera_1'].capture_data(timestamp)
                elif command == '2' and 'camera_2' in self.cameras:
                    print("\nCapturing from camera_2...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    self.cameras['camera_2'].capture_data(timestamp)
                elif command == 's':
                    print("\nCamera Status:")
                    for cam_id, cam in self.cameras.items():
                        if cam.device:
                            uptime = time.time() - cam.connection_time
                            last_capture = "Never"
                            if cam.last_capture_time:
                                last_capture = f"{time.time() - cam.last_capture_time:.1f}s ago"
                            print(f"  {cam_id}: Connected (uptime: {uptime:.1f}s, last capture: {last_capture})")
                        else:
                            print(f"  {cam_id}: Not connected")
                else:
                    print("Invalid command. Use 'c', '1', '2', 's', or 'q'")
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            self.disconnect_all_cameras()
            print("\n✓ System shutdown complete")
    
    def run_batch(self, num_captures: int = 10, interval: float = 5.0):
        """Run in batch mode"""
        if not self.find_and_connect_cameras():
            return
        
        self.save_calibration_data()
        
        print(f"\n🚀 Batch Capture Mode")
        print(f"Will capture {num_captures} times with {interval}s interval")
        print("Cameras maintain connection throughout")
        print("Press Ctrl+C to stop early\n")
        
        try:
            for i in range(num_captures):
                print(f"\nCapture {i+1}/{num_captures}")
                self.capture_synchronized()
                
                if i < num_captures - 1:
                    print(f"Waiting {interval}s until next capture...")
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            print("\n\nBatch capture interrupted")
        finally:
            self.disconnect_all_cameras()
        
        print("\n✓ Batch capture complete")


if __name__ == "__main__":
    import sys
    
    controller = PersistentDualOakDPro()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch mode
        num_captures = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        controller.run_batch(num_captures, interval)
    else:
        # Interactive mode
        controller.run_interactive()


